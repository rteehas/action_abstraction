import copy
import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from .main_two_policy_ppo import ABSTRACTION_ACTOR_ROLE, ABSTRACTION_REF_ROLE
from .two_policy_utils import (
    aggregate_child_rewards,
    as_object_array,
    build_scalar_reward_tensor,
    build_single_turn_chat,
    load_template,
    parse_abstraction_output,
    render_template,
)


class TwoPolicyGRPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        abstraction_tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls,
        processor=None,
        abstraction_processor=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )
        if self.use_critic:
            raise ValueError('Two-policy trainer currently supports actor-only training only')
        if self.use_legacy_worker_impl == 'disable':
            raise ValueError('Two-policy trainer currently supports the legacy worker path only')

        self.abstraction_tokenizer = abstraction_tokenizer
        self.abstraction_processor = abstraction_processor
        # Two-policy rollout prompts are sourced from config templates, not from the
        # parquet prompt column that RLHFDataset initially loads for schema/tooling compatibility.
        self.abstraction_prompt_template = load_template(config.two_policy.abstraction_prompt_template_path)
        self.solver_prompt_template = load_template(config.two_policy.solver_prompt_template_path)

        self.abstraction_runtime_config = copy.deepcopy(config)
        self.abstraction_runtime_config.actor_rollout_ref = copy.deepcopy(config.abstraction_actor_rollout_ref)

        self.use_abstraction_reference_policy = need_reference_policy(self.abstraction_runtime_config)
        self.abstraction_ref_in_actor = self._branch_has_internal_ref(config.abstraction_actor_rollout_ref)
        self.abstraction_use_prefix_grouper = config.abstraction_actor_rollout_ref.actor.get(
            'use_prefix_grouper', False
        )

        if self.config.algorithm.use_kl_in_reward:
            self.abstraction_kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm)
        else:
            self.abstraction_kl_ctrl_in_reward = None

        self.abstraction_actor_rollout_wg = None
        self.abstraction_ref_policy_wg = None
        self.abstraction_async_rollout_manager = None
        self.abstraction_checkpoint_manager = None

    @staticmethod
    def _branch_has_internal_ref(branch_cfg) -> bool:
        lora_rank = branch_cfg.model.get('lora', {}).get('rank', 0)
        if lora_rank <= 0:
            lora_rank = branch_cfg.model.get('lora_rank', 0)
        return lora_rank > 0 or branch_cfg.model.get('lora_adapter_path') is not None

    @contextmanager
    def _use_branch(self, branch_cfg, actor_rollout_wg, ref_policy_wg, ref_in_actor, use_reference_policy, use_prefix):
        old_cfg = self.config.actor_rollout_ref
        old_actor_rollout_wg = self.actor_rollout_wg
        old_ref_policy_wg = getattr(self, 'ref_policy_wg', None)
        old_ref_in_actor = self.ref_in_actor
        old_use_reference_policy = self.use_reference_policy
        old_use_prefix_grouper = self.use_prefix_grouper
        self.config.actor_rollout_ref = branch_cfg
        self.actor_rollout_wg = actor_rollout_wg
        self.ref_policy_wg = ref_policy_wg
        self.ref_in_actor = ref_in_actor
        self.use_reference_policy = use_reference_policy
        self.use_prefix_grouper = use_prefix
        try:
            yield
        finally:
            self.config.actor_rollout_ref = old_cfg
            self.actor_rollout_wg = old_actor_rollout_wg
            self.ref_policy_wg = old_ref_policy_wg
            self.ref_in_actor = old_ref_in_actor
            self.use_reference_policy = old_use_reference_policy
            self.use_prefix_grouper = old_use_prefix_grouper

    def init_workers(self):
        super().init_workers()
        self._init_abstraction_workers()

    def _init_abstraction_workers(self):
        resource_pool = self.resource_pool_manager.get_resource_pool(ABSTRACTION_ACTOR_ROLE)
        class_dict = {
            ABSTRACTION_ACTOR_ROLE: RayClassWithInitArgs(
                cls=self.role_worker_mapping[ABSTRACTION_ACTOR_ROLE],
                config=self.config.abstraction_actor_rollout_ref,
                role='actor_rollout',
            )
        }
        if self.use_abstraction_reference_policy and ABSTRACTION_REF_ROLE in self.role_worker_mapping:
            class_dict[ABSTRACTION_REF_ROLE] = RayClassWithInitArgs(
                cls=self.role_worker_mapping[ABSTRACTION_REF_ROLE],
                config=self.config.abstraction_actor_rollout_ref,
                role='ref',
            )

        wg_kwargs = {'device_name': self.device_name}
        if OmegaConf.select(self.config.trainer, 'ray_wait_register_center_timeout') is not None:
            wg_kwargs['ray_wait_register_center_timeout'] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, 'steps') is not None:
            wg_kwargs['profile_steps'] = OmegaConf.select(self.config.global_profiler, 'steps')
            if OmegaConf.select(self.config.global_profiler, 'tool') == 'nsys':
                wg_kwargs['worker_nsight_options'] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, 'worker_nsight_options')
                )

        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = self.ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls,
            **wg_kwargs,
        )
        spawned = wg_dict.spawn(prefix_set=class_dict.keys())
        self.abstraction_actor_rollout_wg = spawned[ABSTRACTION_ACTOR_ROLE]
        self.abstraction_actor_rollout_wg.init_model()

        if self.use_abstraction_reference_policy and not self.abstraction_ref_in_actor:
            self.abstraction_ref_policy_wg = spawned[ABSTRACTION_REF_ROLE]
            self.abstraction_ref_policy_wg.init_model()
        else:
            self.abstraction_ref_policy_wg = self.abstraction_actor_rollout_wg

        manager_class_fqn = self.config.abstraction_actor_rollout_ref.rollout.get('agent', {}).get(
            'agent_loop_manager_class'
        )
        if manager_class_fqn:
            manager_cls = load_class_from_fqn(manager_class_fqn, 'AgentLoopManager')
        else:
            from verl.experimental.agent_loop import AgentLoopManager

            manager_cls = AgentLoopManager

        self.abstraction_async_rollout_manager = manager_cls.create(
            config=self.abstraction_runtime_config,
            worker_group=self.abstraction_actor_rollout_wg,
            rollout_resource_pool=resource_pool,
            reward_loop_worker_handles=None,
        )
        checkpoint_engine_config = omega_conf_to_dataclass(
            self.config.abstraction_actor_rollout_ref.rollout.checkpoint_engine
        )
        self.abstraction_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.abstraction_actor_rollout_wg,
            replicas=self.abstraction_async_rollout_manager.rollout_replicas,
        )
        self.abstraction_checkpoint_manager.sleep_replicas()

    @staticmethod
    def _jsonable_value(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [TwoPolicyGRPOTrainer._jsonable_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): TwoPolicyGRPOTrainer._jsonable_value(item) for key, item in value.items()}
        return value

    @classmethod
    def _serialize_prompt_for_dump(cls, prompt):
        return cls._jsonable_value(prompt)

    def _collect_two_policy_validation_rollouts(
        self, abstraction_batch: DataProto, solver_batch: DataProto | None
    ) -> list[dict]:
        solver_rows_by_abstraction_uid = defaultdict(list)
        if solver_batch is not None and len(solver_batch) > 0:
            if 'solver_uid' in solver_batch.non_tensor_batch:
                solver_uids = solver_batch.non_tensor_batch['solver_uid']
            elif 'uid' in solver_batch.non_tensor_batch:
                # Solver rollouts currently carry the generic `uid` field.
                # Fall back to that so validation dumping works on resumed runs.
                solver_uids = solver_batch.non_tensor_batch['uid']
            else:
                solver_uids = as_object_array(
                    f"{abstraction_uid}:{idx}"
                    for idx, abstraction_uid in enumerate(solver_batch.non_tensor_batch['abstraction_uid'])
                )
            solver_texts = self._decode_responses(solver_batch, self.tokenizer)
            solver_prompts = [
                self._serialize_prompt_for_dump(prompt)
                for prompt in solver_batch.non_tensor_batch['raw_prompt']
            ]
            for idx, abstraction_uid in enumerate(solver_batch.non_tensor_batch['abstraction_uid']):
                row = {
                    'abstraction_uid': abstraction_uid,
                    'solver_uid': solver_uids[idx],
                    'solver_prompt': solver_prompts[idx],
                    'solver_output': solver_texts[idx],
                    'solver_reward': float(solver_batch.non_tensor_batch['seq_final_reward'][idx]),
                }
                if 'acc' in solver_batch.non_tensor_batch:
                    row['acc'] = float(solver_batch.non_tensor_batch['acc'][idx])
                solver_rows_by_abstraction_uid[abstraction_uid].append(row)

        abstraction_texts = self._decode_responses(abstraction_batch, self.abstraction_tokenizer)
        abstraction_prompts = [
            self._serialize_prompt_for_dump(prompt)
            for prompt in abstraction_batch.non_tensor_batch['raw_prompt']
        ]
        batch_sources = abstraction_batch.non_tensor_batch.get('data_source')
        if batch_sources is None:
            batch_sources = np.asarray(['unknown'] * len(abstraction_batch), dtype=object)
        extra_info = abstraction_batch.non_tensor_batch.get('extra_info')
        reward_model = abstraction_batch.non_tensor_batch.get('reward_model')

        entries = []
        for idx, problem_uid in enumerate(abstraction_batch.non_tensor_batch['problem_uid']):
            abstraction_uid = abstraction_batch.non_tensor_batch['abstraction_uid'][idx]
            reward_model_entry = reward_model[idx] if reward_model is not None else None
            entry = {
                'step': self.global_steps,
                'problem_uid': problem_uid,
                'abstraction_uid': abstraction_uid,
                'data_source': self._jsonable_value(batch_sources[idx]),
                'problem': self._jsonable_value(abstraction_batch.non_tensor_batch['problem'][idx]),
                'ground_truth': self._jsonable_value((reward_model_entry or {}).get('ground_truth')),
                'source_extra_info': self._jsonable_value(extra_info[idx]) if extra_info is not None else None,
                'abstraction_prompt': abstraction_prompts[idx],
                'abstraction_output': abstraction_texts[idx],
                'parsed_abstraction': self._jsonable_value(abstraction_batch.non_tensor_batch['parsed_abstraction'][idx]),
                'abstraction_valid': bool(abstraction_batch.non_tensor_batch['abstraction_valid'][idx]),
                'abstraction_validity_score': float(abstraction_batch.non_tensor_batch['abstraction_validity_score'][idx]),
                'abstraction_failure_reason': self._jsonable_value(
                    abstraction_batch.non_tensor_batch['abstraction_failure_reason'][idx]
                ),
                'abstraction_principle_count': int(abstraction_batch.non_tensor_batch['abstraction_principle_count'][idx]),
                'abstraction_reward': float(abstraction_batch.non_tensor_batch['seq_final_reward'][idx]),
                'abstraction_downstream_reward': float(
                    abstraction_batch.non_tensor_batch['abstraction_downstream_reward'][idx]
                ),
                'abstraction_validity_bonus': float(
                    abstraction_batch.non_tensor_batch['abstraction_validity_bonus'][idx]
                ),
                'solver_rollouts': solver_rows_by_abstraction_uid.get(abstraction_uid, []),
            }
            entries.append(entry)
        return entries

    def _dump_two_policy_validation_rollouts(self, entries: list[dict], dump_path: str):
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        with open(filename, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"Dumped two-policy validation rollouts to {filename}")

    def _resolve_resume_folder(self):
        if self.config.trainer.resume_mode == 'disable':
            return None
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)

        if self.config.trainer.resume_mode == 'auto':
            return find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == 'resume_path':
            assert isinstance(self.config.trainer.resume_from_path, str), 'resume ckpt must be str type'
            assert 'global_step_' in self.config.trainer.resume_from_path, 'resume ckpt must specify the global_steps'
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)
            return global_step_folder

        return None

    @staticmethod
    def _branch_uses_async_save(branch_cfg) -> bool:
        checkpoint_cfg = branch_cfg.actor.checkpoint
        return (hasattr(checkpoint_cfg, "async_save") and checkpoint_cfg.async_save) or (
            "async_save" in checkpoint_cfg and checkpoint_cfg["async_save"]
        )

    def _load_checkpoint(self):
        global_step_folder = self._resolve_resume_folder()
        if global_step_folder is None:
            print('Training from scratch')
            self.global_steps = 0
            return 0

        print(f'Load from checkpoint folder: {global_step_folder}')
        self.global_steps = int(global_step_folder.split('global_step_')[-1])
        print(f'Setting global step to {self.global_steps}')

        self.actor_rollout_wg.load_checkpoint(
            os.path.join(global_step_folder, 'actor'),
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )

        abstraction_ckpt = os.path.join(global_step_folder, 'abstraction_actor')
        if os.path.exists(abstraction_ckpt):
            self.abstraction_actor_rollout_wg.load_checkpoint(
                abstraction_ckpt,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )
        else:
            print(f'Warning: no abstraction checkpoint found at {abstraction_ckpt}, abstraction branch starts fresh')

        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f'Warning: No dataloader state found at {dataloader_local_path}, will start from scratch')

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f'global_step_{self.global_steps}'
        )
        actor_local_path = os.path.join(local_global_step_folder, 'actor')
        abstraction_local_path = os.path.join(local_global_step_folder, 'abstraction_actor')

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        )
        abstraction_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f'global_step_{self.global_steps}',
                'abstraction_actor',
            )
        )
        remove_previous_ckpt_in_save = self.config.trainer.get('remove_previous_ckpt_in_save', False)
        if remove_previous_ckpt_in_save:
            print(
                'Warning: remove_previous_ckpt_in_save is deprecated,'
                + ' set max_actor_ckpt_to_keep=1 instead'
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get('max_actor_ckpt_to_keep', None) if not remove_previous_ckpt_in_save else 1
        )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )
        self.abstraction_actor_rollout_wg.save_checkpoint(
            abstraction_local_path,
            abstraction_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )

        local_mkdir_safe(local_global_step_folder)
        torch.save(self.train_dataloader.state_dict(), os.path.join(local_global_step_folder, 'data.pt'))

        if self._branch_uses_async_save(self.config.actor_rollout_ref) or self._branch_uses_async_save(
            self.config.abstraction_actor_rollout_ref
        ):
            print('skip write latest_checkpointed_iteration.txt when async_save is True')
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, 'latest_checkpointed_iteration.txt'
        )
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _start_profiling(self, do_profile: bool) -> None:
        super()._start_profiling(do_profile)
        if do_profile:
            self.abstraction_actor_rollout_wg.start_profile(role='e2e', profile_step=self.global_steps)
            if (
                self.use_abstraction_reference_policy
                and self.abstraction_ref_policy_wg is not self.abstraction_actor_rollout_wg
            ):
                self.abstraction_ref_policy_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        super()._stop_profiling(do_profile)
        if do_profile:
            self.abstraction_actor_rollout_wg.stop_profile()
            if (
                self.use_abstraction_reference_policy
                and self.abstraction_ref_policy_wg is not self.abstraction_actor_rollout_wg
            ):
                self.abstraction_ref_policy_wg.stop_profile()

    def _decode_responses(self, batch: DataProto, tokenizer):
        responses = batch.batch['responses']
        response_length = responses.size(1)
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        valid_lengths = response_mask.sum(dim=1).tolist()
        texts = []
        for row, valid_length in zip(responses, valid_lengths, strict=True):
            texts.append(tokenizer.decode(row[: int(valid_length)], skip_special_tokens=True))
        return texts

    def _build_problem_batch(self, batch: DataProto) -> DataProto:
        if 'problem' not in batch.non_tensor_batch:
            raise ValueError('Two-policy training data must include a problem column')
        if 'extra_info' not in batch.non_tensor_batch:
            batch.non_tensor_batch['extra_info'] = as_object_array({} for _ in range(len(batch)))
        batch.non_tensor_batch['problem_uid'] = as_object_array(str(uuid.uuid4()) for _ in range(len(batch)))
        # Rebuild the abstraction prompt from the config template so rollout behavior
        # is controlled by two_policy.abstraction_prompt_template_path rather than any
        # pre-rendered prompt stored in the parquet.
        batch.non_tensor_batch['raw_prompt'] = as_object_array(
            build_single_turn_chat(render_template(self.abstraction_prompt_template, problem))
            for problem in batch.non_tensor_batch['problem']
        )
        return batch

    @staticmethod
    def _get_manager_batch_divisor(manager) -> int:
        if hasattr(manager, 'agent_loop_workers'):
            return max(1, len(manager.agent_loop_workers))
        return 1

    def _generate_sequences_padded(self, manager, batch: DataProto, timing_raw: dict, do_profile: bool):
        size_divisor = self._get_manager_batch_divisor(manager)
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
        if do_profile:
            manager.start_profile()
        outputs = manager.generate_sequences(batch_padded)
        if do_profile:
            manager.stop_profile()
        timing_raw.update(outputs.meta_info.get('timing', {}))
        outputs.meta_info.pop('timing', None)
        return unpad_dataproto(outputs, pad_size)

    @staticmethod
    def _drop_overlapping_non_tensor_keys(base: DataProto, update: DataProto) -> DataProto:
        overlap_keys = [key for key in update.non_tensor_batch if key in base.non_tensor_batch]
        if not overlap_keys:
            return update
        # Keep the source batch prompt metadata authoritative and only union new rollout fields.
        update.pop(non_tensor_batch_keys=overlap_keys)
        return update

    @staticmethod
    def _build_rollout_input(batch: DataProto) -> DataProto:
        # The agent loop rebuilds prompt tensors from raw_prompt, so stale dataset prompt tensors must be dropped.
        return batch.select(batch_keys=[])

    def _compute_reward_scores_padded(self, batch: DataProto):
        size_divisor = max(1, len(self.reward_loop_manager.reward_loop_workers))
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, size_divisor)
        reward_padded = self._compute_reward_colocate(batch_padded)
        return unpad_dataproto(reward_padded, pad_size)

    @staticmethod
    def _validity_bonus(validity_score: float, cfg) -> float:
        if validity_score >= 1.0:
            return cfg.valid
        if validity_score > 0:
            return cfg.malformed
        return cfg.invalid

    def _run_two_policy_rollouts(
        self,
        batch: DataProto,
        timing_raw: dict,
        do_profile: bool,
        num_abstractions: int,
        num_solver_rollouts: int,
        solver_response_length: int | None = None,
    ):
        problem_batch = self._build_problem_batch(batch)
        abstraction_batch = self._build_rollout_input(
            problem_batch.repeat(repeat_times=num_abstractions, interleave=True)
        )
        abstraction_batch.meta_info['global_steps'] = self.global_steps
        abstraction_batch.meta_info['temperature'] = self.config.abstraction_actor_rollout_ref.rollout.temperature

        # Solver replicas are kept asleep between rollouts and updates.
        # Calling sleep() again here can issue a second sleep RPC to an already
        # sleeping vLLM engine and kill the server during validation startup.
        self.abstraction_checkpoint_manager.update_weights()
        abstraction_outputs = self._generate_sequences_padded(
            self.abstraction_async_rollout_manager, abstraction_batch, timing_raw, do_profile
        )
        self.abstraction_checkpoint_manager.sleep_replicas()
        abstraction_outputs = self._drop_overlapping_non_tensor_keys(abstraction_batch, abstraction_outputs)
        abstraction_batch = abstraction_batch.union(abstraction_outputs)

        abstraction_texts = self._decode_responses(abstraction_batch, self.abstraction_tokenizer)
        parse_results = [
            parse_abstraction_output(
                text=text,
                open_tag=self.config.two_policy.abstraction_open_tag,
                close_tag=self.config.two_policy.abstraction_close_tag,
                min_chars=self.config.two_policy.min_abstraction_chars,
                invalid_placeholders=tuple(self.config.two_policy.invalid_placeholders),
                require_principle_headers=self.config.two_policy.require_principle_headers,
                max_num_principles=self.config.two_policy.max_num_principles,
            )
            for text in abstraction_texts
        ]
        abstraction_batch.non_tensor_batch['abstraction_uid'] = as_object_array(
            str(uuid.uuid4()) for _ in range(len(abstraction_batch))
        )
        abstraction_batch.non_tensor_batch['abstraction_text'] = as_object_array(text for text in abstraction_texts)
        abstraction_batch.non_tensor_batch['parsed_abstraction'] = as_object_array(
            result.abstraction for result in parse_results
        )
        abstraction_batch.non_tensor_batch['abstraction_valid'] = np.asarray(
            [result.is_valid for result in parse_results], dtype=bool
        )
        abstraction_batch.non_tensor_batch['abstraction_validity_score'] = np.asarray(
            [result.validity_score for result in parse_results], dtype=np.float32
        )
        abstraction_batch.non_tensor_batch['abstraction_failure_reason'] = as_object_array(
            result.failure_reason for result in parse_results
        )
        abstraction_batch.non_tensor_batch['abstraction_principle_count'] = np.asarray(
            [result.principle_count for result in parse_results], dtype=np.int32
        )
        abstraction_batch.non_tensor_batch['uid'] = abstraction_batch.non_tensor_batch['problem_uid'].copy()

        valid_idxs = np.where(abstraction_batch.non_tensor_batch['abstraction_valid'])[0].tolist()
        solver_batch = None
        reward_extra_infos = {}
        solver_scalar_rewards = {}
        aggregated_extra_metrics = defaultdict(dict)

        if valid_idxs:
            solver_batch = self._build_rollout_input(
                abstraction_batch.select_idxs(valid_idxs).repeat(repeat_times=num_solver_rollouts, interleave=True)
            )
            solver_batch.non_tensor_batch['raw_prompt'] = as_object_array(
                build_single_turn_chat(
                    render_template(
                        self.solver_prompt_template,
                        problem=problem,
                        abstraction=abstraction,
                    )
                )
                for problem, abstraction in zip(
                    solver_batch.non_tensor_batch['problem'],
                    solver_batch.non_tensor_batch['parsed_abstraction'],
                    strict=True,
                )
            )
            solver_batch.non_tensor_batch['uid'] = solver_batch.non_tensor_batch['abstraction_uid'].copy()
            solver_batch.meta_info['global_steps'] = self.global_steps
            solver_batch.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature
            if solver_response_length is not None:
                solver_batch.meta_info['response_length'] = solver_response_length

            self.checkpoint_manager.update_weights()
            solver_outputs = self._generate_sequences_padded(self.async_rollout_manager, solver_batch, timing_raw, do_profile)
            self.checkpoint_manager.sleep_replicas()
            solver_outputs = self._drop_overlapping_non_tensor_keys(solver_batch, solver_outputs)
            solver_batch = solver_batch.union(solver_outputs)

            reward_batch = self._compute_reward_scores_padded(solver_batch)
            solver_batch = solver_batch.union(reward_batch)
            reward_tensor, reward_extra_infos = extract_reward(solver_batch)
            solver_batch.batch['response_mask'] = compute_response_mask(solver_batch)
            solver_batch.batch['token_level_scores'] = reward_tensor
            solver_batch.non_tensor_batch['seq_reward'] = reward_tensor.sum(dim=-1).cpu().numpy().astype(np.float32)
            solver_batch.non_tensor_batch['seq_final_reward'] = solver_batch.non_tensor_batch['seq_reward'].copy()

            for abstraction_uid, score in zip(
                solver_batch.non_tensor_batch['abstraction_uid'],
                solver_batch.non_tensor_batch['seq_final_reward'],
                strict=True,
            ):
                solver_scalar_rewards.setdefault(abstraction_uid, []).append(float(score))

            for key, values in reward_extra_infos.items():
                grouped_values = defaultdict(list)
                for abstraction_uid, value in zip(solver_batch.non_tensor_batch['abstraction_uid'], values, strict=True):
                    try:
                        grouped_values[abstraction_uid].append(float(value))
                    except (TypeError, ValueError):
                        grouped_values = None
                        break
                if grouped_values is not None:
                    for abstraction_uid, metric_values in grouped_values.items():
                        aggregated_extra_metrics[key][abstraction_uid] = float(np.mean(metric_values))
        else:
            print(f'Warning: all {len(abstraction_batch)} abstractions were invalid, solver update skipped')

        abstraction_rewards = []
        downstream_rewards = []
        validity_bonuses = []
        for abstraction_uid, validity_score in zip(
            abstraction_batch.non_tensor_batch['abstraction_uid'],
            abstraction_batch.non_tensor_batch['abstraction_validity_score'],
            strict=True,
        ):
            child_rewards = solver_scalar_rewards.get(abstraction_uid, [])
            downstream_reward = aggregate_child_rewards(
                child_rewards,
                self.config.two_policy.aggregation.mode,
                self.config.two_policy.aggregation.mean_weight,
                self.config.two_policy.aggregation.max_weight,
            )
            validity_bonus = self._validity_bonus(validity_score, self.config.two_policy.validity_reward)
            downstream_rewards.append(downstream_reward)
            validity_bonuses.append(validity_bonus)
            abstraction_rewards.append(downstream_reward + validity_bonus)

        abstraction_batch.batch['response_mask'] = compute_response_mask(abstraction_batch)
        abstraction_batch.batch['token_level_scores'] = build_scalar_reward_tensor(
            abstraction_batch.batch['response_mask'],
            torch.tensor(abstraction_rewards, dtype=torch.float32),
        )
        abstraction_batch.non_tensor_batch['seq_reward'] = np.asarray(abstraction_rewards, dtype=np.float32)
        abstraction_batch.non_tensor_batch['seq_final_reward'] = abstraction_batch.non_tensor_batch['seq_reward'].copy()
        abstraction_batch.non_tensor_batch['abstraction_downstream_reward'] = np.asarray(
            downstream_rewards, dtype=np.float32
        )
        abstraction_batch.non_tensor_batch['abstraction_validity_bonus'] = np.asarray(
            validity_bonuses, dtype=np.float32
        )

        for key, abstraction_uid_to_value in aggregated_extra_metrics.items():
            abstraction_batch.non_tensor_batch[key] = np.asarray(
                [abstraction_uid_to_value.get(uid, 0.0) for uid in abstraction_batch.non_tensor_batch['abstraction_uid']],
                dtype=np.float32,
            )

        return abstraction_batch, solver_batch

    @staticmethod
    def _get_metric_array(batch: DataProto, metric_name: str | None):
        if batch is None or len(batch) == 0:
            return np.asarray([], dtype=np.float32)
        metric_name = metric_name or 'seq_final_reward'
        if metric_name in batch.non_tensor_batch:
            return np.asarray(batch.non_tensor_batch[metric_name], dtype=np.float32)
        if metric_name in {'seq_reward', 'score'}:
            return batch.batch['token_level_scores'].sum(dim=-1).cpu().numpy().astype(np.float32)
        return (batch.batch['token_level_rewards'] if 'token_level_rewards' in batch.batch else batch.batch['token_level_scores']).sum(dim=-1).cpu().numpy().astype(np.float32)

    def _filter_group_uids(self, batch: DataProto, group_key: str, metric_name: str | None):
        if batch is None or len(batch) == 0:
            return []
        metrics = self._get_metric_array(batch, metric_name)
        group_to_values = defaultdict(list)
        ordered_groups = []
        for group_uid, metric_value in zip(batch.non_tensor_batch[group_key], metrics, strict=True):
            if group_uid not in group_to_values:
                ordered_groups.append(group_uid)
            group_to_values[group_uid].append(float(metric_value))
        return [uid for uid in ordered_groups if np.std(group_to_values[uid]) > 0 or len(group_to_values[uid]) == 1]

    @staticmethod
    def _select_by_group_uids(batch: DataProto, group_key: str, kept_group_uids):
        if batch is None or len(batch) == 0:
            return batch
        kept_group_uids = set(kept_group_uids)
        mask = np.asarray([uid in kept_group_uids for uid in batch.non_tensor_batch[group_key]], dtype=bool)
        return batch.select_idxs(mask)

    def _cap_problem_groups(self, abstraction_batch: DataProto, solver_batch: DataProto | None, max_problems: int):
        ordered_problem_uids = []
        seen = set()
        for uid in abstraction_batch.non_tensor_batch['problem_uid']:
            if uid not in seen:
                ordered_problem_uids.append(uid)
                seen.add(uid)
            if len(ordered_problem_uids) == max_problems:
                break
        abstraction_batch = self._select_by_group_uids(abstraction_batch, 'problem_uid', ordered_problem_uids)
        if solver_batch is not None:
            solver_batch = self._select_by_group_uids(solver_batch, 'problem_uid', ordered_problem_uids)
        return abstraction_batch, solver_batch, len(ordered_problem_uids)

    @staticmethod
    def _rename_actor_metrics(metrics: dict, branch_prefix: str):
        if branch_prefix == 'actor':
            return metrics
        renamed = {}
        for key, value in metrics.items():
            if key.startswith('actor/'):
                renamed[f'{branch_prefix}/{key[len("actor/"):]}'] = value
            elif key == 'perf/mfu/actor':
                renamed[f'perf/mfu/{branch_prefix}'] = value
            else:
                renamed[key] = value
        return renamed

    @staticmethod
    def _rename_kl_metrics(metrics: dict, branch_prefix: str):
        if branch_prefix == 'actor':
            return metrics
        renamed = {}
        for key, value in metrics.items():
            if key.startswith('actor/'):
                renamed[f'{branch_prefix}/{key[len("actor/"):]}'] = value
            else:
                renamed[key] = value
        return renamed

    def _prepare_branch_batch(
        self,
        batch: DataProto | None,
        branch_cfg,
        actor_rollout_wg,
        ref_policy_wg,
        ref_in_actor: bool,
        use_reference_policy: bool,
        use_prefix_grouper: bool,
        num_repeat: int,
        branch_prefix: str,
        kl_ctrl,
        metrics: dict,
        timing_raw: dict,
        timing_prefix: str,
    ):
        if batch is None or len(batch) == 0:
            return batch

        if 'response_mask' not in batch.batch:
            batch.batch['response_mask'] = compute_response_mask(batch)

        with self._use_branch(
            branch_cfg,
            actor_rollout_wg,
            ref_policy_wg,
            ref_in_actor,
            use_reference_policy,
            use_prefix_grouper,
        ):
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics, logging_prefix=f'{branch_prefix}_seqlen')

            batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

            with marked_timer(f'{timing_prefix}_old_log_prob', timing_raw, color='blue'):
                old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                entropys = old_log_prob.batch['entropys']
                entropy_agg = agg_loss(
                    loss_mat=entropys,
                    loss_mask=batch.batch['response_mask'],
                    loss_agg_mode=branch_cfg.actor.loss_agg_mode,
                    loss_scale_factor=branch_cfg.actor.loss_scale_factor,
                )
                metrics[f'{branch_prefix}/entropy'] = entropy_agg.detach().item()
                metrics[f'perf/mfu/{branch_prefix}_infer'] = old_log_prob_mfu
                old_log_prob.batch.pop('entropys')
                batch = batch.union(old_log_prob)

            if use_reference_policy:
                with marked_timer(f'{timing_prefix}_ref', timing_raw, color='olive'):
                    ref_log_prob = self._compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

        if self.config.algorithm.use_kl_in_reward:
            batch, kl_metrics = apply_kl_penalty(
                batch,
                kl_ctrl=kl_ctrl,
                kl_penalty=self.config.algorithm.kl_penalty,
            )
            metrics.update(self._rename_kl_metrics(kl_metrics, branch_prefix))
        else:
            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

        with marked_timer(f'{timing_prefix}_adv', timing_raw, color='brown'):
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=num_repeat,
                norm_adv_by_std_in_grpo=self.config.algorithm.get('norm_adv_by_std_in_grpo', True),
                config=self.config.algorithm,
            )
        return batch

    def _update_actor_for_branch(
        self,
        batch: DataProto | None,
        branch_cfg,
        actor_rollout_wg,
        ref_policy_wg,
        ref_in_actor: bool,
        use_reference_policy: bool,
        use_prefix_grouper: bool,
        branch_prefix: str,
    ):
        if batch is None or len(batch) == 0:
            return {}
        with self._use_branch(
            branch_cfg,
            actor_rollout_wg,
            ref_policy_wg,
            ref_in_actor,
            use_reference_policy,
            use_prefix_grouper,
        ):
            actor_output = self._update_actor(batch)
        return self._rename_actor_metrics(reduce_metrics(actor_output.meta_info['metrics']), branch_prefix)

    @staticmethod
    def _reward_extra_infos_dict(batch: DataProto):
        keys = batch.meta_info.get('reward_extra_keys', [])
        return {key: batch.non_tensor_batch[key].tolist() for key in keys if key in batch.non_tensor_batch}

    @staticmethod
    def _summarize_batches(abstraction_batch: DataProto, solver_batch: DataProto | None):
        metrics = {
            'two_policy/abstraction_valid_rate': float(np.mean(abstraction_batch.non_tensor_batch['abstraction_valid'])),
            'two_policy/abstraction_reward_mean': float(np.mean(abstraction_batch.non_tensor_batch['seq_final_reward'])),
            'two_policy/abstraction_downstream_reward_mean': float(
                np.mean(abstraction_batch.non_tensor_batch['abstraction_downstream_reward'])
            ),
        }
        if solver_batch is not None and len(solver_batch) > 0:
            metrics['two_policy/solver_reward_mean'] = float(np.mean(solver_batch.non_tensor_batch['seq_final_reward']))
            if 'acc' in solver_batch.non_tensor_batch:
                metrics['two_policy/solver_acc_mean'] = float(np.mean(solver_batch.non_tensor_batch['acc']))
        else:
            metrics['two_policy/solver_reward_mean'] = 0.0
        return metrics

    def _validate(self, merged: bool = False):
        del merged
        num_abstractions = self.config.two_policy.validation_num_abstractions
        num_solver_rollouts = self.config.two_policy.validation_num_solver_rollouts
        solver_response_length = self.config.two_policy.validation_solver_response_length
        val_data_dir = self.config.trainer.get('validation_data_dir', None)

        problem_best_rewards = []
        problem_mean_rewards = []
        problem_best_accs = []
        problem_mean_accs = []
        abstraction_validities = []
        validation_dump_entries = []

        source_problem_best_rewards = defaultdict(list)
        source_problem_mean_rewards = defaultdict(list)
        source_problem_best_accs = defaultdict(list)
        source_problem_mean_accs = defaultdict(list)
        source_abstraction_validities = defaultdict(list)
        source_names = set()

        for batch_dict in self.val_dataloader:
            batch = DataProto.from_single_dict(batch_dict)
            abstraction_batch, solver_batch = self._run_two_policy_rollouts(
                batch=batch,
                timing_raw=defaultdict(float),
                do_profile=False,
                num_abstractions=num_abstractions,
                num_solver_rollouts=num_solver_rollouts,
                solver_response_length=solver_response_length,
            )
            if val_data_dir:
                validation_dump_entries.extend(
                    self._collect_two_policy_validation_rollouts(
                        abstraction_batch=abstraction_batch,
                        solver_batch=solver_batch,
                    )
                )

            abstraction_valid_array = abstraction_batch.non_tensor_batch['abstraction_valid'].astype(np.float32)
            abstraction_validities.extend(abstraction_valid_array.tolist())

            batch_sources = abstraction_batch.non_tensor_batch.get('data_source')
            if batch_sources is None:
                batch_sources = np.asarray(['unknown'] * len(abstraction_batch), dtype=object)
            normalized_sources = [str(source).strip().lower().replace(' ', '_') for source in batch_sources]
            source_names.update(normalized_sources)
            for source, is_valid in zip(normalized_sources, abstraction_valid_array, strict=True):
                source_abstraction_validities[source].append(float(is_valid))

            problem_uids = abstraction_batch.non_tensor_batch['problem_uid']
            unique_problem_uids = []
            problem_source = {}
            seen = set()
            for idx, uid in enumerate(problem_uids):
                if uid not in seen:
                    unique_problem_uids.append(uid)
                    problem_source[uid] = normalized_sources[idx]
                    seen.add(uid)

            if solver_batch is None or len(solver_batch) == 0:
                for problem_uid in unique_problem_uids:
                    source = problem_source[problem_uid]
                    problem_best_rewards.append(0.0)
                    problem_mean_rewards.append(0.0)
                    source_problem_best_rewards[source].append(0.0)
                    source_problem_mean_rewards[source].append(0.0)
                continue

            reward_by_problem = defaultdict(list)
            acc_by_problem = defaultdict(list)
            for idx, problem_uid in enumerate(solver_batch.non_tensor_batch['problem_uid']):
                reward_by_problem[problem_uid].append(float(solver_batch.non_tensor_batch['seq_final_reward'][idx]))
                if 'acc' in solver_batch.non_tensor_batch:
                    acc_by_problem[problem_uid].append(float(solver_batch.non_tensor_batch['acc'][idx]))

            for problem_uid in unique_problem_uids:
                source = problem_source[problem_uid]
                rewards = reward_by_problem.get(problem_uid, [0.0])
                best_reward = float(np.max(rewards))
                mean_reward = float(np.mean(rewards))
                problem_best_rewards.append(best_reward)
                problem_mean_rewards.append(mean_reward)
                source_problem_best_rewards[source].append(best_reward)
                source_problem_mean_rewards[source].append(mean_reward)
                if acc_by_problem:
                    accs = acc_by_problem.get(problem_uid, [0.0])
                    best_acc = float(np.max(accs))
                    mean_acc = float(np.mean(accs))
                    problem_best_accs.append(best_acc)
                    problem_mean_accs.append(mean_acc)
                    source_problem_best_accs[source].append(best_acc)
                    source_problem_mean_accs[source].append(mean_acc)

        metrics = {
            'val/abstraction_valid_rate': float(np.mean(abstraction_validities)) if abstraction_validities else 0.0,
            'val/problem_best_reward_mean': float(np.mean(problem_best_rewards)) if problem_best_rewards else 0.0,
            'val/problem_mean_reward_mean': float(np.mean(problem_mean_rewards)) if problem_mean_rewards else 0.0,
        }
        if problem_best_accs:
            metrics['val/problem_best_acc_mean'] = float(np.mean(problem_best_accs))
            metrics['val/problem_mean_acc_mean'] = float(np.mean(problem_mean_accs))

        for source in sorted(source_names):
            metrics[f'val/{source}/abstraction_valid_rate'] = (
                float(np.mean(source_abstraction_validities[source])) if source_abstraction_validities[source] else 0.0
            )
            metrics[f'val/{source}/problem_best_reward_mean'] = (
                float(np.mean(source_problem_best_rewards[source])) if source_problem_best_rewards[source] else 0.0
            )
            metrics[f'val/{source}/problem_mean_reward_mean'] = (
                float(np.mean(source_problem_mean_rewards[source])) if source_problem_mean_rewards[source] else 0.0
            )
            if problem_best_accs:
                metrics[f'val/{source}/problem_best_acc_mean'] = (
                    float(np.mean(source_problem_best_accs[source])) if source_problem_best_accs[source] else 0.0
                )
                metrics[f'val/{source}/problem_mean_acc_mean'] = (
                    float(np.mean(source_problem_mean_accs[source])) if source_problem_mean_accs[source] else 0.0
                )

        if val_data_dir:
            self._dump_two_policy_validation_rollouts(entries=validation_dump_entries, dump_path=val_data_dir)
        return metrics


    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        self.max_steps_duration = 0

        self._load_checkpoint()
        self.checkpoint_manager.update_weights()
        self.checkpoint_manager.sleep_replicas()
        self.abstraction_checkpoint_manager.update_weights()
        self.abstraction_checkpoint_manager.sleep_replicas()

        if self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            assert val_metrics, f'{val_metrics=}'
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc='Training Progress')

        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        abstraction_accum = []
        solver_accum = []
        timing_raw = defaultdict(float)
        num_prompt_in_batch = 0
        num_gen_batches = 0
        current_epoch = self.global_steps // len(self.train_dataloader)

        filter_cfg = self.config.algorithm.get('filter_groups', None)
        filter_enabled = filter_cfg is not None and filter_cfg.enable
        filter_metric = filter_cfg.metric if filter_enabled else None
        max_num_gen_batches = filter_cfg.max_num_gen_batches if filter_enabled else 0

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, 'async_calls_finalize_fn_exec'):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                if hasattr(self.abstraction_actor_rollout_wg, 'async_calls_finalize_fn_exec'):
                    self.abstraction_actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

                metrics = {}
                with marked_timer('start_profile', timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch = DataProto.from_single_dict(batch_dict)
                is_last_step = self.global_steps >= self.total_training_steps
                num_gen_batches += 1

                with marked_timer('step', timing_raw):
                    abstraction_batch, solver_batch = self._run_two_policy_rollouts(
                        batch=batch,
                        timing_raw=timing_raw,
                        do_profile=curr_step_profile,
                        num_abstractions=self.config.two_policy.num_abstractions,
                        num_solver_rollouts=self.config.two_policy.num_solver_rollouts,
                    )

                    if filter_enabled:
                        kept_problem_uids = self._filter_group_uids(abstraction_batch, 'problem_uid', filter_metric)
                        abstraction_batch = self._select_by_group_uids(abstraction_batch, 'problem_uid', kept_problem_uids)
                        if solver_batch is not None:
                            solver_batch = self._select_by_group_uids(solver_batch, 'problem_uid', kept_problem_uids)
                        kept_problem_count = len(kept_problem_uids)
                        metrics['two_policy/filter_problem_groups_kept'] = kept_problem_count
                        metrics['two_policy/filter_problem_groups_total'] = len(batch)
                        if kept_problem_count > 0:
                            abstraction_accum.append(abstraction_batch)
                            if solver_batch is not None and len(solver_batch) > 0:
                                solver_accum.append(solver_batch)
                            num_prompt_in_batch += kept_problem_count
                        if num_prompt_in_batch < self.config.data.train_batch_size:
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                continue
                            raise ValueError(
                                f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many batches while filtering.'
                            )
                        abstraction_batch = DataProto.concat(abstraction_accum)
                        solver_batch = DataProto.concat(solver_accum) if solver_accum else None
                        abstraction_batch, solver_batch, _ = self._cap_problem_groups(
                            abstraction_batch, solver_batch, self.config.data.train_batch_size
                        )
                    else:
                        abstraction_accum = []
                        solver_accum = []
                        num_prompt_in_batch = 0

                    if self.config.two_policy.filter_solver_groups and solver_batch is not None and len(solver_batch) > 0:
                        kept_abstraction_uids = self._filter_group_uids(solver_batch, 'abstraction_uid', filter_metric)
                        metrics['two_policy/filter_solver_groups_kept'] = len(kept_abstraction_uids)
                        solver_batch = self._select_by_group_uids(solver_batch, 'abstraction_uid', kept_abstraction_uids)

                    abstraction_batch = self._prepare_branch_batch(
                        batch=abstraction_batch,
                        branch_cfg=self.config.abstraction_actor_rollout_ref,
                        actor_rollout_wg=self.abstraction_actor_rollout_wg,
                        ref_policy_wg=self.abstraction_ref_policy_wg,
                        ref_in_actor=self.abstraction_ref_in_actor,
                        use_reference_policy=self.use_abstraction_reference_policy,
                        use_prefix_grouper=self.abstraction_use_prefix_grouper,
                        num_repeat=self.config.two_policy.num_abstractions,
                        branch_prefix='abstraction_actor',
                        kl_ctrl=self.abstraction_kl_ctrl_in_reward,
                        metrics=metrics,
                        timing_raw=timing_raw,
                        timing_prefix='abstraction',
                    )
                    solver_batch = self._prepare_branch_batch(
                        batch=solver_batch,
                        branch_cfg=self.config.actor_rollout_ref,
                        actor_rollout_wg=self.actor_rollout_wg,
                        ref_policy_wg=self.ref_policy_wg,
                        ref_in_actor=self.ref_in_actor,
                        use_reference_policy=self.use_reference_policy,
                        use_prefix_grouper=self.use_prefix_grouper,
                        num_repeat=self.config.two_policy.num_solver_rollouts,
                        branch_prefix='actor',
                        kl_ctrl=self.kl_ctrl_in_reward if self.config.algorithm.use_kl_in_reward else None,
                        metrics=metrics,
                        timing_raw=timing_raw,
                        timing_prefix='solver',
                    )

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer('update_solver_actor', timing_raw, color='red'):
                            metrics.update(
                                self._update_actor_for_branch(
                                    batch=solver_batch,
                                    branch_cfg=self.config.actor_rollout_ref,
                                    actor_rollout_wg=self.actor_rollout_wg,
                                    ref_policy_wg=self.ref_policy_wg,
                                    ref_in_actor=self.ref_in_actor,
                                    use_reference_policy=self.use_reference_policy,
                                    use_prefix_grouper=self.use_prefix_grouper,
                                    branch_prefix='actor',
                                )
                            )
                        with marked_timer('update_abstraction_actor', timing_raw, color='red'):
                            metrics.update(
                                self._update_actor_for_branch(
                                    batch=abstraction_batch,
                                    branch_cfg=self.config.abstraction_actor_rollout_ref,
                                    actor_rollout_wg=self.abstraction_actor_rollout_wg,
                                    ref_policy_wg=self.abstraction_ref_policy_wg,
                                    ref_in_actor=self.abstraction_ref_in_actor,
                                    use_reference_policy=self.use_abstraction_reference_policy,
                                    use_prefix_grouper=self.abstraction_use_prefix_grouper,
                                    branch_prefix='abstraction_actor',
                                )
                            )

                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            with marked_timer('save_checkpoint', timing_raw, color='green'):
                                self._save_checkpoint()

                        with marked_timer('update_weights', timing_raw, color='red'):
                            self.checkpoint_manager.update_weights()
                            self.checkpoint_manager.sleep_replicas()
                            self.abstraction_checkpoint_manager.update_weights()
                            self.abstraction_checkpoint_manager.sleep_replicas()

                    rollout_data_dir = self.config.trainer.get('rollout_data_dir', None)
                    if rollout_data_dir and solver_batch is not None and len(solver_batch) > 0:
                        self._log_rollout_data(
                            solver_batch,
                            self._reward_extra_infos_dict(solver_batch),
                            timing_raw,
                            rollout_data_dir,
                        )

                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer('testing', timing_raw, color='green'):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer('stop_profile', timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw['step']
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update({'training/global_step': self.global_steps, 'training/epoch': epoch})
                metrics.update(self._summarize_batches(abstraction_batch, solver_batch))
                metric_batch = solver_batch if solver_batch is not None and len(solver_batch) > 0 else abstraction_batch
                metrics.update(compute_data_metrics(batch=metric_batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=metric_batch, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=metric_batch,
                        timing_raw=timing_raw,
                        n_gpus=self.resource_pool_manager.get_n_gpus(),
                    )
                )
                metrics.update(compute_variance_proxy_metrics(batch=metric_batch, gradient_norm=None))
                metrics['train/num_gen_batches'] = num_gen_batches

                logger.log(data=metrics, step=self.global_steps)

                abstraction_accum = []
                solver_accum = []
                timing_raw = defaultdict(float)
                num_prompt_in_batch = 0
                num_gen_batches = 0

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, 'async_calls_finalize_fn_exec'):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    if hasattr(self.abstraction_actor_rollout_wg, 'async_calls_finalize_fn_exec'):
                        self.abstraction_actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return

        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f'global_step_{self.global_steps}')
        if not os.path.exists(checkpoint_dir):
            timing_raw = defaultdict(float)
            with marked_timer('save_checkpoint', timing_raw, color='green'):
                self._save_checkpoint()
            logger.log(data={f'timing/{k}': v for k, v in timing_raw.items()}, step=self.global_steps)
