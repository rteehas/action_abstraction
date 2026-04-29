from __future__ import annotations

import copy
import math
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.trainer.ppo.reward import extract_reward
from verl.utils.model import compute_position_id_with_mask

from .deepseek_vm import DeepSeekVMScorer
from .two_policy_trainer import TwoPolicyGRPOTrainer
from .two_policy_utils import (
    aggregate_child_rewards,
    as_object_array,
    build_scalar_reward_tensor,
    build_single_turn_chat,
    parse_abstraction_output,
    render_template,
)


@dataclass
class TraceState:
    prompt_token_ids: list[int]
    response_token_ids: list[int] = field(default_factory=list)
    finished_exact: bool = False
    pruned: bool = False
    proxy_reward: float = 0.0
    vm_score: float = 0.0
    observed_tokens: int = 0
    prune_stage_idx: int = -1
    stage_count: int = 0
    terminal_reason: str = "active"


class TwoPolicyVMPruneGRPOTrainer(TwoPolicyGRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vm_cfg = self.config.two_policy.get("vm_pruning", None)
        self._vm_pruning_cfg = vm_cfg
        self._vm_pruning_active = False
        self._vm_scorer = None
        if vm_cfg is not None and vm_cfg.get("enable", False):
            self._vm_scorer = DeepSeekVMScorer(
                model_path=str(vm_cfg.model_path),
                tokenizer_path=str(vm_cfg.tokenizer_path),
                torch_dtype=str(vm_cfg.get("torch_dtype", "bfloat16")),
                attn_implementation=str(vm_cfg.get("attn_implementation", "flash_attention_2")),
                device=vm_cfg.get("device", None),
                device_map=vm_cfg.get("device_map", None),
                batch_size=int(vm_cfg.get("scoring_batch_size", 8)),
                offload_enable=bool(vm_cfg.offload.get("enable", False)),
                offload_folder=vm_cfg.offload.get("folder", None),
                offload_state_dict=bool(vm_cfg.offload.get("state_dict", True)),
                low_cpu_mem_usage=bool(vm_cfg.offload.get("low_cpu_mem_usage", True)),
            )

    def fit(self):
        prev = self._vm_pruning_active
        self._vm_pruning_active = bool(self._vm_pruning_cfg is not None and self._vm_pruning_cfg.get("enable", False))
        try:
            return super().fit()
        finally:
            self._vm_pruning_active = prev

    def _validate(self, merged: bool = False):
        prev = self._vm_pruning_active
        apply_in_validation = bool(
            self._vm_pruning_cfg is not None
            and self._vm_pruning_cfg.get("enable", False)
            and self._vm_pruning_cfg.get("apply_in_validation", False)
        )
        self._vm_pruning_active = apply_in_validation
        try:
            return super()._validate(merged=merged)
        finally:
            self._vm_pruning_active = prev

    @staticmethod
    def _chat_prompt_token_ids(tokenizer, raw_prompt) -> list[int]:
        prompt_text = tokenizer.apply_chat_template(raw_prompt, add_generation_prompt=True, tokenize=False)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
        return prompt_ids.tolist()

    @contextmanager
    def _temporary_manager_prompt_length(self, manager, prompt_length: int):
        old_prompt_length = manager.rollout_config.prompt_length
        manager.rollout_config.prompt_length = max(int(prompt_length), int(old_prompt_length))
        try:
            yield
        finally:
            manager.rollout_config.prompt_length = old_prompt_length

    @staticmethod
    def _stage_valid_response_lengths(batch: DataProto) -> list[int]:
        prompt_length = batch.batch["prompts"].size(1)
        return batch.batch["attention_mask"][:, prompt_length:].sum(dim=1).cpu().tolist()

    @staticmethod
    def _fill_reward_extra_key(length: int, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if values.dtype == object:
            filled = np.empty(length, dtype=object)
            filled[:] = None
        else:
            filled = np.zeros(length, dtype=values.dtype)
        return filled

    def _compute_keep_count(self, group_size: int, group_cfg) -> int:
        keep_count = group_cfg.get("keep_count", None)
        min_keep = max(1, int(group_cfg.get("min_keep", 1)))
        if keep_count is not None:
            keep = int(keep_count)
        else:
            keep_fraction = float(group_cfg.get("keep_fraction", 0.5))
            rounded = group_size * keep_fraction
            rounding = str(group_cfg.get("rounding", "ceil")).lower()
            if rounding == "floor":
                keep = math.floor(rounded)
            else:
                keep = math.ceil(rounded)
        keep = max(min_keep, keep)
        return min(group_size, keep)

    def _is_abstraction_update_pruning_mode(
        self,
        *,
        num_solver_rollouts: int,
        solver_temperature: float | None,
        solver_top_p: float | None,
        solver_top_k: int | None,
    ) -> bool:
        schedule_cfg = self.config.two_policy.get("decoupled_solver_schedule", None)
        if schedule_cfg is None or not schedule_cfg.get("enable", False):
            return False

        default_rollout_cfg = self.config.actor_rollout_ref.rollout
        actual_temperature = (
            float(default_rollout_cfg.temperature) if solver_temperature is None else float(solver_temperature)
        )
        actual_top_p = float(default_rollout_cfg.get("top_p", 1.0) if solver_top_p is None else solver_top_p)
        actual_top_k = int(default_rollout_cfg.get("top_k", 0) if solver_top_k is None else solver_top_k)
        return (
            int(num_solver_rollouts) == int(schedule_cfg.get("non_update_solver_rollouts", 1))
            and abs(actual_temperature - float(schedule_cfg.get("non_update_solver_temperature", 0.0))) < 1e-8
            and abs(actual_top_p - float(schedule_cfg.get("non_update_solver_top_p", 1.0))) < 1e-8
            and actual_top_k == int(schedule_cfg.get("non_update_solver_top_k", -1))
        )

    def _get_vm_group_cfg(
        self,
        *,
        num_solver_rollouts: int,
        solver_temperature: float | None,
        solver_top_p: float | None,
        solver_top_k: int | None,
    ):
        assert self._vm_pruning_cfg is not None
        if self._is_abstraction_update_pruning_mode(
            num_solver_rollouts=num_solver_rollouts,
            solver_temperature=solver_temperature,
            solver_top_p=solver_top_p,
            solver_top_k=solver_top_k,
        ):
            return self._vm_pruning_cfg.abstraction_update, "problem_uid"
        return self._vm_pruning_cfg.solver_update, "abstraction_uid"

    def _score_active_partial_traces(
        self,
        solver_source_batch: DataProto,
        trace_states: list[TraceState],
        active_indices: list[int],
    ) -> None:
        if not active_indices:
            return
        problems = [str(solver_source_batch.non_tensor_batch["problem"][idx]) for idx in active_indices]
        responses = [
            self.tokenizer.decode(trace_states[idx].response_token_ids, skip_special_tokens=True) for idx in active_indices
        ]
        scores, observed_tokens = self._vm_scorer.score_partial_texts(problems=problems, responses=responses)
        for idx, score, token_count in zip(active_indices, scores, observed_tokens, strict=True):
            trace_states[idx].vm_score = float(score)
            trace_states[idx].observed_tokens = int(token_count)

    def _prune_active_traces(
        self,
        *,
        solver_source_batch: DataProto,
        trace_states: list[TraceState],
        active_indices: list[int],
        group_key: str,
        group_cfg,
        stage_idx: int,
    ) -> list[int]:
        group_to_indices: dict[object, list[int]] = defaultdict(list)
        for idx in active_indices:
            group_to_indices[solver_source_batch.non_tensor_batch[group_key][idx]].append(idx)

        indices_to_score: list[int] = []
        for group_indices in group_to_indices.values():
            keep_count = self._compute_keep_count(len(group_indices), group_cfg)
            if keep_count < len(group_indices):
                indices_to_score.extend(group_indices)
        self._score_active_partial_traces(
            solver_source_batch=solver_source_batch,
            trace_states=trace_states,
            active_indices=indices_to_score,
        )

        next_active: list[int] = []
        for group_indices in group_to_indices.values():
            keep_count = self._compute_keep_count(len(group_indices), group_cfg)
            if keep_count >= len(group_indices):
                next_active.extend(group_indices)
                continue

            ranked = sorted(group_indices, key=lambda idx: (-trace_states[idx].vm_score, idx))
            kept = set(ranked[:keep_count])
            for idx in group_indices:
                if idx in kept:
                    next_active.append(idx)
                    continue
                trace_states[idx].pruned = True
                trace_states[idx].finished_exact = False
                trace_states[idx].proxy_reward = float(trace_states[idx].vm_score)
                trace_states[idx].prune_stage_idx = int(stage_idx)
                trace_states[idx].terminal_reason = "vm_pruned"
        return sorted(next_active)

    def _build_final_batch_from_traces(self, source_batch: DataProto, trace_states: list[TraceState]) -> DataProto:
        prompt_ids_list = [trace.prompt_token_ids for trace in trace_states]
        response_ids_list = [trace.response_token_ids for trace in trace_states]
        if not prompt_ids_list:
            return source_batch

        max_prompt_len = max(len(item) for item in prompt_ids_list)
        max_response_len = max(1, max(len(item) for item in response_ids_list))
        batch_size = len(trace_states)

        prompts = torch.zeros((batch_size, max_prompt_len), dtype=torch.long)
        prompt_attention_mask = torch.zeros((batch_size, max_prompt_len), dtype=torch.long)
        responses = torch.zeros((batch_size, max_response_len), dtype=torch.long)
        response_attention_mask = torch.zeros((batch_size, max_response_len), dtype=torch.long)

        for row_idx, (prompt_ids, response_ids) in enumerate(zip(prompt_ids_list, response_ids_list, strict=True)):
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)
            response_tensor = torch.tensor(response_ids, dtype=torch.long)
            prompts[row_idx, -len(prompt_ids) :] = prompt_tensor
            prompt_attention_mask[row_idx, -len(prompt_ids) :] = 1
            responses[row_idx, : len(response_ids)] = response_tensor
            response_attention_mask[row_idx, : len(response_ids)] = 1

        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        input_ids = torch.cat([prompts, responses], dim=1)
        response_mask = response_attention_mask.clone()
        position_ids = compute_position_id_with_mask(attention_mask)

        non_tensor_batch = {
            key: value.copy() if isinstance(value, np.ndarray) else np.asarray(value) for key, value in source_batch.non_tensor_batch.items()
        }
        non_tensor_batch["vm_pruned"] = np.asarray([trace.pruned for trace in trace_states], dtype=bool)
        non_tensor_batch["vm_proxy_reward"] = np.asarray([trace.proxy_reward for trace in trace_states], dtype=np.float32)
        non_tensor_batch["vm_score"] = np.asarray([trace.vm_score for trace in trace_states], dtype=np.float32)
        non_tensor_batch["vm_observed_tokens"] = np.asarray(
            [trace.observed_tokens for trace in trace_states], dtype=np.int32
        )
        non_tensor_batch["vm_prune_stage_idx"] = np.asarray(
            [trace.prune_stage_idx for trace in trace_states], dtype=np.int32
        )
        non_tensor_batch["vm_stage_count"] = np.asarray([trace.stage_count for trace in trace_states], dtype=np.int32)
        non_tensor_batch["vm_terminal_reason"] = as_object_array(trace.terminal_reason for trace in trace_states)
        non_tensor_batch["vm_finished_exact"] = np.asarray(
            [trace.finished_exact and not trace.pruned for trace in trace_states], dtype=bool
        )

        return DataProto.from_dict(
            tensors={
                "prompts": prompts,
                "responses": responses,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            non_tensors=non_tensor_batch,
            meta_info=copy.deepcopy(source_batch.meta_info),
        )

    def _assign_terminal_rewards(self, batch: DataProto, trace_states: list[TraceState]) -> tuple[torch.Tensor, dict]:
        reward_extra_keys: list[str] = []
        exact_indices = [
            idx
            for idx, trace in enumerate(trace_states)
            if trace.finished_exact and not trace.pruned and len(trace.response_token_ids) > 0
        ]
        rm_scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)

        if exact_indices:
            exact_batch = batch.select_idxs(exact_indices)
            reward_batch = self._compute_reward_scores_padded(exact_batch)
            exact_batch = exact_batch.union(reward_batch)
            exact_reward_tensor, reward_extra_infos = extract_reward(exact_batch)
            rm_scores[torch.tensor(exact_indices, dtype=torch.long)] = exact_reward_tensor
            reward_extra_keys = list(reward_extra_infos.keys())
            for key, values in reward_extra_infos.items():
                filled = self._fill_reward_extra_key(len(batch), values)
                filled[np.asarray(exact_indices, dtype=np.int32)] = values
                batch.non_tensor_batch[key] = filled

        response_lengths = batch.batch["response_mask"].sum(dim=1).cpu().tolist()
        for idx, (trace, response_length) in enumerate(zip(trace_states, response_lengths, strict=True)):
            if trace.pruned and response_length > 0:
                rm_scores[idx, int(response_length) - 1] = float(trace.proxy_reward)

        batch.batch["rm_scores"] = rm_scores
        batch.meta_info["reward_extra_keys"] = reward_extra_keys
        return extract_reward(batch)

    def _run_solver_rollouts_with_vm_pruning(
        self,
        *,
        solver_batch: DataProto,
        timing_raw: dict,
        do_profile: bool,
        total_response_length: int,
        num_solver_rollouts: int,
        solver_temperature: float | None,
        solver_top_p: float | None,
        solver_top_k: int | None,
    ) -> DataProto:
        assert self._vm_pruning_cfg is not None and self._vm_scorer is not None
        block_length = max(1, int(self._vm_pruning_cfg.block_length))
        group_cfg, group_key = self._get_vm_group_cfg(
            num_solver_rollouts=num_solver_rollouts,
            solver_temperature=solver_temperature,
            solver_top_p=solver_top_p,
            solver_top_k=solver_top_k,
        )

        trace_states = [
            TraceState(prompt_token_ids=self._chat_prompt_token_ids(self.tokenizer, raw_prompt))
            for raw_prompt in solver_batch.non_tensor_batch["raw_prompt"]
        ]
        active_indices = list(range(len(trace_states)))
        stage_idx = 0

        while active_indices:
            stage_idx += 1
            remaining = [
                total_response_length - len(trace_states[idx].response_token_ids)
                for idx in active_indices
                if not trace_states[idx].pruned and not trace_states[idx].finished_exact
            ]
            if not remaining:
                break

            stage_budget = max(1, min(block_length, min(remaining)))
            stage_batch = solver_batch.select_idxs(active_indices)
            stage_batch.meta_info = copy.deepcopy(stage_batch.meta_info)
            stage_batch.meta_info["response_length"] = stage_budget
            stage_batch.non_tensor_batch["prefill_response_ids"] = as_object_array(
                np.asarray(trace_states[idx].response_token_ids, dtype=np.int64) for idx in active_indices
            )

            max_prompt_len = max(
                len(trace_states[idx].prompt_token_ids) + len(trace_states[idx].response_token_ids) for idx in active_indices
            )
            stage_batch.meta_info["prompt_length"] = max_prompt_len
            stage_outputs = self._generate_sequences_padded(
                self.async_rollout_manager,
                stage_batch,
                timing_raw,
                do_profile,
            )

            valid_lengths = self._stage_valid_response_lengths(stage_outputs)
            stop_reasons = stage_outputs.non_tensor_batch.get("stop_reason")
            next_active: list[int] = []
            for local_idx, global_idx in enumerate(active_indices):
                new_len = int(valid_lengths[local_idx])
                if new_len > 0:
                    new_tokens = stage_outputs.batch["responses"][local_idx, :new_len].cpu().tolist()
                    trace_states[global_idx].response_token_ids.extend(new_tokens)
                trace_states[global_idx].stage_count += 1

                total_generated = len(trace_states[global_idx].response_token_ids)
                naturally_completed = new_len < stage_budget
                hit_total_limit = total_generated >= total_response_length
                if naturally_completed and total_generated == 0:
                    trace_states[global_idx].pruned = True
                    trace_states[global_idx].proxy_reward = 0.0
                    trace_states[global_idx].terminal_reason = "empty_response"
                    continue
                if naturally_completed or hit_total_limit:
                    trace_states[global_idx].finished_exact = True
                    trace_states[global_idx].terminal_reason = (
                        "exact_complete"
                        if naturally_completed
                        else "max_response_length"
                    )
                else:
                    if stop_reasons is not None and stop_reasons[local_idx] == "aborted":
                        trace_states[global_idx].finished_exact = True
                        trace_states[global_idx].terminal_reason = "aborted"
                    else:
                        next_active.append(global_idx)

            if not next_active:
                active_indices = []
                continue
            active_indices = self._prune_active_traces(
                solver_source_batch=solver_batch,
                trace_states=trace_states,
                active_indices=next_active,
                group_key=group_key,
                group_cfg=group_cfg,
                stage_idx=stage_idx,
            )

        final_solver_batch = self._build_final_batch_from_traces(solver_batch, trace_states)
        reward_tensor, _ = self._assign_terminal_rewards(final_solver_batch, trace_states)
        final_solver_batch.batch["response_mask"] = compute_response_mask(final_solver_batch)
        final_solver_batch.batch["token_level_scores"] = reward_tensor
        final_solver_batch.non_tensor_batch["seq_reward"] = reward_tensor.sum(dim=-1).cpu().numpy().astype(np.float32)
        final_solver_batch.non_tensor_batch["seq_final_reward"] = (
            final_solver_batch.non_tensor_batch["seq_reward"].copy()
        )
        return final_solver_batch

    def _should_use_vm_pruning(self) -> bool:
        return bool(
            self._vm_pruning_active
            and self._vm_pruning_cfg is not None
            and self._vm_pruning_cfg.get("enable", False)
            and self._vm_scorer is not None
        )

    def _run_two_policy_rollouts(
        self,
        batch: DataProto,
        timing_raw: dict,
        do_profile: bool,
        num_abstractions: int,
        num_solver_rollouts: int,
        solver_response_length: int | None = None,
        solver_temperature: float | None = None,
        solver_top_p: float | None = None,
        solver_top_k: int | None = None,
    ):
        if not self._should_use_vm_pruning():
            return super()._run_two_policy_rollouts(
                batch=batch,
                timing_raw=timing_raw,
                do_profile=do_profile,
                num_abstractions=num_abstractions,
                num_solver_rollouts=num_solver_rollouts,
                solver_response_length=solver_response_length,
                solver_temperature=solver_temperature,
                solver_top_p=solver_top_p,
                solver_top_k=solver_top_k,
            )

        problem_batch = self._build_problem_batch(batch)
        abstraction_batch = self._build_rollout_input(
            problem_batch.repeat(repeat_times=num_abstractions, interleave=True)
        )
        abstraction_batch.meta_info["global_steps"] = self.global_steps
        abstraction_batch.meta_info["temperature"] = self.config.abstraction_actor_rollout_ref.rollout.temperature

        self.abstraction_checkpoint_manager.update_weights()
        abstraction_outputs = self._generate_sequences_padded(
            self.abstraction_async_rollout_manager,
            abstraction_batch,
            timing_raw,
            do_profile,
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
        abstraction_batch.non_tensor_batch["abstraction_uid"] = as_object_array(
            str(uuid.uuid4()) for _ in range(len(abstraction_batch))
        )
        abstraction_batch.non_tensor_batch["abstraction_text"] = as_object_array(text for text in abstraction_texts)
        abstraction_batch.non_tensor_batch["parsed_abstraction"] = as_object_array(
            result.abstraction for result in parse_results
        )
        abstraction_batch.non_tensor_batch["solver_conditioning_abstraction"] = as_object_array(
            result.solver_conditioning_abstraction for result in parse_results
        )
        abstraction_batch.non_tensor_batch["abstraction_valid"] = np.asarray(
            [result.is_valid for result in parse_results], dtype=bool
        )
        abstraction_batch.non_tensor_batch["abstraction_validity_score"] = np.asarray(
            [result.validity_score for result in parse_results], dtype=np.float32
        )
        abstraction_batch.non_tensor_batch["abstraction_failure_reason"] = as_object_array(
            result.failure_reason for result in parse_results
        )
        abstraction_batch.non_tensor_batch["abstraction_principle_count"] = np.asarray(
            [result.principle_count for result in parse_results], dtype=np.int32
        )
        abstraction_batch.non_tensor_batch["uid"] = abstraction_batch.non_tensor_batch["problem_uid"].copy()

        solver_ready_idxs = [
            idx
            for idx, abstraction in enumerate(abstraction_batch.non_tensor_batch["solver_conditioning_abstraction"])
            if abstraction is not None
        ]
        solver_batch = None
        reward_extra_infos = {}
        solver_scalar_rewards = {}
        aggregated_extra_metrics = defaultdict(dict)

        if solver_ready_idxs:
            solver_batch = self._build_rollout_input(
                abstraction_batch.select_idxs(solver_ready_idxs).repeat(repeat_times=num_solver_rollouts, interleave=True)
            )
            solver_batch.non_tensor_batch["raw_prompt"] = as_object_array(
                build_single_turn_chat(
                    render_template(
                        self.solver_prompt_template,
                        problem=problem,
                        abstraction=abstraction,
                    )
                )
                for problem, abstraction in zip(
                    solver_batch.non_tensor_batch["problem"],
                    solver_batch.non_tensor_batch["solver_conditioning_abstraction"],
                    strict=True,
                )
            )
            solver_batch.non_tensor_batch["uid"] = solver_batch.non_tensor_batch["abstraction_uid"].copy()
            solver_batch.meta_info["global_steps"] = self.global_steps
            solver_batch.meta_info["temperature"] = (
                self.config.actor_rollout_ref.rollout.temperature if solver_temperature is None else solver_temperature
            )
            solver_batch.meta_info["top_p"] = (
                self.config.actor_rollout_ref.rollout.get("top_p", 1.0) if solver_top_p is None else solver_top_p
            )
            solver_batch.meta_info["top_k"] = (
                self.config.actor_rollout_ref.rollout.get("top_k", 0) if solver_top_k is None else solver_top_k
            )
            total_response_length = int(
                self.config.actor_rollout_ref.rollout.response_length
                if solver_response_length is None
                else solver_response_length
            )
            solver_batch.meta_info["response_length"] = total_response_length

            self.checkpoint_manager.update_weights()
            solver_batch = self._run_solver_rollouts_with_vm_pruning(
                solver_batch=solver_batch,
                timing_raw=timing_raw,
                do_profile=do_profile,
                total_response_length=total_response_length,
                num_solver_rollouts=num_solver_rollouts,
                solver_temperature=solver_temperature,
                solver_top_p=solver_top_p,
                solver_top_k=solver_top_k,
            )
            self.checkpoint_manager.sleep_replicas()

            reward_tensor, reward_extra_infos = extract_reward(solver_batch)
            for abstraction_uid, score in zip(
                solver_batch.non_tensor_batch["abstraction_uid"],
                solver_batch.non_tensor_batch["seq_final_reward"],
                strict=True,
            ):
                solver_scalar_rewards.setdefault(abstraction_uid, []).append(float(score))

            for key, values in reward_extra_infos.items():
                grouped_values = defaultdict(list)
                for abstraction_uid, value in zip(
                    solver_batch.non_tensor_batch["abstraction_uid"],
                    values,
                    strict=True,
                ):
                    try:
                        grouped_values[abstraction_uid].append(float(value))
                    except (TypeError, ValueError):
                        grouped_values = None
                        break
                if grouped_values is not None:
                    for abstraction_uid, metric_values in grouped_values.items():
                        aggregated_extra_metrics[key][abstraction_uid] = float(np.mean(metric_values))
        else:
            print(
                f"Warning: all {len(abstraction_batch)} abstractions were unusable for solver conditioning, "
                "solver update skipped"
            )

        abstraction_rewards = []
        downstream_rewards = []
        validity_bonuses = []
        for abstraction_uid, validity_score in zip(
            abstraction_batch.non_tensor_batch["abstraction_uid"],
            abstraction_batch.non_tensor_batch["abstraction_validity_score"],
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

        abstraction_batch.batch["response_mask"] = compute_response_mask(abstraction_batch)
        abstraction_batch.batch["token_level_scores"] = build_scalar_reward_tensor(
            abstraction_batch.batch["response_mask"],
            torch.tensor(abstraction_rewards, dtype=torch.float32),
        )
        abstraction_batch.non_tensor_batch["seq_reward"] = np.asarray(abstraction_rewards, dtype=np.float32)
        abstraction_batch.non_tensor_batch["seq_final_reward"] = abstraction_batch.non_tensor_batch["seq_reward"].copy()
        abstraction_batch.non_tensor_batch["abstraction_downstream_reward"] = np.asarray(
            downstream_rewards, dtype=np.float32
        )
        abstraction_batch.non_tensor_batch["abstraction_validity_bonus"] = np.asarray(
            validity_bonuses, dtype=np.float32
        )

        for key, abstraction_uid_to_value in aggregated_extra_metrics.items():
            abstraction_batch.non_tensor_batch[key] = np.asarray(
                [abstraction_uid_to_value.get(uid, 0.0) for uid in abstraction_batch.non_tensor_batch["abstraction_uid"]],
                dtype=np.float32,
            )

        return abstraction_batch, solver_batch

    @staticmethod
    def _summarize_batches(abstraction_batch: DataProto, solver_batch: DataProto | None):
        metrics = TwoPolicyGRPOTrainer._summarize_batches(abstraction_batch, solver_batch)
        if solver_batch is not None and len(solver_batch) > 0 and "vm_pruned" in solver_batch.non_tensor_batch:
            metrics["two_policy/vm_pruned_rate"] = float(np.mean(solver_batch.non_tensor_batch["vm_pruned"]))
            metrics["two_policy/vm_proxy_reward_mean"] = float(
                np.mean(solver_batch.non_tensor_batch["vm_proxy_reward"])
            )
            metrics["two_policy/vm_finished_exact_rate"] = float(
                np.mean(solver_batch.non_tensor_batch["vm_finished_exact"])
            )
        return metrics
