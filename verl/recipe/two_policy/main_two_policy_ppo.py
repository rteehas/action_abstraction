import copy
import os
import socket
import uuid

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

ABSTRACTION_ACTOR_ROLE = 'abstraction_actor_rollout'
ABSTRACTION_REF_ROLE = 'abstraction_ref'


def _set_rollout_server_name_prefix(branch_cfg, prefix: str):
    actor_rollout_ref = branch_cfg.get('actor_rollout_ref')
    rollout_cfg = actor_rollout_ref.rollout if actor_rollout_ref is not None else branch_cfg.rollout
    OmegaConf.update(rollout_cfg, 'custom.server_name_prefix', prefix, force_add=True)


def _clone_with_actor_rollout_ref(config, actor_rollout_ref):
    cloned = copy.deepcopy(config)
    cloned.actor_rollout_ref = copy.deepcopy(actor_rollout_ref)
    return cloned


class TwoPolicyTaskRunner(TaskRunner):
    def _add_branch_actor_rollout_worker(self, branch_cfg, role_key: str, pool_name: str):
        from verl.single_controller.ray import RayWorkerGroup

        use_legacy_worker_impl = branch_cfg.trainer.get('use_legacy_worker_impl', 'auto')
        if use_legacy_worker_impl == 'disable':
            raise ValueError('Two-policy trainer currently supports the legacy worker path only')

        if branch_cfg.actor_rollout_ref.actor.strategy in {'fsdp', 'fsdp2'}:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif branch_cfg.actor_rollout_ref.actor.strategy == 'megatron':
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError

        self.role_worker_mapping[role_key] = ray.remote(actor_rollout_cls)
        self.mapping[role_key] = pool_name
        return actor_rollout_cls, ray_worker_group_cls

    def _add_branch_ref_worker(self, branch_cfg, role_key: str, ref_policy_cls, pool_name: str):
        use_legacy_worker_impl = branch_cfg.trainer.get('use_legacy_worker_impl', 'auto')
        if use_legacy_worker_impl == 'disable':
            return
        if need_reference_policy(branch_cfg):
            self.role_worker_mapping[role_key] = ray.remote(ref_policy_cls)
            self.mapping[role_key] = pool_name

    def init_resource_pool_mgr(self, config):
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        pool_cfg = config.two_policy.resource_pools
        if pool_cfg.mode == 'colocate':
            resource_pool_spec = {'global_pool': [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
            if config.reward.reward_model.enable_resource_pool:
                resource_pool_spec['reward_pool'] = [
                    config.reward.reward_model.n_gpus_per_node
                ] * config.reward.reward_model.nnodes
        elif pool_cfg.mode == 'split':
            solver_nnodes = pool_cfg.solver_nnodes or config.trainer.nnodes
            abstraction_nnodes = pool_cfg.abstraction_nnodes or config.trainer.nnodes
            solver_gpus = pool_cfg.solver_gpus_per_node
            abstraction_gpus = pool_cfg.abstraction_gpus_per_node
            if solver_gpus is None or abstraction_gpus is None:
                if config.trainer.n_gpus_per_node >= 2:
                    solver_gpus = config.trainer.n_gpus_per_node // 2
                    abstraction_gpus = config.trainer.n_gpus_per_node - solver_gpus
                else:
                    raise ValueError(
                        'split resource pools require explicit per-branch gpu counts when only 1 GPU is available'
                    )
            resource_pool_spec = {
                pool_cfg.solver_pool_id: [solver_gpus] * solver_nnodes,
                pool_cfg.abstraction_pool_id: [abstraction_gpus] * abstraction_nnodes,
            }
            if config.reward.reward_model.enable_resource_pool:
                resource_pool_spec['reward_pool'] = [
                    config.reward.reward_model.n_gpus_per_node
                ] * config.reward.reward_model.nnodes
        else:
            raise ValueError(f'Unsupported resource pool mode: {pool_cfg.mode}')

        if config.reward.reward_model.enable and not config.reward.reward_model.enable_resource_pool:
            self.mapping[Role.RewardModel] = pool_cfg.solver_pool_id

        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def add_reward_model_resource_pool(self, config):
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward.reward_model.enable:
            if config.reward.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = 'reward_pool'
            else:
                self.mapping[Role.RewardModel] = config.two_policy.resource_pools.solver_pool_id

    def run(self, config):
        from pprint import pprint

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        print(f'TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}')
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        if need_critic(config):
            raise ValueError('Two-policy trainer currently supports GRPO/DAPO-style actor-only training only')
        if config.trainer.get('use_legacy_worker_impl', 'auto') == 'disable':
            raise ValueError('Two-policy trainer currently supports the legacy worker path only')

        run_token = uuid.uuid4().hex[:8]
        _set_rollout_server_name_prefix(config, f'solver_{run_token}')
        _set_rollout_server_name_prefix(config.abstraction_actor_rollout_ref, f'abstraction_{run_token}')

        abstraction_pool = config.two_policy.resource_pools.abstraction_pool_id

        solver_actor_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        abstraction_cfg = _clone_with_actor_rollout_ref(config, config.abstraction_actor_rollout_ref)
        abstraction_actor_cls, abstraction_wg_cls = self._add_branch_actor_rollout_worker(
            abstraction_cfg, ABSTRACTION_ACTOR_ROLE, abstraction_pool
        )
        if abstraction_wg_cls is not ray_worker_group_cls:
            raise ValueError('Solver and abstraction branches must use the same worker-group backend')

        self.add_reward_model_resource_pool(config)
        self.add_ref_policy_worker(config, solver_actor_cls)
        self._add_branch_ref_worker(abstraction_cfg, ABSTRACTION_REF_ROLE, abstraction_actor_cls, abstraction_pool)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=False,
        )
        validate_config(
            config=abstraction_cfg,
            use_reference_policy=need_reference_policy(abstraction_cfg),
            use_critic=False,
        )

        trust_remote_code = config.data.get('trust_remote_code', False)
        abstraction_local_path = copy_to_local(
            config.abstraction_actor_rollout_ref.model.path,
            use_shm=config.abstraction_actor_rollout_ref.model.get('use_shm', False),
        )
        solver_local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get('use_shm', False),
        )

        abstraction_tokenizer = hf_tokenizer(abstraction_local_path, trust_remote_code=trust_remote_code)
        abstraction_processor = hf_processor(abstraction_local_path, trust_remote_code=trust_remote_code, use_fast=True)
        solver_tokenizer = hf_tokenizer(solver_local_path, trust_remote_code=trust_remote_code)
        solver_processor = hf_processor(solver_local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            abstraction_tokenizer,
            abstraction_processor,
            is_train=True,
            max_samples=config.data.get('train_max_samples', -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            abstraction_tokenizer,
            abstraction_processor,
            is_train=False,
            max_samples=config.data.get('val_max_samples', -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from .two_policy_trainer import TwoPolicyGRPOTrainer

        trainer = TwoPolicyGRPOTrainer(
            config=config,
            tokenizer=solver_tokenizer,
            abstraction_tokenizer=abstraction_tokenizer,
            processor=solver_processor,
            abstraction_processor=abstraction_processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path='config', config_name='two_policy_trainer', version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(TwoPolicyTaskRunner))


if __name__ == '__main__':
    main()
