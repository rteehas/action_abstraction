import hydra
import ray

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device

from .main_two_policy_ppo import TwoPolicyTaskRunner, persist_resolved_run_config


@hydra.main(config_path='config', config_name='two_policy_dapo_trainer', version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    persist_resolved_run_config(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(TwoPolicyTaskRunner))


if __name__ == '__main__':
    main()
