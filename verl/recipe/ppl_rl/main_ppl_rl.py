from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device


def persist_resolved_run_config(config) -> None:
    checkpoint_dir = OmegaConf.select(config, "trainer.default_local_dir")
    if not checkpoint_dir:
        return

    run_dir = Path(os.path.expanduser(str(checkpoint_dir))).parent
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = OmegaConf.to_container(config, resolve=True, throw_on_missing=False)
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved_config, indent=2, sort_keys=True) + "\n")
    (run_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))

    try:
        hydra_overrides = HydraConfig.get().overrides.task
    except Exception:
        hydra_overrides = []
    if hydra_overrides:
        (run_dir / "hydra_overrides.txt").write_text("\n".join(hydra_overrides) + "\n")


@hydra.main(config_path="config", config_name="ppl_rl_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    persist_resolved_run_config(config)
    run_ppo(config)


if __name__ == "__main__":
    main()
