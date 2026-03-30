from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .config import register_configs
from .pipeline import run_training

register_configs()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_id, save_dir, logs_dir = run_training(cfg)

    print(f"Saved adapter to: {save_dir}")
    print(f"Lightning logs: {logs_dir}")
    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()
