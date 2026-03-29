from __future__ import annotations

import hydra
from omegaconf import DictConfig
from .config import load_app_config
from .pipeline import run_training


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    app_cfg = load_app_config(cfg)
    run_id, save_dir, logs_dir = run_training(app_cfg)

    print(f"Saved adapter to: {save_dir}")
    print(f"Lightning logs: {logs_dir}")
    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()
