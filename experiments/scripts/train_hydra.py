from __future__ import annotations


import hydra
from omegaconf import DictConfig

from train_adapter.config import load_app_config
from train_adapter.pipeline import run_training


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    app_cfg = load_app_config(cfg)
    save_dir, logs_dir = run_training(app_cfg)

    print(f"Saved adapter to: {save_dir}")
    print(f"Lightning logs: {logs_dir}")


if __name__ == "__main__":
    main()