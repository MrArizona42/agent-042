from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from .config import register_configs
from .pipeline import run_training

register_configs()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    skip_post_train_evaluation = os.getenv("TRAIN_ADAPTER_SKIP_POST_TRAIN_EVAL", "").lower() in {
        "1",
        "true",
        "yes",
    }
    run_id, save_dir, run_artifacts_dir, manifest_path = run_training(
        cfg,
        skip_post_train_evaluation=skip_post_train_evaluation,
    )

    print(f"Saved adapter to: {save_dir}")
    print(f"Run artifacts: {run_artifacts_dir}")
    print(f"training_manifest={manifest_path}")
    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()
