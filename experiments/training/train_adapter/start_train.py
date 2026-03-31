from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .config import register_configs
from .pipeline import run_training

register_configs()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_id, save_dir, run_artifacts_dir, summary_path = run_training(cfg)
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))

    print(f"Saved adapter to: {save_dir}")
    print(f"Run artifacts: {run_artifacts_dir}")
    print(f"training_summary={summary_path}")
    if summary.get("best_val_loss") is not None:
        print(f"best_val_loss={summary['best_val_loss']}")
    elif summary.get("best_model_score") is not None:
        print(f"best_model_score={summary['best_model_score']}")
    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()
