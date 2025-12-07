from pathlib import Path

import hydra
from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig, OmegaConf


def save_dataset(
    dataset_name: str, dataset_config: str, train_split: str, val_split: str, target_dir: Path
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset {dataset_name}:{dataset_config}...")
    ds_train = load_dataset(dataset_name, dataset_config, split=train_split)
    ds_val = load_dataset(dataset_name, dataset_config, split=val_split)
    ds = DatasetDict({"train": ds_train, "validation": ds_val})
    print(f"Saving dataset to disk at: {target_dir}")
    ds.save_to_disk(str(target_dir))
    return target_dir


@hydra.main(config_path="../conf", config_name="config-assets", version_base=None)
def main(cfg: DictConfig):
    print("=== Prefetch data with config ===")
    print(OmegaConf.to_yaml(cfg))

    project_root = Path(cfg.paths.project_root)

    dataset_name = cfg.assets.dataset.name
    dataset_config = cfg.assets.dataset.config
    train_split = cfg.assets.dataset.train_split
    val_split = cfg.assets.dataset.val_split
    dataset_target = Path(cfg.assets.dataset.target_dir)
    dataset_dir = dataset_target if dataset_target.is_absolute() else project_root / dataset_target
    print(f"Dataset dir (resolved): {dataset_dir}")

    dataset_path = save_dataset(dataset_name, dataset_config, train_split, val_split, dataset_dir)

    print("\nSummary:")
    print(f"- Dataset stored at: {dataset_path}")


if __name__ == "__main__":
    main()
