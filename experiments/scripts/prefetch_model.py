from pathlib import Path

import hydra
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf


def download_model(model_id: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model repo {model_id} to {target_dir}...")
    local_repo = snapshot_download(
        repo_id=model_id, local_dir=str(target_dir), local_dir_use_symlinks=False
    )
    return Path(local_repo)


@hydra.main(config_path="../conf", config_name="config-assets", version_base=None)
def main(cfg: DictConfig):
    print("=== Prefetch model with config ===")
    print(OmegaConf.to_yaml(cfg))

    model_id = cfg.assets.model.id
    model_dir = Path(cfg.paths.project_root) / Path(cfg.assets.model.target_dir)

    print(model_id)
    print(model_dir)

    # dataset_name = cfg.assets.dataset.name
    # dataset_config = cfg.assets.dataset.config
    # train_split = cfg.assets.dataset.train_split
    # val_split = cfg.assets.dataset.val_split
    # dataset_dir = Path(cfg.assets.dataset.target_dir)
    #
    model_path = download_model(model_id, model_dir)
    # dataset_path = save_dataset(dataset_name, dataset_config, train_split, val_split, dataset_dir)
    #
    # print("\nSummary:")
    print(f"- Model stored at: {model_path}")
    # print(f"- Dataset stored at: {dataset_path}")


if __name__ == "__main__":
    main()
