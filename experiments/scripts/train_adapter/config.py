from dataclasses import dataclass
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class PathsConfig:
    project_root: str


@dataclass
class ModelConfig:
    dtype: str
    device_map: str
    load_in_4bit: bool
    bnb_4bit_use_double_quant: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: Optional[str]
    gradient_checkpointing: bool
    local_path: str
    offload_folder: Optional[str]


@dataclass
class LoraSection:
    r: int
    lora_alpha: float
    lora_dropout: float
    target_modules: List[str]


@dataclass
class DataConfig:
    max_seq_length: int
    batch_size: int
    num_workers: int
    local_path: str
    prompt_template: str


@dataclass
class TrainingConfig:
    lr: float
    weight_decay: float


@dataclass
class SchedulerConfig:
    enabled: bool
    type: str
    warmup_steps: int
    start_factor: float
    interval: str
    frequency: int
    T_max: int
    eta_min: float


@dataclass
class TrainerConfig:
    max_epochs: int
    devices: int
    accelerator: str
    precision: str
    gradient_clip_val: float
    accumulate_grad_batches: int
    log_every_n_steps: int
    val_check_interval: float | int


@dataclass
class OutputConfig:
    save_dir: str


@dataclass
class MlflowConfig:
    log_artifacts: bool
    log_metrics: bool
    log_params: bool
    experiment_name: str
    run_name: str
    env_path: Optional[str]


@dataclass
class ExperimentConfig:
    seed: Optional[int]
    model: ModelConfig
    lora: LoraSection
    data: DataConfig
    training: TrainingConfig
    trainer: TrainerConfig
    output: OutputConfig
    mlflow: MlflowConfig
    scheduler: Optional[SchedulerConfig] = None


@dataclass
class AppConfig:
    paths: PathsConfig
    experiment: ExperimentConfig


def load_app_config(cfg: DictConfig) -> AppConfig:
    """Convert a Hydra DictConfig into structured dataclasses."""
    structured = OmegaConf.structured(AppConfig)
    merged = OmegaConf.merge(structured, cfg)
    return OmegaConf.to_object(merged)
