from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Domain configs — no _target_, used via typed Python access
# ---------------------------------------------------------------------------


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
    source_max_length: int
    target_max_length: int
    train_on_inputs: bool
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


# ---------------------------------------------------------------------------
# Instantiable configs — _target_ lets hydra.utils.instantiate build objects
# ---------------------------------------------------------------------------


@dataclass
class CheckpointConf:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    filename: str = "adapter-{epoch:02d}-{val_loss:.4f}"
    save_top_k: int = 3
    monitor: str = "val_loss"
    mode: str = "min"
    save_last: bool = True


@dataclass
class EarlyStoppingConf:
    _target_: str = "pytorch_lightning.callbacks.EarlyStopping"
    monitor: str = "val_loss"
    patience: int = 5
    mode: str = "min"


@dataclass
class CallbacksConf:
    checkpoint: CheckpointConf = field(default_factory=CheckpointConf)
    early_stopping: Optional[EarlyStoppingConf] = None


@dataclass
class TrainerConf:
    _target_: str = "pytorch_lightning.Trainer"
    max_epochs: int = 1
    devices: int = 1
    accelerator: str = "gpu"
    precision: str = "16-mixed"
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 8
    log_every_n_steps: int = 10
    val_check_interval: Any = 0.25
    num_sanity_val_steps: int = 0


# ---------------------------------------------------------------------------
# Top-level composite configs
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    seed: Optional[int]
    model: ModelConfig
    lora: LoraSection
    data: DataConfig
    training: TrainingConfig
    trainer: TrainerConf
    callbacks: CallbacksConf
    output: OutputConfig
    mlflow: MlflowConfig
    scheduler: Optional[SchedulerConfig] = None


@dataclass
class AppConfig:
    paths: PathsConfig
    experiment: ExperimentConfig


# ---------------------------------------------------------------------------
# Hydra ConfigStore registration
# ---------------------------------------------------------------------------


def register_configs() -> None:
    """Register structured configs so Hydra validates YAML at load time."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=AppConfig)


def load_app_config(cfg: DictConfig) -> AppConfig:
    """Convert a Hydra DictConfig into structured dataclasses."""
    structured = OmegaConf.structured(AppConfig)
    merged = OmegaConf.merge(structured, cfg)
    return OmegaConf.to_object(merged)
