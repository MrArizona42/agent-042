from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Domain configs — no _target_, used via typed Python access
# ---------------------------------------------------------------------------


@dataclass
class PathsConfig:
    project_root: str = MISSING


@dataclass
class TaskConfig:
    name: str = MISSING
    run_name_prefix: str = MISSING
    prompt_template: str = MISSING
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    local_path: str = MISSING
    train_split: str = MISSING
    validation_split: Optional[str] = None
    validation_fraction: float = 0.0
    split_seed: int = 42
    prompt_field_map: dict[str, str] = MISSING
    target_field: str = MISSING
    name: Optional[str] = None


@dataclass
class ModelConfig:
    dtype: str = "float16"
    device_map: str = "auto"
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: Optional[str] = "float16"
    gradient_checkpointing: bool = True
    tokenizer_use_fast: bool = True
    local_files_only: bool = True
    trust_remote_code: bool = False
    local_path: str = MISSING
    name: Optional[str] = None
    offload_folder: Optional[str] = "${paths.project_root}/assets/models/offload"


@dataclass
class LoraSection:
    r: int = MISSING
    lora_alpha: float = MISSING
    lora_dropout: float = MISSING
    target_modules: List[str] = field(default_factory=list)


@dataclass
class DataConfig:
    max_seq_length: int = MISSING
    source_max_length: int = MISSING
    target_max_length: int = MISSING
    train_on_inputs: bool = MISSING
    batch_size: int = MISSING
    num_workers: int = MISSING


@dataclass
class TrainingConfig:
    seed: Optional[int] = MISSING
    lr: float = MISSING
    weight_decay: float = MISSING


@dataclass
class SchedulerConfig:
    enabled: bool = MISSING
    type: str = MISSING
    warmup_steps: int = MISSING
    start_factor: float = MISSING
    interval: str = MISSING
    frequency: int = MISSING
    T_max: int = MISSING
    eta_min: float = MISSING


@dataclass
class TrackingConfig:
    pipeline_name: str = "train_adapter"
    log_artifacts: bool = True
    log_metrics: bool = True
    log_params: bool = True
    env_path: Optional[str] = ".env"


@dataclass
class DataModuleConf:
    _target_: str = "experiments.training.train_adapter.data_module.PromptTargetDataModule"
    shuffle: bool = True


@dataclass
class MLFlowLoggerConf:
    _target_: str = "pytorch_lightning.loggers.MLFlowLogger"
    experiment_name: str = "default"
    run_name: Optional[str] = None
    log_model: bool = False
    tags: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Instantiable configs — _target_ lets hydra.utils.instantiate build objects
# ---------------------------------------------------------------------------


@dataclass
class CheckpointConf:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    filename: str = MISSING
    save_top_k: int = MISSING
    monitor: str = MISSING
    mode: str = MISSING
    save_last: bool = MISSING


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
    max_epochs: int = MISSING
    devices: int = MISSING
    accelerator: str = MISSING
    precision: str = MISSING
    gradient_clip_val: float = MISSING
    accumulate_grad_batches: int = MISSING
    log_every_n_steps: int = MISSING
    val_check_interval: Any = MISSING
    num_sanity_val_steps: int = MISSING
    limit_train_batches: Any = MISSING
    limit_val_batches: Any = MISSING
    enable_progress_bar: bool = MISSING


# ---------------------------------------------------------------------------
# Top-level composite configs
# ---------------------------------------------------------------------------


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraSection = field(default_factory=LoraSection)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: Optional[SchedulerConfig] = None
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    data_module: DataModuleConf = field(default_factory=DataModuleConf)
    trainer: TrainerConf = field(default_factory=TrainerConf)
    callbacks: CallbacksConf = field(default_factory=CallbacksConf)
    logger: MLFlowLoggerConf = field(default_factory=MLFlowLoggerConf)


# ---------------------------------------------------------------------------
# Hydra ConfigStore registration
# ---------------------------------------------------------------------------


_CONFIGS_REGISTERED = False


def register_configs() -> None:
    """Register structured configs so Hydra validates YAML at load time."""
    global _CONFIGS_REGISTERED
    if _CONFIGS_REGISTERED:
        return

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=AppConfig)
    cs.store(group="paths", name="base_paths", node=PathsConfig)
    cs.store(group="task", name="base_task", node=TaskConfig)
    cs.store(group="dataset", name="base_dataset", node=DatasetConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="lora", name="base_lora", node=LoraSection)
    cs.store(group="data", name="base_data", node=DataConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="scheduler", name="base_scheduler", node=SchedulerConfig)
    cs.store(group="trainer", name="base_trainer", node=TrainerConf)
    cs.store(group="callbacks", name="base_callbacks", node=CallbacksConf)
    cs.store(group="logger", name="base_logger", node=MLFlowLoggerConf)
    _CONFIGS_REGISTERED = True


def load_app_config(cfg: DictConfig) -> AppConfig:
    """Convert a Hydra DictConfig into structured dataclasses."""
    structured = OmegaConf.structured(AppConfig)
    merged = OmegaConf.merge(structured, cfg)
    return OmegaConf.to_object(merged)
