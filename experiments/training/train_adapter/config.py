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
class TrackingConfig:
    log_artifacts: bool
    log_metrics: bool
    log_params: bool
    env_path: Optional[str]


@dataclass
class EvaluationConfig:
    enabled: bool = True
    task: str = "summarize"
    dataset_name: str = "arxiv_summarization"
    metrics: List[str] = field(default_factory=lambda: ["rouge_l"])
    sample_limit: int = 32
    batch_size: int = 1
    max_new_tokens: int = 256
    temperature: float = 0.0
    do_sample: bool = False
    fail_on_error: bool = True


@dataclass
class DataModuleConf:
    _target_: str = "experiments.training.train_adapter.data_module.ArxivDataModule"
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
    limit_train_batches: Any = 1.0
    limit_val_batches: Any = 1.0
    enable_progress_bar: bool = True


# ---------------------------------------------------------------------------
# Top-level composite configs
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    seed: Optional[int]
    model: ModelConfig
    lora: LoraSection
    data: DataConfig
    data_module: DataModuleConf
    training: TrainingConfig
    trainer: TrainerConf
    callbacks: CallbacksConf
    logger: MLFlowLoggerConf
    tracking: TrackingConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    scheduler: Optional[SchedulerConfig] = None


@dataclass
class AppConfig:
    paths: PathsConfig
    experiment: ExperimentConfig


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
    cs.store(group="experiment", name="base_experiment", node=ExperimentConfig)
    _CONFIGS_REGISTERED = True


def load_app_config(cfg: DictConfig) -> AppConfig:
    """Convert a Hydra DictConfig into structured dataclasses."""
    structured = OmegaConf.structured(AppConfig)
    merged = OmegaConf.merge(structured, cfg)
    return OmegaConf.to_object(merged)
