"""Tests for EvalConfig pydantic model and build_eval_config helper."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing eval package from experiments/scripts
_scripts_dir = Path(__file__).resolve().parents[2] / "experiments" / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from eval.config import (  # noqa: E402
    AdapterConfig,
    DatasetConfig,
    EvalConfig,
    GenerationConfig,
    RAGConfig,
)


class TestEvalConfigDefaults:
    """Verify default values match the eval_config.yaml baseline."""

    def test_default_task(self):
        cfg = EvalConfig()
        assert cfg.task == "chat"

    def test_default_tier(self):
        cfg = EvalConfig()
        assert cfg.tier == "regression"

    def test_default_base_model(self):
        cfg = EvalConfig()
        assert cfg.base_model == "/models/Qwen/Qwen3-0.6B"

    def test_default_dataset(self):
        cfg = EvalConfig()
        assert cfg.dataset.name == "hotpotqa"
        assert cfg.dataset.split == "validation"
        assert cfg.dataset.max_examples == 200
        assert cfg.dataset.seed == 42

    def test_default_generation(self):
        cfg = EvalConfig()
        assert cfg.generation.temperature == 0.1
        assert cfg.generation.top_p == 0.95
        assert cfg.generation.max_tokens == 512

    def test_default_adapter_is_none(self):
        cfg = EvalConfig()
        assert cfg.adapter.name is None
        assert cfg.adapter.version is None

    def test_default_rag_enabled(self):
        cfg = EvalConfig()
        assert cfg.rag.enabled is True
        assert cfg.rag.knowledge_base == "arxiv"

    def test_default_metrics(self):
        cfg = EvalConfig()
        assert cfg.metrics.bert_score_model == "roberta-large"


class TestEvalConfigCustom:
    """Verify custom config values are properly applied."""

    def test_custom_task(self):
        cfg = EvalConfig(task="summarize", tier="full")
        assert cfg.task == "summarize"
        assert cfg.tier == "full"

    def test_custom_adapter(self):
        cfg = EvalConfig(adapter=AdapterConfig(name="lora-summ", version=3))
        assert cfg.adapter.name == "lora-summ"
        assert cfg.adapter.version == 3

    def test_custom_rag_disabled(self):
        cfg = EvalConfig(rag=RAGConfig(enabled=False, knowledge_base=None))
        assert cfg.rag.enabled is False
        assert cfg.rag.knowledge_base is None

    def test_custom_generation_params(self):
        cfg = EvalConfig(generation=GenerationConfig(temperature=0.7, max_tokens=1024))
        assert cfg.generation.temperature == 0.7
        assert cfg.generation.max_tokens == 1024

    def test_custom_dataset_no_limit(self):
        cfg = EvalConfig(dataset=DatasetConfig(name="arxiv-summarization", max_examples=None))
        assert cfg.dataset.max_examples is None


class TestEvalConfigSerialization:
    """Verify serialization for JSONB storage."""

    def test_model_dump_json_roundtrip(self):
        cfg = EvalConfig(
            task="chat",
            adapter=AdapterConfig(name="lora-qa", version=5),
            rag=RAGConfig(retrieval_top_k=10),
        )
        data = cfg.model_dump(mode="json")
        assert isinstance(data, dict)
        assert data["task"] == "chat"
        assert data["adapter"]["name"] == "lora-qa"
        assert data["adapter"]["version"] == 5
        assert data["rag"]["retrieval_top_k"] == 10

    def test_all_fields_serialized(self):
        cfg = EvalConfig()
        data = cfg.model_dump(mode="json")
        expected_keys = {
            "base_model", "vllm_base_url", "adapter", "rag",
            "task", "tier", "dataset", "task_metrics", "judge",
            "generation", "metrics", "db_url",
        }
        assert expected_keys == set(data.keys())

    def test_nested_objects_are_dicts(self):
        cfg = EvalConfig()
        data = cfg.model_dump(mode="json")
        assert isinstance(data["adapter"], dict)
        assert isinstance(data["rag"], dict)
        assert isinstance(data["generation"], dict)
        assert isinstance(data["dataset"], dict)


class TestBuildEvalConfig:
    """Test building EvalConfig from OmegaConf DictConfig."""

    def test_build_from_omegaconf(self):
        from eval.run_eval import build_eval_config
        from omegaconf import OmegaConf

        raw = {
            "model": {"base_model": "test-model", "vllm_base_url": "http://test:8000"},
            "adapter": {"name": "adapter-a", "version": 2},
            "rag": {"enabled": False, "knowledge_base": None},
            "task": "summarize",
            "tier": "full",
            "dataset": {"name": "arxiv", "split": "validation", "max_examples": 100, "seed": 42},
            "task_metrics": ["rouge_l", "bert_score"],
            "judge": {"enabled": False},
            "generation": {"temperature": 0.5, "top_p": 0.9, "max_tokens": 256},
            "metrics": {"bert_score_model": "roberta-large"},
            "db_url": "postgresql://localhost/test",
        }
        cfg = OmegaConf.create(raw)
        eval_cfg = build_eval_config(cfg)

        assert eval_cfg.base_model == "test-model"
        assert eval_cfg.adapter.name == "adapter-a"
        assert eval_cfg.adapter.version == 2
        assert eval_cfg.rag.enabled is False
        assert eval_cfg.task == "summarize"
        assert eval_cfg.tier == "full"
        assert eval_cfg.dataset.max_examples == 100
        assert eval_cfg.generation.temperature == 0.5
        assert eval_cfg.db_url == "postgresql://localhost/test"

    def test_build_with_minimal_config(self):
        from eval.run_eval import build_eval_config
        from omegaconf import OmegaConf

        raw = {
            "model": {"base_model": "m", "vllm_base_url": "http://x"},
            "task": "chat",
            "tier": "regression",
            "dataset": {"name": "d", "split": "s"},
            "generation": {"temperature": 0.1, "top_p": 0.95, "max_tokens": 512},
            "db_url": "postgresql://localhost/db",
        }
        cfg = OmegaConf.create(raw)
        eval_cfg = build_eval_config(cfg)
        assert eval_cfg.task == "chat"


class TestMakeSyncUrl:
    """Test DB URL conversion from async to sync driver."""

    def test_asyncpg_to_psycopg2(self):
        from eval.run_eval import _make_sync_url

        url = "postgresql+asyncpg://user:pass@host:5432/db"
        assert _make_sync_url(url) == "postgresql+psycopg2://user:pass@host:5432/db"

    def test_plain_postgresql_unchanged(self):
        from eval.run_eval import _make_sync_url

        url = "postgresql://user:pass@host:5432/db"
        assert _make_sync_url(url) == "postgresql://user:pass@host:5432/db"

    def test_psycopg2_unchanged(self):
        from eval.run_eval import _make_sync_url

        url = "postgresql+psycopg2://user:pass@host/db"
        assert _make_sync_url(url) == "postgresql+psycopg2://user:pass@host/db"
