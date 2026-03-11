"""Tests for EvalConfig pydantic model."""

from __future__ import annotations

import pytest
from eval.config import EvalConfig


class TestEvalConfigDefaults:
    """Verify default values and required fields."""

    def test_minimal_construction(self):
        """Construct with only required fields."""
        cfg = EvalConfig(
            base_model="Qwen/Qwen3-0.6B",
            dataset_name="hotpotqa",
            task="chat",
        )
        assert cfg.base_model == "Qwen/Qwen3-0.6B"
        assert cfg.dataset_name == "hotpotqa"
        assert cfg.task == "chat"
        assert cfg.tier == "regression"
        assert cfg.temperature == 0.1
        assert cfg.top_p == 0.95
        assert cfg.max_tokens == 512
        assert cfg.seed == 42
        assert cfg.bert_score_model == "roberta-large"

    def test_adapter_fields_nullable(self):
        cfg = EvalConfig(
            base_model="Qwen/Qwen3-0.6B",
            dataset_name="hotpotqa",
            task="chat",
        )
        assert cfg.adapter_name is None
        assert cfg.adapter_version is None
        assert cfg.adapter_mlflow_run_id is None

    def test_full_construction(self):
        cfg = EvalConfig(
            base_model="Qwen/Qwen3-0.6B",
            dataset_name="arxiv-summarization",
            task="summarize",
            tier="full",
            adapter_name="lora-summarization",
            adapter_version=3,
            rag_enabled=True,
            knowledge_base="arxiv",
            max_examples=None,
            judge_enabled=True,
            judge_model="gemini-2.0-flash",
            temperature=0.3,
        )
        assert cfg.adapter_name == "lora-summarization"
        assert cfg.adapter_version == 3
        assert cfg.tier == "full"
        assert cfg.judge_enabled is True
        assert cfg.judge_model == "gemini-2.0-flash"
        assert cfg.temperature == 0.3


class TestEvalConfigFrozen:
    """EvalConfig should be immutable."""

    def test_immutability(self):
        cfg = EvalConfig(
            base_model="Qwen/Qwen3-0.6B",
            dataset_name="hotpotqa",
            task="chat",
        )
        with pytest.raises(Exception):
            cfg.temperature = 0.5


class TestEvalConfigSerialization:
    """model_dump / model_dump_json round-trip."""

    def test_round_trip(self):
        cfg = EvalConfig(
            base_model="Qwen/Qwen3-0.6B",
            dataset_name="hotpotqa",
            task="chat",
            adapter_name="lora-chat",
            adapter_version=2,
        )
        data = cfg.model_dump()
        restored = EvalConfig(**data)
        assert restored == cfg

    def test_json_serializable(self):
        cfg = EvalConfig(
            base_model="Qwen/Qwen3-0.6B",
            dataset_name="hotpotqa",
            task="chat",
        )
        json_str = cfg.model_dump_json()
        assert "Qwen/Qwen3-0.6B" in json_str
        assert "hotpotqa" in json_str


class TestEvalConfigRequiredFields:
    """base_model, dataset_name, and task are required."""

    def test_missing_base_model(self):
        with pytest.raises(Exception):
            EvalConfig(dataset_name="hotpotqa", task="chat")

    def test_missing_dataset_name(self):
        with pytest.raises(Exception):
            EvalConfig(base_model="Qwen/Qwen3-0.6B", task="chat")

    def test_missing_task(self):
        with pytest.raises(Exception):
            EvalConfig(base_model="Qwen/Qwen3-0.6B", dataset_name="hotpotqa")
