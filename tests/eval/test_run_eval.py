"""Tests for the run_eval config builder."""

from __future__ import annotations

from eval.config import EvalConfig
from omegaconf import OmegaConf

# We can't easily test full Hydra resolution in unit tests, but we can test
# the build_eval_config function with a hand-crafted OmegaConf dict.

def _sample_cfg() -> dict:
    """Return a dict matching the structure of a resolved eval_config.yaml."""
    return {
        "model": {"base_model": "/models/Qwen/Qwen3-0.6B", "vllm_base_url": "http://localhost:8000"},
        "adapter": {"name": None, "version": None},
        "tier": "regression",
        "task": "chat",
        "dataset": {"name": "hotpotqa", "split": "validation", "max_examples": 200, "seed": 42},
        "generation": {"temperature": 0.1, "top_p": 0.95, "max_tokens": 512},
        "rag": {
            "enabled": True,
            "knowledge_base": "arxiv",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunking_strategy": "fixed_token",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "retrieval_top_k": 5,
            "score_threshold": 0.35,
            "reranking_strategy": "none",
        },
        "judge": {"enabled": False, "model": None},
        "metrics": {"bert_score_model": "roberta-large"},
        "db_url": "postgresql+asyncpg://user:pass@localhost/agent042",
    }


class TestBuildEvalConfig:
    def test_default_chat_config(self):
        # Inline import to avoid Hydra decorator issues
        from eval.run_eval import build_eval_config

        cfg = OmegaConf.create(_sample_cfg())
        eval_config = build_eval_config(cfg)

        assert isinstance(eval_config, EvalConfig)
        assert eval_config.base_model == "/models/Qwen/Qwen3-0.6B"
        assert eval_config.task == "chat"
        assert eval_config.tier == "regression"
        assert eval_config.dataset_name == "hotpotqa"
        assert eval_config.max_examples == 200
        assert eval_config.rag_enabled is True
        assert eval_config.knowledge_base == "arxiv"
        assert eval_config.judge_enabled is False
        assert eval_config.judge_model is None

    def test_summarize_with_adapter(self):
        from eval.run_eval import build_eval_config

        raw = _sample_cfg()
        raw["task"] = "summarize"
        raw["tier"] = "full"
        raw["adapter"]["name"] = "lora-summarization"
        raw["adapter"]["version"] = 3
        raw["dataset"]["name"] = "arxiv-summarization"
        raw["dataset"]["max_examples"] = None

        cfg = OmegaConf.create(raw)
        eval_config = build_eval_config(cfg)

        assert eval_config.task == "summarize"
        assert eval_config.tier == "full"
        assert eval_config.adapter_name == "lora-summarization"
        assert eval_config.adapter_version == 3
        assert eval_config.max_examples is None

    def test_code_no_rag_no_judge(self):
        from eval.run_eval import build_eval_config

        raw = _sample_cfg()
        raw["task"] = "code"
        raw["rag"]["enabled"] = False
        raw["rag"]["knowledge_base"] = None
        raw["judge"]["enabled"] = False
        raw["dataset"]["name"] = "humaneval"
        raw["dataset"]["split"] = "test"
        raw["dataset"]["max_examples"] = None

        cfg = OmegaConf.create(raw)
        eval_config = build_eval_config(cfg)

        assert eval_config.task == "code"
        assert eval_config.rag_enabled is False
        assert eval_config.knowledge_base is None
        assert eval_config.judge_enabled is False
