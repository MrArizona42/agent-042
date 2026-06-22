"""Versioned project identity around LlamaIndex prompt templates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_REFINE_PROMPT, DEFAULT_TEXT_QA_PROMPT
from pydantic import BaseModel, ConfigDict, Field


class PromptIdentity(BaseModel):
    """Stable identity persisted with generation observations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_id: str
    prompt_version: str
    prompt_digest: str
    prompt_params: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProjectQueryPrompts:
    """LlamaIndex query templates plus project-owned version identity."""

    identity: PromptIdentity
    text_qa_template: PromptTemplate
    refine_template: PromptTemplate

    @classmethod
    def create(
        cls,
        *,
        prompt_id: str,
        prompt_version: str,
        text_qa_template: str,
        refine_template: str,
        prompt_params: dict[str, Any] | None = None,
    ) -> "ProjectQueryPrompts":
        params = dict(prompt_params or {})
        canonical = json.dumps(
            {
                "prompt_id": prompt_id,
                "prompt_version": prompt_version,
                "text_qa_template": text_qa_template,
                "refine_template": refine_template,
                "prompt_params": params,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            identity=PromptIdentity(
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                prompt_digest=f"sha256:{hashlib.sha256(canonical).hexdigest()}",
                prompt_params=params,
            ),
            text_qa_template=PromptTemplate(text_qa_template),
            refine_template=PromptTemplate(refine_template),
        )


DEFAULT_RAG_QUERY_PROMPTS = ProjectQueryPrompts.create(
    prompt_id="rag.query.default",
    prompt_version="1",
    text_qa_template=DEFAULT_TEXT_QA_PROMPT.get_template(),
    refine_template=DEFAULT_REFINE_PROMPT.get_template(),
    prompt_params={"response_mode": "compact"},
)
