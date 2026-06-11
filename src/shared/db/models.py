"""SQLAlchemy ORM models for the agent042 database.

Tables
------
- users            — authenticated users (Google OIDC)
- chat_sessions    — per-user conversation sessions
- chat_messages    — individual messages within a session
- eval_runs        — evaluation benchmark results
- eval_samples     — per-sample evaluation details
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    sub: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str | None] = mapped_column(Text)
    name: Mapped[str | None] = mapped_column(Text)
    picture: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (UniqueConstraint("provider", "sub", name="uq_provider_sub"),)

    sessions: Mapped[list[ChatSession]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    user: Mapped[User] = relationship(back_populates="sessions")
    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class EvalRun(Base):
    """One evaluation metric result per (task, dataset, metric, rag_alias, lora_alias)."""

    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text, nullable=False, default="running")

    # Task / dataset / metric
    task: Mapped[str] = mapped_column(Text, nullable=False)
    dataset_name: Mapped[str] = mapped_column(Text, nullable=False)
    metric_name: Mapped[str] = mapped_column(Text, nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)

    # Model
    base_model: Mapped[str] = mapped_column(Text, nullable=False)
    adapter_name: Mapped[str | None] = mapped_column(Text)
    adapter_version: Mapped[int | None] = mapped_column(Integer)
    adapter_mlflow_run_id: Mapped[str | None] = mapped_column(Text)
    lora_alias: Mapped[str | None] = mapped_column(Text)

    # RAG
    rag_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rag_alias: Mapped[str | None] = mapped_column(Text)
    knowledge_base: Mapped[str | None] = mapped_column(Text)
    qdrant_alias: Mapped[str | None] = mapped_column(Text)
    qdrant_collection: Mapped[str | None] = mapped_column(Text)
    rag_manifest_id: Mapped[str | None] = mapped_column(Text)
    embedding_model: Mapped[str | None] = mapped_column(Text)
    chunking_strategy: Mapped[str | None] = mapped_column(Text)
    chunk_size: Mapped[int | None] = mapped_column(Integer)
    chunk_overlap: Mapped[int | None] = mapped_column(Integer)
    retrieval_top_k: Mapped[int | None] = mapped_column(Integer)
    score_threshold: Mapped[float | None] = mapped_column(Float)
    qdrant_snapshot_id: Mapped[str | None] = mapped_column(Text)
    dataset_dvc_hash: Mapped[str | None] = mapped_column(Text)
    reranking_strategy: Mapped[str | None] = mapped_column(Text)

    # Judge & metrics config
    judge_backend: Mapped[str | None] = mapped_column(Text)
    judge_model: Mapped[str | None] = mapped_column(Text)
    bert_score_model: Mapped[str | None] = mapped_column(Text)

    # Generation params
    temperature: Mapped[float | None] = mapped_column(Float)
    max_tokens: Mapped[int | None] = mapped_column(Integer)
    eval_verdict: Mapped[str | None] = mapped_column(Text)

    extra: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)

    samples: Mapped[list["EvalSample"]] = relationship(
        back_populates="eval_run", cascade="all, delete-orphan"
    )


class EvalSample(Base):
    """Per-sample evaluation detail, linked to an :class:`EvalRun`."""

    __tablename__ = "eval_samples"
    __table_args__ = (UniqueConstraint("eval_run_id", "sample_idx"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    eval_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False
    )
    sample_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_id: Mapped[str | None] = mapped_column(Text)

    input: Mapped[str | None] = mapped_column(Text)
    output: Mapped[str | None] = mapped_column(Text)
    reference: Mapped[str | None] = mapped_column(Text)

    detail: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    eval_run: Mapped[EvalRun] = relationship(back_populates="samples")
