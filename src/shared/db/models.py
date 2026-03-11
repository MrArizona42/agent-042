"""SQLAlchemy ORM models for the agent042 database.

Tables
------
- users            — authenticated users (Google OIDC)
- chat_sessions    — per-user conversation sessions
- chat_messages    — individual messages within a session
- eval_runs        — evaluation run metadata and config
- eval_metrics     — aggregate metrics per eval run
- eval_examples    — per-example results for drill-down
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
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

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
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

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped[ChatSession] = relationship(back_populates="messages")


# ──────────────────────────────────────────────
# Evaluation models
# ──────────────────────────────────────────────


class EvalRun(Base):
    """One row per evaluation execution."""

    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text, nullable=False, default="running")
    tier: Mapped[str] = mapped_column(Text, nullable=False)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # Denormalized for fast filtering:
    base_model: Mapped[str] = mapped_column(Text, nullable=False)
    adapter_name: Mapped[str | None] = mapped_column(Text)
    adapter_version: Mapped[int | None] = mapped_column(Integer)
    dataset_name: Mapped[str] = mapped_column(Text, nullable=False)
    dataset_split: Mapped[str] = mapped_column(Text, nullable=False)
    knowledge_base: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)

    metrics: Mapped[list[EvalMetric]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    examples: Mapped[list[EvalExample]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_eval_runs_task", "task"),
        Index("idx_eval_runs_adapter", "adapter_name", "adapter_version"),
        Index("idx_eval_runs_created", created_at.desc()),
        Index("idx_eval_runs_config", "config", postgresql_using="gin"),
    )


class EvalMetric(Base):
    """Aggregate metrics: one row per metric per run."""

    __tablename__ = "eval_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("eval_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    metric_name: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    run: Mapped[EvalRun] = relationship(back_populates="metrics")

    __table_args__ = (
        UniqueConstraint("run_id", "metric_name", name="uq_eval_metric_run_name"),
        Index("idx_eval_metrics_run", "run_id"),
    )


class EvalExample(Base):
    """Per-example results: for drill-down and debugging."""

    __tablename__ = "eval_examples"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("eval_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    example_index: Mapped[int] = mapped_column(Integer, nullable=False)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    reference_text: Mapped[str | None] = mapped_column(Text)
    generated_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Per-example scores (nullable — not all metrics apply to all tasks)
    relevance: Mapped[int | None] = mapped_column(SmallInteger)
    correctness: Mapped[int | None] = mapped_column(SmallInteger)
    faithfulness: Mapped[int | None] = mapped_column(SmallInteger)
    coverage: Mapped[int | None] = mapped_column(SmallInteger)
    rouge_l: Mapped[float | None] = mapped_column(Float)
    bert_score: Mapped[float | None] = mapped_column(Float)
    # Code-specific
    executable: Mapped[bool | None] = mapped_column(Boolean)
    tests_passed: Mapped[bool | None] = mapped_column(Boolean)
    execution_error: Mapped[str | None] = mapped_column(Text)
    # RAG-specific
    retrieved_docs: Mapped[dict | None] = mapped_column(JSONB)
    groundedness: Mapped[float | None] = mapped_column(Float)

    run: Mapped[EvalRun] = relationship(back_populates="examples")

    __table_args__ = (Index("idx_eval_examples_run", "run_id"),)
