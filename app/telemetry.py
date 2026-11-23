"""Async request telemetry backed by SQLAlchemy."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TriageEvent(Base):
    __tablename__ = "triage_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    department: Mapped[str] = mapped_column(String(64))
    intent: Mapped[str] = mapped_column(String(256))
    confidence_score: Mapped[float] = mapped_column(Float)
    routing_status: Mapped[str] = mapped_column(String(64))
    human_override: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class TelemetryStore:
    """Owns the async database connection and ROI aggregation queries."""

    def __init__(self, database_url: str) -> None:
        normalized = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if normalized.startswith("sqlite:///") and "+aiosqlite" not in normalized:
            normalized = normalized.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        self.engine = create_async_engine(normalized, pool_pre_ping=True)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_tables(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def record(self, **values: Any) -> None:
        async with self.sessions() as session:
            session.add(TriageEvent(**values))
            await session.commit()

    async def record_override(self, tenant_id: str, corrected_intent: str) -> None:
        """Store a human correction for later ROI and retraining analysis."""
        await self.record(
            tenant_id=tenant_id,
            latency_ms=0.0,
            department="human_override",
            intent=corrected_intent,
            confidence_score=1.0,
            routing_status="human_override",
            human_override=corrected_intent,
        )

    async def summary(self, tenant_id: str) -> dict[str, Any]:
        async with self.sessions() as session:
            rows = (
                await session.execute(
                    select(TriageEvent).where(TriageEvent.tenant_id == tenant_id)
                )
            ).scalars().all()
        if not rows:
            return {"auto_triage_rate": 0.0, "p95_latency_ms": 0.0, "intent_breakdown": {}}
        latencies = sorted(row.latency_ms for row in rows)
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        intents: dict[str, int] = {}
        for row in rows:
            intents[row.intent] = intents.get(row.intent, 0) + 1
        auto = sum(row.routing_status == "auto_routed" for row in rows)
        return {
            "auto_triage_rate": auto / len(rows),
            "p95_latency_ms": p95,
            "intent_breakdown": intents,
        }
