"""FastAPI application for authenticated, multi-tenant support triage."""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TYPE_CHECKING

from fastapi import Depends, FastAPI, HTTPException

from app.schemas import (
    FeedbackRequest,
    RoutingStatus,
    TriageRequest,
    TriageResponse,
)
from app.security import TenantConfig, authenticate_api_key, redact_pii, sanitize_ticket
from app.telemetry import TelemetryStore

if TYPE_CHECKING:
    from app.inference import TriageInferenceEngine


def _database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/triage.db")


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.telemetry = TelemetryStore(_database_url())
    await application.state.telemetry.create_tables()
    application.state.engine = None
    yield
    await application.state.telemetry.engine.dispose()


app = FastAPI(title="Customer Support Triage API", version="1.0.0", lifespan=lifespan)


def get_engine() -> "TriageInferenceEngine":
    engine = getattr(app.state, "engine", None)
    if engine is None:
        from app.inference import TriageInferenceEngine

        engine = TriageInferenceEngine()
        app.state.engine = engine
    return engine


def _run_triage(request: TriageRequest, tenant: TenantConfig) -> TriageResponse:
    engine = get_engine()
    clean_text, _ = redact_pii(sanitize_ticket(request.ticket_text))
    started = time.perf_counter()
    prediction, confidence = engine.predict(clean_text, tenant.adapter_path or None)
    latency = (time.perf_counter() - started) * 1000
    threshold = getattr(engine, "confidence_threshold", 0.85)
    return TriageResponse(
        prediction=prediction,
        confidence_score=confidence,
        routing_status=(
            RoutingStatus.auto_routed
            if confidence >= threshold
            else RoutingStatus.needs_human_review
        ),
        latency_ms=latency,
        adapter_version=getattr(engine, "adapter_version", tenant.adapter_version),
    )


async def _record(request: TriageRequest, tenant: TenantConfig, response: TriageResponse) -> None:
    telemetry = getattr(app.state, "telemetry", None)
    if telemetry:
        await telemetry.record(
            tenant_id=tenant.tenant_id,
            latency_ms=response.latency_ms,
            department=response.prediction.department.value,
            intent=response.prediction.intent,
            confidence_score=response.confidence_score,
            routing_status=response.routing_status.value,
        )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    try:
        get_engine()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model is not ready: {exc}") from exc
    return {"status": "ready"}


@app.post("/v1/triage", response_model=TriageResponse)
async def triage(
    request: TriageRequest, tenant: TenantConfig = Depends(authenticate_api_key)
) -> TriageResponse:
    response = _run_triage(request, tenant)
    await _record(request, tenant, response)
    return response


def _normalize_webhook(payload: dict[str, Any], source: str) -> TriageRequest:
    if source == "zendesk":
        ticket = payload.get("ticket", payload)
        text = ticket.get("description") or ticket.get("comment", {}).get("body", "")
        return TriageRequest(
            ticket_text=text, ticket_id=str(ticket.get("id", "")) or None, source=source, metadata=payload
        )
    ticket = payload.get("ticket", payload)
    text = ticket.get("description") or ticket.get("requester", {}).get("message", "")
    return TriageRequest(
        ticket_text=text, ticket_id=str(ticket.get("id", "")) or None, source=source, metadata=payload
    )


def _webhook_actions(response: TriageResponse) -> dict[str, Any]:
    return {
        "tags": [response.prediction.department.value, response.prediction.intent],
        "priority": response.prediction.priority.value,
        "assignee": response.prediction.department.value,
        "triage": response.model_dump(mode="json"),
    }


@app.post("/v1/webhooks/{provider}")
async def webhook(
    provider: str, payload: dict[str, Any], tenant: TenantConfig = Depends(authenticate_api_key)
) -> dict[str, Any]:
    if provider not in {"zendesk", "freshdesk"}:
        raise HTTPException(status_code=404, detail="Unsupported webhook provider")
    request = _normalize_webhook(payload, provider)
    response = _run_triage(request, tenant)
    await _record(request, tenant, response)
    return _webhook_actions(response)


def write_feedback_record(request: FeedbackRequest, tenant: TenantConfig) -> Path:
    """Persist one sanitized ChatML record for scheduled active learning."""
    clean_text, _ = redact_pii(request.ticket_text)
    destination = Path(os.getenv("FEEDBACK_QUEUE_PATH", "data/active_learning/feedback_queue.jsonl"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "messages": [
            {"role": "system", "content": "Classify support tickets into the triage schema."},
            {"role": "user", "content": clean_text},
            {"role": "assistant", "content": request.corrected_prediction.model_dump_json()},
        ],
        "tenant_id": tenant.tenant_id,
        "ticket_id": request.ticket_id,
    }
    with destination.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")
    return destination


@app.post("/v1/feedback")
async def feedback(
    request: FeedbackRequest, tenant: TenantConfig = Depends(authenticate_api_key)
) -> dict[str, str]:
    write_feedback_record(request, tenant)
    telemetry = getattr(app.state, "telemetry", None)
    if telemetry:
        await telemetry.record_override(tenant.tenant_id, request.corrected_prediction.intent)
    return {"status": "queued"}


@app.get("/v1/metrics/summary")
async def metrics_summary(
    tenant: TenantConfig = Depends(authenticate_api_key),
) -> dict[str, Any]:
    telemetry = getattr(app.state, "telemetry", None)
    if telemetry is None:
        return {"auto_triage_rate": 0.0, "p95_latency_ms": 0.0, "intent_breakdown": {}}
    return await telemetry.summary(tenant.tenant_id)
