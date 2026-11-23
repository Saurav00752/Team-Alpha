import json

from app.api import app
from app.api import _normalize_webhook, _run_triage, write_feedback_record
from app.schemas import FeedbackRequest, TriagePrediction, TriageRequest
from app.security import TenantConfig


class MockEngine:
    confidence_threshold = 0.85
    adapter_version = "test"

    def __init__(self, confidence: float):
        self.confidence = confidence

    def predict(self, text, adapter_path=None):
        return (
            TriagePrediction(
                department="billing",
                priority="high",
                intent="refund_request",
                requires_refund=True,
            ),
            self.confidence,
        )


def client(monkeypatch, confidence=0.5):
    monkeypatch.setenv(
        "API_KEYS_JSON",
        json.dumps({"test-key": {"tenant_id": "tenant-a", "adapter_path": ""}}),
    )
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test-triage.db")
    app.state.engine = MockEngine(confidence)
    return TenantConfig(tenant_id="tenant-a", adapter_path="")


def test_low_confidence_requires_human_review(monkeypatch):
    tenant = client(monkeypatch, 0.5)
    response = _run_triage(
        TriageRequest(ticket_text="I need a refund"), tenant
    )
    assert response.routing_status.value == "needs_human_review"


def test_webhook_normalization_and_feedback_queue(monkeypatch, tmp_path):
    queue = tmp_path / "feedback.jsonl"
    monkeypatch.setenv("FEEDBACK_QUEUE_PATH", str(queue))
    tenant = client(monkeypatch, 0.95)
    request = _normalize_webhook(
        {"ticket": {"id": 42, "description": "Please refund me"}}, "zendesk"
    )
    assert request.ticket_id == "42"
    feedback = FeedbackRequest(
        ticket_text="Email me at jane@example.com",
        prediction=TriagePrediction(
            department="billing", priority="high", intent="refund", requires_refund=True
        ),
        corrected_prediction=TriagePrediction(
            department="billing", priority="high", intent="refund", requires_refund=True
        ),
    )
    destination = write_feedback_record(feedback, tenant)
    assert destination.exists()
    record = json.loads(queue.read_text(encoding="utf-8"))
    assert "jane@example.com" not in record["messages"][1]["content"]
