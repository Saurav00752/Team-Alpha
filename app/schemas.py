"""Validated request and response models for the triage service."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Department(str, Enum):
    shipping = "shipping"
    billing = "billing"
    security = "security"
    account_support = "account_support"
    out_of_scope = "out_of_scope"


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class RoutingStatus(str, Enum):
    auto_routed = "auto_routed"
    needs_human_review = "needs_human_review"


class TriageRequest(BaseModel):
    """A support ticket submitted by an authenticated tenant."""

    model_config = ConfigDict(extra="forbid")

    ticket_text: str = Field(min_length=1, max_length=20_000)
    ticket_id: str | None = Field(default=None, max_length=256)
    source: str = Field(default="api", min_length=1, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ticket_text")
    @classmethod
    def validate_ticket_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ticket_text cannot be blank")
        return value


class TriagePrediction(BaseModel):
    """The constrained business labels emitted by the model."""

    model_config = ConfigDict(extra="forbid")

    department: Department
    priority: Priority
    intent: str = Field(min_length=1, max_length=256)
    secondary_intent: str | None = Field(default=None, max_length=256)
    requires_refund: bool


class TriageResponse(BaseModel):
    """A triage result and operational routing metadata."""

    model_config = ConfigDict(extra="forbid")

    prediction: TriagePrediction
    confidence_score: float = Field(ge=0.0, le=1.0)
    routing_status: RoutingStatus
    latency_ms: float = Field(ge=0.0)
    adapter_version: str


class FeedbackRequest(BaseModel):
    """A corrected label set supplied by a human support agent."""

    model_config = ConfigDict(extra="forbid")

    ticket_text: str = Field(min_length=1, max_length=20_000)
    prediction: TriagePrediction
    corrected_prediction: TriagePrediction
    ticket_id: str | None = None

