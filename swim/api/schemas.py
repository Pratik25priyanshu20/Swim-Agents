# swim/api/schemas.py

"""Pydantic schemas for the SWIM REST API with input sanitization."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from swim.shared.sanitize import (
    sanitize_image_name,
    sanitize_lake_name,
    sanitize_webhook_url,
)


class LocationInput(BaseModel):
    lake: Optional[str] = Field(None, description="German lake name (e.g., Bodensee)")
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)

    @field_validator("lake")
    @classmethod
    def validate_lake(cls, v):
        if v is not None:
            return sanitize_lake_name(v)
        return v


class PipelineRequest(BaseModel):
    location: LocationInput
    horizon_days: int = Field(7, ge=1, le=30)
    image_name: Optional[str] = None
    use_a2a: bool = Field(True, description="Use A2A protocol for inter-agent communication")
    webhook_url: Optional[str] = Field(None, description="URL for push notification on completion")

    @field_validator("image_name")
    @classmethod
    def validate_image(cls, v):
        if v is not None:
            return sanitize_image_name(v)
        return v

    @field_validator("webhook_url")
    @classmethod
    def validate_webhook(cls, v):
        if v is not None:
            return sanitize_webhook_url(v)
        return v


class RiskAssessment(BaseModel):
    level: str
    score: float
    calibrated_score: float = 0.0
    raw_score: float = 0.0
    confidence: str
    confidence_interval: Optional[Dict[str, float]] = None
    uncertainty: Optional[Dict[str, float]] = None
    evidence: List[str]
    recommendation: str
    sources_used: int = 0
    fusion_method: str = "calibrated_weighted_average"
    agent_scores: Optional[List[Dict[str, Any]]] = None


class PipelineResponse(BaseModel):
    risk_assessment: RiskAssessment
    agent_results: Dict[str, Any]
    execution_summary: Dict[str, Any]
    metadata: Dict[str, Any]


class HealthResponse(BaseModel):
    agent: str
    status: str
    checks: Dict[str, Any]


class PredictRequest(BaseModel):
    lake: str = "Bodensee"
    horizon_days: int = Field(7, ge=1, le=30)

    @field_validator("lake")
    @classmethod
    def validate_lake(cls, v):
        return sanitize_lake_name(v)


class PredictResponse(BaseModel):
    location: Dict[str, Any]
    bloom_probability: float
    risk_level: str
    confidence: float
    model_used: str


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Question to search for")
    top_k: int = Field(3, ge=1, le=20, description="Number of results to return")
