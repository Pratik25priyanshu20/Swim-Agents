# swim/api/endpoints.py

"""FastAPI REST API gateway for the SWIM platform."""

import asyncio
import logging
import time
from typing import Any, Dict

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from swim.agents.predikt.config import GERMAN_LAKES
from swim.api.schemas import (
    HealthResponse,
    PipelineRequest,
    PipelineResponse,
    PredictRequest,
    PredictResponse,
    RAGQueryRequest,
)
from swim.observability.error_tracking import error_tracker
from swim.observability.metrics import (
    get_content_type,
    get_metrics_text,
    record_request,
)
from swim.shared.auth import require_auth
from swim.shared.rate_limit import RateLimiter

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SWIM Platform API",
    description="REST API for the Surface Water Intelligence & Monitoring platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimiter, max_requests=60, window_seconds=60)


# ---------------------------------------------------------------------------
# Metrics middleware — records request count + latency
# ---------------------------------------------------------------------------

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - start
    record_request(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
        duration=duration,
    )
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_location(req) -> Dict[str, Any]:
    if req.location.lake:
        if req.location.lake not in GERMAN_LAKES:
            raise HTTPException(404, f"Unknown lake: {req.location.lake}")
        meta = GERMAN_LAKES[req.location.lake]
        return {"name": req.location.lake, "latitude": meta["lat"], "longitude": meta["lon"]}
    if req.location.latitude is not None and req.location.longitude is not None:
        return {"name": "custom_location", "latitude": req.location.latitude, "longitude": req.location.longitude}
    raise HTTPException(400, "Provide either lake name or lat/lon coordinates")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"service": "SWIM Platform", "version": "1.0.0", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        agent="API Gateway",
        status="healthy",
        checks={"error_tracker": error_tracker.get_summary()},
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=get_metrics_text(), media_type=get_content_type())


@app.get("/lakes")
async def list_lakes():
    return {
        name: {"lat": meta["lat"], "lon": meta["lon"], "trophic_status": meta["trophic_status"]}
        for name, meta in GERMAN_LAKES.items()
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, _auth: dict = Depends(require_auth)):
    """Run PREDIKT agent for a single lake."""
    if req.lake not in GERMAN_LAKES:
        raise HTTPException(404, f"Unknown lake: {req.lake}")

    from swim.agents.predikt.predikt_agent import PrediktAgent
    agent = PrediktAgent()
    meta = GERMAN_LAKES[req.lake]
    location = {"name": req.lake, "latitude": meta["lat"], "longitude": meta["lon"]}

    result = await asyncio.to_thread(
        agent.predict_bloom_probability, location, req.horizon_days
    )
    return PredictResponse(
        location=location,
        bloom_probability=result["bloom_probability"],
        risk_level=result["risk_level"],
        confidence=result["confidence"],
        model_used=result["model_used"],
    )


@app.post("/pipeline", response_model=PipelineResponse)
async def run_pipeline(req: PipelineRequest, _auth: dict = Depends(require_auth)):
    """Run the full SWIM pipeline."""
    location = _resolve_location(req)

    if req.use_a2a:
        from swim.agents.orchestrator.a2a_orchestrator import SWIMOrchestrator
        orch = SWIMOrchestrator()
        if req.webhook_url:
            orch.register_webhook(req.webhook_url)
        try:
            result = await orch.run_pipeline(
                location=location,
                horizon_days=req.horizon_days,
                image_name=req.image_name,
            )
        finally:
            await orch.close()
    else:
        from swim.agents.main_agent.controller import MainAgentController
        controller = MainAgentController()
        result = await asyncio.to_thread(
            controller.run_full_pipeline,
            location=location,
            horizon_days=req.horizon_days,
            image_name=req.image_name,
        )

    return result


@app.get("/errors")
async def error_summary():
    """Return error tracking summary."""
    return error_tracker.get_summary()


@app.get("/drift")
async def drift_status():
    """Return current data drift detection status."""
    from swim.data_processing.drift_detector import drift_detector
    if not drift_detector.has_reference:
        return {"status": "no_reference", "message": "No drift reference available. Run retraining first."}
    # Check drift on latest processed data if available
    try:
        from swim.agents.predikt.retraining import load_training_data, FEATURE_NAMES
        X, _ = load_training_data()
        report = drift_detector.check(FEATURE_NAMES[:X.shape[1]], X[-50:])  # check last 50 rows
        return report
    except FileNotFoundError:
        return {"status": "no_data", "message": "No processed data to check against."}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/auth/token")
async def create_token(_auth: dict = Depends(require_auth)):
    """Generate a new JWT (requires existing valid credentials)."""
    from swim.shared.auth import generate_token
    try:
        token = generate_token()
        return {"token": token, "type": "bearer", "expires_in": 3600}
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# ---------------------------------------------------------------------------
# RAG Knowledge Base endpoints
# ---------------------------------------------------------------------------

@app.post("/rag/upload")
async def rag_upload(
    file: UploadFile = File(...),
    category: str = "uploaded",
    _auth: dict = Depends(require_auth),
):
    """Upload a document file and ingest it into the knowledge base."""
    from swim.rag.file_loader import SUPPORTED_EXTENSIONS
    from swim.rag.ingest import ingest_uploaded_bytes

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    content = await file.read()
    chunks_added = ingest_uploaded_bytes(file.filename, content, category=category)
    return {"filename": file.filename, "chunks_added": chunks_added, "category": category}


@app.post("/rag/query")
async def rag_query(req: RAGQueryRequest, _auth: dict = Depends(require_auth)):
    """Query the RAG knowledge base."""
    from swim.rag.ingest import query_kb

    results = query_kb(req.question, top_k=req.top_k)
    return {"question": req.question, "results": results, "count": len(results)}


@app.get("/rag/stats")
async def rag_stats(_auth: dict = Depends(require_auth)):
    """Return knowledge base statistics."""
    from swim.rag.ingest import get_kb_stats

    return get_kb_stats()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
