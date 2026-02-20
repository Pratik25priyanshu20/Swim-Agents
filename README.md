# SWIM Platform: Surface Water Intelligence & Monitoring

A multi-agent AI system for monitoring, analyzing, and predicting Harmful Algal Blooms (HABs) across German lakes. Agents communicate via Google's A2A (Agent-to-Agent) protocol, making them independently deployable and interoperable.

---

## Architecture

```
                    +-----------------+
                    |   Orchestrator  |  :10000
                    |  (A2A Server)   |
                    +--------+--------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v--+   +------v-----+  +-----v------+
     |  HOMOGEN   |   |  CALIBRO   |  |   VISIOS   |
     |  :10001    |   |  :10002    |  |   :10003   |
     +-----+------+   +------+-----+  +-----+------+
           |                  |              |
           +--------+---------+--------------+
                    |
              +-----v------+
              |  PREDIKT   |
              |  :10004    |
              +------------+
```

**Pipeline stages:** HOMOGEN (harmonize data) -> CALIBRO + VISIOS (parallel) -> PREDIKT (forecast) -> Risk Fusion

| Agent | Role |
|-------|------|
| **HOMOGEN** | Harmonizes raw lake data from CSV, Excel, and API sources into standardized parquet |
| **CALIBRO** | Calibrates Sentinel-2 satellite data for chlorophyll-a and turbidity indices |
| **VISIOS** | Detects HABs visually from lake images using EXIF metadata and color analysis |
| **PREDIKT** | Predicts bloom probability using ensemble ML (Random Forest + Gradient Boosting) |
| **Orchestrator** | Chains agents via A2A, fuses results into calibrated risk scores |

---

## Quick Start

### Prerequisites

- Python 3.10+
- A Google Gemini API key

### Setup

```bash
# Clone and install
git clone <repo-url> && cd ERAY_HEIDELBERG
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

### Run

**Option A: All agents via A2A (recommended)**

```bash
python -m swim.launcher
```

This starts all 5 A2A servers (HOMOGEN, CALIBRO, VISIOS, PREDIKT, Orchestrator) on ports 10000-10004.

**Option B: CLI for a single lake**

```bash
python main.py --lake Bodensee --horizon 7 --a2a
```

**Option C: REST API**

```bash
python -m swim.api.endpoints

# Then:
curl -X POST http://localhost:8000/pipeline \
  -H "Content-Type: application/json" \
  -d '{"location": {"lake": "Bodensee"}, "horizon_days": 7}'
```

**Option D: Streamlit dashboard**

```bash
streamlit run streamlit_app.py
```

---

## REST API

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | No | Service info |
| `/health` | GET | No | Health check |
| `/metrics` | GET | No | Prometheus metrics |
| `/lakes` | GET | No | List available lakes |
| `/drift` | GET | No | Data drift detection status |
| `/errors` | GET | No | Error tracking summary |
| `/predict` | POST | Yes | Single-lake prediction |
| `/pipeline` | POST | Yes | Full SWIM pipeline |
| `/auth/token` | POST | Yes | Generate JWT token |
| `/rag/upload` | POST | Yes | Upload document to knowledge base (PDF/TXT/CSV) |
| `/rag/query` | POST | Yes | Query RAG knowledge base |
| `/rag/stats` | GET | Yes | Knowledge base statistics |

### Authentication

Set `SWIM_API_KEYS` (comma-separated) and/or `SWIM_JWT_SECRET` in `.env`. When neither is set, auth is disabled (dev mode).

```bash
# API key
curl -H "X-API-Key: your-key" http://localhost:8000/predict ...

# JWT
curl -H "Authorization: Bearer <token>" http://localhost:8000/predict ...
```

---

## Docker

```bash
cd docker
docker compose up --build
```

Services: orchestrator (:10000), homogen (:10001), calibro (:10002), visios (:10003), predikt (:10004).

---

## Project Structure

```
swim/
  agents/
    homogen/          # Data harmonization agent
    calibro/          # Satellite calibration agent
    visios/           # Visual image analysis agent
    predikt/          # ML prediction agent
    orchestrator/     # A2A orchestrator + risk fusion
    main_agent/       # Direct-call controller (non-A2A)
  api/                # FastAPI REST gateway
  data_processing/    # ETL, cleaning, drift detection
  observability/      # Structured logging, metrics, error tracking
  rag/                # RAG knowledge base (embeddings, retrievers, file ingestion)
  shared/             # Auth, config, paths, alerting, rate limiting
  models/             # ML model definitions
data/
  raw/                # Raw source data
  harmonized/         # Standardized parquet output
  processed/          # ML-ready features
  knowledge/          # Uploaded RAG documents (PDF, TXT, CSV)
models/               # Trained model artifacts (.pkl)
tests/                # Unit tests
docker/               # Dockerfiles + compose
config.yaml           # Central configuration
```

---

## Key Features

- **A2A Protocol**: Each agent is a standalone A2A server with its own AgentCard, discoverable at `/.well-known/agent.json`
- **Calibrated Risk Fusion**: Multi-strategy probability extraction with isotonic regression calibration and uncertainty quantification
- **Data Drift Detection**: KS test + PSI comparing incoming distributions against training baselines
- **Prometheus Metrics**: `/metrics` endpoint with request counts, latencies, active pipelines, risk gauges
- **Security**: JWT + API key auth, per-IP rate limiting, input sanitization (prompt injection, SQL injection, XSS)
- **Alerting**: Email + Slack notifications on critical risk levels
- **SQLite Persistence**: Pipeline run history, agent memory, task tracking
- **RAG Knowledge Base**: Document upload (PDF/TXT/CSV), vector embeddings (Google GenAI + fallback TF-IDF), cosine-similarity retrieval, and automatic context injection into agent queries at orchestration time. Built-in retrievers for EU/German water quality policy, climate data, and historical reports

---

## Configuration

Central config in `config.yaml`. Environment variables override config values:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Google Gemini LLM key (required) |
| `SWIM_API_KEYS` | API key auth for REST gateway |
| `SWIM_JWT_SECRET` | JWT signing secret |
| `SWIM_LLM_MODEL` | Override LLM model |
| `SWIM_LOG_LEVEL` | Log level (INFO/DEBUG) |
| `PREDIKT_MODEL_TYPE` | ensemble / lstm / sarima / rule_based |

---

## Testing

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## Available Lakes

| Lake | Region | Trophic Status |
|------|--------|----------------|
| Bodensee | Baden-Wurttemberg/Bavaria | Oligotrophic |
| Chiemsee | Bavaria | Mesotrophic |
| Starnberger See | Bavaria | Oligotrophic |
| Ammersee | Bavaria | Mesotrophic |
| Muritz | Mecklenburg-Vorpommern | Eutrophic |

---

## Research

SWIM serves as the experimental testbed for PhD research on **"Agentic AI for Multi-Modal Earth Observation"** — investigating whether specialized, communicating agents outperform monolithic models for environmental monitoring.

### Research Questions

| RQ | Question | Status |
|----|----------|--------|
| **RQ1** | Does agent specialization improve robustness under sensor failure? | Preliminary results |
| **RQ2** | Does inter-agent communication improve predictions? | Preliminary results |
| **RQ3** | How should the orchestrator resolve agent conflicts? | Preliminary results |

### Preliminary Results (Phase 1-3)

**Dataset:** 2,551 records across 7 German lakes, 14 features (9 in-situ + 5 satellite)

| Experiment | Key Finding |
|-----------|-------------|
| Monolithic baselines | Tabular ensemble AUROC 0.814, Transformer 0.806, LSTM 0.782 |
| Modality ablation | Satellite-only (5 features) outperforms all-features (0.850 vs 0.794) |
| Feature dropout | Monolithic GB loses 21.5% AUROC at 70% dropout; LSTM loses only 7.5% |
| Agentic vs monolithic | Specialist agents + orchestrator degrade less under sensor failure |
| Communication protocols | Prediction-sharing agents outperform independent agents under dropout |
| Conflict resolution | Confidence-weighted fusion beats naive averaging on disagreement samples |

**Research notebooks:** `notebooks/research_package/` (Phase 1: EDA, Phase 2: Deep Learning, Phase 3: Agentic)

See [docs/PROGRESS.md](docs/PROGRESS.md) for detailed research progress and remaining work.

---

## License

MIT
