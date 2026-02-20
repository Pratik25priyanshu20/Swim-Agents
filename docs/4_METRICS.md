# SWIM Platform - Key Metrics & Numbers

> Print this or keep it handy. These are the numbers that impress.

---

## System Scale

| Metric | Value |
|--------|-------|
| Lines of Python | **14,000+** (swim/ core) |
| Python files | **150+** modules |
| Unit tests | **112+** passing (13 skip due to optional deps) |
| Test files | **12** |
| REST API endpoints | **12** (including 3 RAG endpoints) |
| Autonomous agents | **5** (HOMOGEN, CALIBRO, VISIOS, PREDIKT, Orchestrator) |
| LangGraph workflows | **5** agentic state graphs |
| Docker containers | **5** + docker-compose |
| Supported lakes | **5** German lakes (extensible) |

---

## ML Pipeline

| Metric | Value |
|--------|-------|
| Models in ensemble | **2** (Random Forest + Gradient Boosting) |
| Input features | **10** (chlorophyll-a, water temp, turbidity, DO, pH, N, P, solar radiation, wind, precipitation) |
| Training records | **52,312** samples (2008-2023) |
| Validation | **10-fold** cross-validation |
| Base accuracy | **82%** (ensemble), **91.7%** (deep learning backend) |
| Forecast horizons | **3** (3-day, 7-day, 14-day) |
| Drift detection | **KS test** (p<0.05) + **PSI** (warning >0.1, critical >0.2) |

---

## Risk Fusion

| Component | Detail |
|-----------|--------|
| Extraction strategies | **4** (JSON 95%, labelled 85%, percentage 70%, bare decimal 40%) |
| Calibration | **Isotonic regression** on ground-truth pairs |
| Agent weights | PREDIKT 40%, CALIBRO 30%, VISIOS 20%, HOMOGEN 10% |
| Uncertainty model | Agent disagreement (50%) + extraction uncertainty (30%) + coverage (20%) |
| Output | Calibrated score + confidence interval + per-agent breakdown |

---

## Security

| Feature | Implementation |
|---------|---------------|
| Authentication | JWT (HS256) + API key (X-API-Key header) |
| Rate limiting | 60 req/min per IP, sliding window |
| Input sanitization | 6 prompt injection patterns, 4 SQL injection patterns, HTML/XSS stripping |
| Path traversal | Blocked in image names |
| SSRF protection | Internal IP blocking (localhost, 10.x, 172.16-31.x, 192.168.x) |
| Dev mode | Auth auto-disabled when no credentials configured |

---

## Observability

| Metric Type | Tracked |
|-------------|---------|
| Counters | Requests, pipeline runs, agent calls, auth failures, rate limit hits |
| Histograms | Request latency, pipeline duration, per-agent latency |
| Gauges | Active pipelines, latest risk score, risk level |
| Logging | Structured JSON with trace_id + span_id propagation |
| Error tracking | Per-agent health, error summaries, auto-recovery |
| Persistence | SQLite pipeline run history |

---

## Infrastructure

| Component | Technology |
|-----------|-----------|
| Agent protocol | Google A2A (Agent-to-Agent) |
| Agent framework | LangGraph (StateGraph) |
| LLM backend | Google Gemini 2.5 Flash |
| API gateway | FastAPI |
| Dashboard | Streamlit (8 tabs, incl. Knowledge Base) |
| ML framework | scikit-learn + TensorFlow |
| Data format | Apache Parquet |
| Containerization | Docker + Docker Compose |
| Metrics | Prometheus |
| Database | SQLite |

---

## Dashboard Tabs

| # | Tab | Content |
|---|-----|---------|
| 1 | Command Center | Agent status cards, data footprint, lake overview |
| 2 | Live Map | PyDeck 3-layer map (predictions, satellite, photos) |
| 3 | Predictions | Forecast selector, bloom probability, high-risk lakes |
| 4 | Satellite Intel | Satellite data sample, cloud coverage, unique lakes |
| 5 | Visual Analysis | Image analysis metrics, classification distribution |
| 6 | Knowledge Base | RAG document upload, stats, vector search, document browser |
| 7 | Pipeline Control | Lake selector, A2A toggle, full pipeline execution |
| 8 | Risk Analytics | Historical scores, risk distribution, per-location breakdown |

---

## RAG Knowledge Base

| Component | Detail |
|-----------|--------|
| Embedding model | Google GenAI (`models/embedding-001`) + fallback TF-IDF hash |
| Vector store | In-memory `KnowledgeBase` with cosine similarity, `.npz` persistence |
| Chunking | 500-word windows with 50-word overlap |
| File formats | PDF (PyMuPDF), TXT, MD, CSV (pandas) |
| Built-in retrievers | Policy (4 docs), Climate (3 docs), Reports (auto-ingested), Lake Info (5 lakes) |
| API endpoints | `/rag/upload`, `/rag/query`, `/rag/stats` (all authenticated) |
| Context injection | Orchestrator enriches CALIBRO + PREDIKT queries with top-3 RAG results |
| Dashboard | Upload, stats donut chart, vector search, document browser |

---

## Research Experiments (Phase 1-3)

| Metric | Value |
|--------|-------|
| Research dataset | **2,551** unified records (7 lakes, 14 features) |
| In-situ features | **9** (chlorophyll-a, turbidity, DO, pH, temperature, conductivity, wind, air temp, humidity) |
| Satellite features | **5** (NDVI, surface temperature, chlorophyll index, turbidity index, cloud coverage) |
| Best monolithic AUROC | **0.814** (GB+RF ensemble) |
| Best single-modality AUROC | **0.850** (satellite-only GradientBoosting) |
| Deep learning models | LSTM (0.782), Transformer (0.806) |
| Dropout robustness at 70% | GB loses 21.5%, RF loses 17.2%, LSTM loses 7.5% |
| Agentic fusion strategies | **6** (simple avg, weighted, confidence, entropy, conflict-aware, stacking) |
| Communication protocols tested | **3** (independent, prediction sharing, confidence-gated) |
| Conflict resolution strategies | **6** (average, trust-calibro, trust-visios, max-conf, weighted-conf, entropy) |
| Dropout experiment trials | **20** per rate x 8 rates = 160 evaluations per model |
| Statistical significance | Wilcoxon signed-rank p<0.05 |
| GPU used | NVIDIA RTX A6000 |
| Publication figures generated | **22** across 3 phases |
| Research notebooks | **3** (EDA, Deep Learning, Agentic) |

---

## Quick Comparison: What Makes This Different

| Typical ML Project | SWIM Platform |
|--------------------|---------------|
| Single notebook | 5 autonomous agents |
| Hardcoded pipeline | A2A protocol (inter-agent communication standard) |
| No context awareness | RAG knowledge base with document upload + context injection |
| No auth | JWT + API keys + rate limiting + sanitization |
| print() debugging | Prometheus metrics + structured logging + distributed tracing |
| model.predict() | Calibrated risk fusion with uncertainty quantification |
| Static model | Drift detection with auto-uncertainty widening |
| localhost only | Docker Compose with 5 containers |
| No tests | 112+ unit tests |
