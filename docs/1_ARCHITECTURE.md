# SWIM Platform Architecture

## System Overview

```
                          User / Dashboard / CLI
                                  |
                    +-------------v--------------+
                    |     REST API Gateway        |
                    |  FastAPI :8000              |
                    |  JWT + API Key Auth         |
                    |  Rate Limiting (60/min)     |
                    |  Input Sanitization         |
                    |  Prometheus /metrics        |
                    |  RAG Upload/Query/Stats     |
                    +-------------+--------------+
                                  |
                    +-------------v--------------+
                    |       ORCHESTRATOR          |
                    |       A2A :10000            |
                    |                             |
                    |  Circuit Breaker            |
                    |  Retry w/ Backoff           |
                    |  Distributed Tracing        |
                    |  RAG Context Injection      |
                    |  Calibrated Risk Fusion     |
                    +--+-------+-------+---------+
                       |       |       |
          Stage 1      |       |       |
          (Sequential) |       |       |
              +--------v--+    |       |
              |  HOMOGEN   |   |       |
              |  A2A:10001 |   |       |
              |            |   |       |
              | CSV/Excel/ |   |       |
              | API Ingest |   |       |
              | Harmonize  |   |       |
              | Validate   |   |       |
              +-----+------+   |       |
                    |          |       |
          Stage 2   +----+     |       |
          (Parallel)     |     |       |
              +----------v-+  +v-------v-------+
              |  CALIBRO    |  |    VISIOS      |
              |  A2A:10002  |  |    A2A:10003   |
              |             |  |                |
              | Sentinel-2  |  | Color Analysis |
              | Chlorophyll |  | GPS Extraction |
              | Turbidity   |  | Bloom Scoring  |
              | Quality QC  |  | Classification |
              +------+------+  +-------+--------+
                     |                 |
                     +--------+--------+
          Stage 3             |
          (Sequential)        |
                    +---------v---------+
                    |      PREDIKT      |
                    |      A2A:10004    |
                    |                   |
                    | RF + GBM Ensemble |
                    | Drift Detection   |
                    | Uncertainty QF    |
                    | 5 German Lakes    |
                    +-------------------+
                              |
                    +---------v---------+
                    |    RISK FUSION     |
                    |                    |
                    | 4-Strategy Extract |
                    | Isotonic Calibrate |
                    | Weighted Average   |
                    | Confidence Bounds  |
                    +--------------------+
                              |
                              v
              +-------------------------------+
              |        Risk Assessment        |
              |  Score: 0.0 - 1.0             |
              |  Level: low/moderate/high/crit |
              |  CI: [lower, upper]            |
              |  Per-agent breakdown           |
              +-------------------------------+
```

## Data Flow

```
Raw Data                    Harmonized              ML-Ready              Prediction
+---------+                +----------+            +--------+            +----------+
| CSV     |  HOMOGEN       | Parquet  |  CALIBRO   | Merged |  PREDIKT  | Bloom    |
| Excel   | ------------>  | Schema   | -------+-> | Feature| --------> | Prob     |
| API     |  validate      | Metadata |        |   | Matrix |  ensemble | Risk     |
+---------+  clean         +----------+        |   +--------+  drift    | Uncert.  |
             harmonize                         |                detect   +----------+
                                    VISIOS     |
                                    -------+---+
                                    images |
                                    GPS    |
                                    color  +
```

## Agent Communication: A2A Protocol

```
Agent A                          Agent B
   |                                |
   |  POST /.well-known/agent.json  |
   |  <-- AgentCard (skills, caps)  |
   |                                |
   |  POST /a2a (SendMessageRequest)|
   |  --> { query, trace_id }       |
   |                                |
   |  <-- TaskStatusUpdate(working) |
   |  <-- TaskArtifactUpdate(text)  |
   |  <-- TaskStatusUpdate(done)    |
   |                                |
```

Each agent is a standalone A2A server with its own `AgentCard`, discoverable endpoint, and skill declarations. The orchestrator discovers agents at startup and chains them via the A2A protocol.

## Security Layers

```
Request --> Rate Limiter (60/min per IP)
        --> Input Sanitization (prompt injection, SQL injection, XSS)
        --> Auth (JWT HS256 or API Key)
        --> Endpoint Handler
        --> Response with X-RateLimit-* headers
```

## Observability Stack

```
Prometheus Metrics (/metrics)         Structured Logging (JSON)
  - swim_requests_total                 - trace_id propagation
  - swim_pipeline_duration_seconds      - span_id per operation
  - swim_agent_calls_total              - JSON format output
  - swim_latest_risk_score              - per-agent timing
  - swim_active_pipelines
                                      Error Tracking
SQLite Persistence                      - per-agent health
  - pipeline_runs history               - error summaries
  - agent_memory                        - auto-recovery
  - task tracking
```

## Risk Fusion Detail

```
Weights:  PREDIKT 40%  |  CALIBRO 30%  |  VISIOS 20%  |  HOMOGEN 10%

           Agent Output
               |
     +---------v----------+
     | 4-Strategy Extract  |
     | JSON (95% conf)     |
     | Labelled (85%)      |
     | Percentage (70%)    |
     | Bare decimal (40%)  |
     +---------+-----------+
               |
     +---------v----------+
     | Isotonic Calibration|
     | (trained on ground  |
     |  truth pairs)       |
     +---------+-----------+
               |
     +---------v----------+
     | Weighted Fusion     |
     | eff_w = w * conf    |
     | score = sum(p*w)/W  |
     +---------+-----------+
               |
     +---------v----------+
     | Uncertainty         |
     | = disagreement 50%  |
     | + extraction  30%   |
     | + coverage    20%   |
     +---------------------+
```

## RAG Knowledge Base Pipeline

```
                 Upload (PDF/TXT/CSV/MD)
                         |
              +----------v-----------+
              |     File Loader       |
              |  PyMuPDF (PDF)        |
              |  pandas (CSV)         |
              |  plain read (TXT/MD)  |
              +----------+-----------+
                         |
              +----------v-----------+
              |    Text Chunking      |
              |  500-word windows     |
              |  50-word overlap      |
              +----------+-----------+
                         |
              +----------v-----------+
              |    Embedding          |
              |  Google GenAI         |
              |  (fallback: TF-IDF)  |
              +----------+-----------+
                         |
              +----------v-----------+
              |   Vector Store        |
              |  In-memory KnowledgeBase
              |  Cosine similarity    |
              |  Save/Load (.npz)     |
              +----------+-----------+
                         |
         +---------------+----------------+
         |                                |
+--------v---------+          +-----------v-----------+
|  /rag/query API  |          |  Orchestrator Inject  |
|  /rag/stats API  |          |  CALIBRO + PREDIKT    |
|  Dashboard Tab   |          |  queries enriched     |
+------------------+          +-----------------------+

Built-in Retrievers:
  - Policy: EU Bathing Water Directive, BadegewV, WHO Guidelines, OGewV
  - Climate: DWD temperature trends, HAB growth conditions, rainfall effects
  - Reports: Historical pipeline run summaries (auto-ingested)
  - Lake Info: German lake metadata from PREDIKT config
  - Uploaded: User documents via /rag/upload or dashboard
```
