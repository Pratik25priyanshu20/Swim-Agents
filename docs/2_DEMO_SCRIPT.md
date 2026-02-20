# SWIM Platform - 5-Minute Demo Script

> **Setting:** Screen shared. Terminal + browser ready. `.env` configured with `GEMINI_API_KEY`.

---

## Minute 0:00 - 0:30 | The Hook

**Say:**

> "Harmful algal blooms cost Germany over 100 million euros annually in water treatment, tourism loss, and ecological damage. Current monitoring is reactive — authorities find out about blooms after they happen. SWIM changes that. It's a multi-agent AI system that predicts blooms before they form, using satellite data, visual analysis, and machine learning."

---

## Minute 0:30 - 1:30 | Start the System

**Terminal 1 — Launch all agents:**

```bash
python -m swim.launcher
```

**Say while it boots (5 agents starting on ports 10000-10004):**

> "SWIM has 5 autonomous agents that communicate using Google's A2A protocol — the same standard Google built for agent interoperability. Each agent is independently deployable and discoverable."

**Show the agent discovery (Terminal 2):**

```bash
curl -s http://localhost:10001/.well-known/agent.json | python3 -m json.tool | head -20
```

> "Every agent publishes an AgentCard — its identity, skills, and capabilities. The orchestrator discovers agents automatically at startup."

---

## Minute 1:30 - 2:30 | Run a Prediction

**Run the CLI:**

```bash
python main.py --lake Bodensee --horizon 7 --a2a
```

**Walk through the output as it appears:**

> "Watch the pipeline stages:
> 1. HOMOGEN harmonizes raw data from CSV, Excel, and API sources into standardized parquet
> 2. CALIBRO and VISIOS run in parallel — CALIBRO processes Sentinel-2 satellite imagery for chlorophyll and turbidity, while VISIOS analyzes lake photos for visual bloom indicators
> 3. PREDIKT runs the ML ensemble — Random Forest plus Gradient Boosting — with built-in drift detection
> 4. Finally, calibrated risk fusion combines all four agents with weighted probabilities and uncertainty bounds"

**Highlight the output:**

> "The result gives us a calibrated bloom probability, a confidence interval, and a per-agent breakdown. Each agent's extraction confidence feeds into the final score. This isn't just a number — it's a quantified, traceable prediction."

---

## Minute 2:30 - 3:30 | Show the Dashboard

**Open browser:** `streamlit run streamlit_app.py`

**Click through tabs quickly:**

1. **Command Center tab:** "System status — all 5 agents healthy, data footprint stats"
2. **Knowledge Base tab:** "This is the RAG pipeline. Upload a PDF — say, an EU water quality directive — and it gets chunked, embedded with Google GenAI, and stored in the vector store. Now I can query it..."
   - Upload a sample PDF, show chunk count
   - Type a question like "What are the EU thresholds for cyanobacteria?" and show the retrieved context with relevance scores
   - "This context is automatically injected into CALIBRO and PREDIKT agent queries at orchestration time — the agents get domain knowledge without you having to copy-paste anything"
3. **Pipeline Control tab:** "Select a lake, toggle A2A protocol, run the full pipeline. The orchestrator now enriches agent queries with RAG context before sending them"
4. **Map tab:** "Three data layers on one map — PREDIKT predictions in blue, satellite data color-coded by chlorophyll, and VISIOS image locations in purple"
5. **Risk Analytics tab:** "Historical risk scores over time, pulled from the SQLite database. You can see risk level distributions and per-location breakdowns"

---

## Minute 3:30 - 4:15 | Show the API + Security + RAG

**Terminal:**

```bash
# Health check
curl http://localhost:8000/health | python3 -m json.tool

# Authenticated prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"lake": "Bodensee", "horizon_days": 7}'

# RAG query — ask the knowledge base directly
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"question": "What are HABs?", "top_k": 3}'

# RAG stats
curl http://localhost:8000/rag/stats -H "X-API-Key: your-key" | python3 -m json.tool

# Metrics endpoint
curl http://localhost:8000/metrics | head -20
```

**Say:**

> "The REST API has JWT and API key auth, per-IP rate limiting at 60 requests per minute, and input sanitization that blocks prompt injection, SQL injection, and XSS. Every request is tracked with Prometheus metrics. And notice the RAG endpoints — you can upload documents, query the knowledge base, and check stats programmatically. The orchestrator automatically pulls relevant context from the knowledge base and injects it into agent queries."

---

## Minute 4:15 - 4:45 | Show the Tests

**Terminal:**

```bash
pytest tests/ -v --tb=no -q
```

> "112 tests passing. Covering the ML pipeline, risk fusion, drift detection, authentication, rate limiting, input sanitization, and observability. Tests gracefully skip when optional dependencies aren't installed."

---

## Minute 4:45 - 5:00 | The Close

**Say:**

> "To summarize: SWIM is a production-ready platform with 5 autonomous agents, A2A protocol for interoperability, an ML ensemble with drift detection, calibrated probabilistic risk fusion, a RAG knowledge base with document upload and context injection, an 8-tab Streamlit dashboard, REST API with full security, Prometheus observability, and Docker deployment.
>
> It's 14,000+ lines of Python, 112+ tests, 12 REST endpoints, and it solves a real environmental problem. Questions?"

---

## Backup Commands (if asked)

```bash
# Show drift detection
curl http://localhost:8000/drift | python3 -m json.tool

# Show error tracking
curl http://localhost:8000/errors | python3 -m json.tool

# Upload a document to RAG knowledge base
curl -X POST http://localhost:8000/rag/upload \
  -H "X-API-Key: your-key" \
  -F "file=@docs/sample_policy.pdf" \
  -F "category=policy"

# Query the RAG knowledge base
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"question": "What are the WHO guidelines for cyanobacteria?", "top_k": 5}'

# Docker deployment
cd docker && docker compose up --build

# Generate JWT token
curl -X POST http://localhost:8000/auth/token \
  -H "X-API-Key: your-key" | python3 -m json.tool
```
