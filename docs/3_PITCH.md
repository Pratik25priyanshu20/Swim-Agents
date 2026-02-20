# SWIM Platform - 2-Minute Pitch

> Use this for interviews, presentations, or when someone asks "what did you build?"

---

## The Pitch (read time: ~2 minutes)

### The Problem (20 seconds)

Harmful algal blooms are increasing across German lakes due to climate change and nutrient runoff. They produce toxins that shut down drinking water plants, close beaches, and kill aquatic ecosystems. Current monitoring is reactive — by the time a bloom is visible, it's already too late. Germany needs a predictive early warning system.

### What I Built (40 seconds)

SWIM — Surface Water Intelligence & Monitoring — is a multi-agent AI platform that predicts harmful algal blooms before they form. It combines four specialized AI agents:

- **HOMOGEN** harmonizes raw data from multiple sources — CSV, Excel, APIs — into standardized parquet
- **CALIBRO** processes Sentinel-2 satellite imagery for chlorophyll-a and turbidity indices
- **VISIOS** analyzes user-submitted lake photos using computer vision for visual bloom detection
- **PREDIKT** runs a Random Forest + Gradient Boosting ensemble to forecast bloom probability with uncertainty quantification

A **RAG knowledge base** lets users upload domain documents (PDF, TXT, CSV) that get chunked, embedded with Google GenAI, and stored in a vector store. The orchestrator automatically retrieves relevant context and injects it into agent queries — so agents get domain knowledge without manual prompt engineering.

An **Orchestrator** chains these agents together using Google's A2A (Agent-to-Agent) protocol — the same interoperability standard Google uses internally. Each agent is independently deployable and discoverable.

### What Makes It Production-Ready (30 seconds)

This isn't a notebook. It's a deployable system with:

- **Calibrated risk fusion** — 4-strategy probability extraction with isotonic regression calibration and confidence intervals
- **Data drift detection** — Kolmogorov-Smirnov tests and Population Stability Index detect when incoming data diverges from training distribution
- **Full security** — JWT + API key authentication, per-IP rate limiting, input sanitization blocking prompt injection, SQL injection, and XSS
- **Observability** — Prometheus metrics, structured JSON logging with distributed trace IDs, error tracking per agent
- **Deployment** — Docker Compose for all 5 agents, REST API gateway, Streamlit dashboard

### The Numbers (15 seconds)

- 14,000+ lines of Python across 5 autonomous agents
- 112+ unit tests, 12 REST endpoints, 5 Docker containers
- 5 LangGraph agentic workflows with circuit breakers and retry logic
- RAG pipeline with document upload, vector embeddings, and context injection
- 8-tab Streamlit dashboard with knowledge base management
- Covers 5 German lakes with 3 forecast horizons (3, 7, 14 days)

### Why It Matters (15 seconds)

This project demonstrates I can design and build a complete AI system — not just a model, but the infrastructure around it: agent orchestration, ML pipelines with drift detection, security, observability, and deployment. It's the kind of system that runs in production, not just in a demo.

---

## One-Liner Versions

**For a recruiter:**
> "I built a multi-agent AI platform that predicts harmful algal blooms in German lakes using satellite data, computer vision, ML, and RAG — with A2A protocol, drift detection, and full production security."

**For a technical lead:**
> "It's a 5-agent LangGraph + A2A system with RAG-augmented orchestration, calibrated probabilistic risk fusion, KS/PSI drift detection, Prometheus observability, and JWT/rate-limited FastAPI — deployed via Docker Compose."

**For a non-technical audience:**
> "I built an AI system that warns about toxic algae in lakes before they become dangerous — it combines satellite images, photos, machine learning, and a knowledge base of water quality policies to give early warnings."
