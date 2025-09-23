#!/bin/bash

# Assume we're already in ERAY_HEIDELBERG

# Root files
touch .env .gitignore README.md requirements.txt pyproject.toml

# Config
mkdir -p config/settings
touch config/schema_ontology.yaml config/logging_config.py
touch config/settings/{base.py,development.py,production.py}

# Data
mkdir -p data/raw/{in_situ,satellite,tertiary/{gov,social,ngo,media,legal,citizen_science}}
mkdir -p data/{intermediate,processed,vectorstore}

# Database
mkdir -p db/{schema,seeds,models,ingest}
touch db/schema/create_tables.sql
touch db/seeds/seed_data.sql
touch db/models/orm_models.py
touch db/ingest/{insert_data.py,fetch_data.py}

# Notebooks
mkdir -p notebooks/{01_exploration,02_modeling,03_validation}

# Core app
mkdir -p swim/{agents/{homogen,calibro,visios},workflows,data_processing/{etl,cleaning},models,rag/retrievers,cli,utils,api,observability}
touch swim/__init__.py
touch swim/agents/homogen/{prompts.py,tools.py,memory.py,harmonizer.py}
touch swim/agents/calibro/{satellite_fetch.py,calibrate_model.py,validate.py}
touch swim/agents/visios/{image_classifier.py,visual_interface.py}
touch swim/workflows/{homogen_workflow.py,calibro_workflow.py,visios_workflow.py}
touch swim/data_processing/etl/{parse_pdf.py,parse_html.py,parse_api.py,citizen_ingestor.py}
touch swim/data_processing/cleaning/harmonize_schema.py
touch swim/data_processing/{geospatial.py,temporal.py}
touch swim/models/{calibration.py,anomaly_detection.py,visual_detection.py}
touch swim/rag/{embedding.py,document_processor.py,query_router.py}
touch swim/rag/retrievers/{policy_retriever.py,climate_retriever.py,reports_retriever.py}
touch swim/cli/app.py
touch swim/utils/{geo_helpers.py,file_manager.py,visualization.py}
touch swim/api/{endpoints.py,schemas.py}
touch swim/observability/{logger.py,dag_visualizer.py,error_tracking.py}

# Scripts
mkdir -p scripts
touch scripts/{setup_directories.py,download_satellite.py,index_documents.py}

# Tests
mkdir -p tests/{agents,models,data_processing,api}
touch tests/conftest.py

# Docker & Deployment
mkdir -p docker
touch docker/{Dockerfile.homogen,Dockerfile.calibro,Dockerfile.visios,docker-compose.yml}
mkdir -p deployment/{k8s,monitoring/grafana_dashboards}
touch deployment/k8s/{homogen.yaml,calibro.yaml,visios.yaml}
touch deployment/monitoring/prometheus.yml

echo "✅ Subdirectories and files created inside ERAY_HEIDELBERG"
