"""Ingestion helpers for the HOMOGEN agent."""

from swim.agents.homogen.ingestion.csv_ingestor import load_csv
from swim.agents.homogen.ingestion.excel_ingestor import load_excel

__all__ = ["load_csv", "load_excel"]
