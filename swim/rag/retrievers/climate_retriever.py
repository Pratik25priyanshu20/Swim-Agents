# swim/rag/retrievers/climate_retriever.py

"""Retriever for climate and weather context relevant to HAB prediction."""

from swim.rag.document_processor import Document, knowledge_base


CLIMATE_DOCS = [
    Document(
        text=(
            "German lake temperatures have increased by 0.3-0.5°C per decade since 1980. "
            "Summer stratification periods are extending by 2-3 weeks per decade. "
            "Earlier stratification onset favors earlier cyanobacteria growth. "
            "Lakes in southern Bavaria (Bodensee, Chiemsee, Starnberger See) are less affected "
            "due to deep mixing, while shallow lakes (Müritz) are more vulnerable."
        ),
        source="DWD Climate Report 2024",
        category="climate",
    ),
    Document(
        text=(
            "Optimal HAB growth conditions in German lakes: water temperature 20-28°C, "
            "wind speed < 3 m/s (promotes stratification), solar radiation > 400 W/m², "
            "total phosphorus > 0.03 mg/L, total nitrogen > 1.0 mg/L. "
            "The critical period is typically July-September."
        ),
        source="domain_knowledge",
        category="climate",
    ),
    Document(
        text=(
            "Heavy rainfall events increase nutrient runoff into lakes, particularly phosphorus "
            "and nitrogen from agricultural land. A 2-3 week lag is typical between heavy rain "
            "and bloom initiation. Extended dry periods with high temperatures also concentrate "
            "nutrients through evaporation, especially in shallow lakes."
        ),
        source="domain_knowledge",
        category="climate",
    ),
]


def load_climate_documents():
    """Add climate documents to the global knowledge base."""
    knowledge_base.add_documents(CLIMATE_DOCS)
