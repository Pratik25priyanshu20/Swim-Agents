# swim/rag/retrievers/policy_retriever.py

"""Retriever for EU/German water quality policy documents."""

from swim.rag.document_processor import Document, knowledge_base


POLICY_DOCS = [
    Document(
        text=(
            "EU Bathing Water Directive 2006/7/EC requires member states to monitor bathing waters for "
            "intestinal enterococci and E. coli. Cyanobacteria monitoring is required when visual inspection "
            "or proliferation trends suggest risk. Waters are classified annually as excellent, good, sufficient, or poor."
        ),
        source="EU Directive 2006/7/EC",
        category="policy",
    ),
    Document(
        text=(
            "German Badegewässerverordnung (BadegewV) implements the EU Bathing Water Directive. "
            "Monitoring frequency: at least monthly during bathing season (May 15 - September 15). "
            "Additional sampling required if cyanobacteria exceed 100,000 cells/mL or chlorophyll-a exceeds 50 µg/L."
        ),
        source="BadegewV",
        category="policy",
    ),
    Document(
        text=(
            "WHO Guidelines for Safe Recreational Water Environments recommend a three-tier alert framework: "
            "Guidance Level 1: 20,000 cyanobacteria cells/mL — increase monitoring. "
            "Guidance Level 2: 100,000 cells/mL — issue public advisory, restrict activities. "
            "Guidance Level 3: visible scum formation — close bathing site."
        ),
        source="WHO Guidelines",
        category="policy",
    ),
    Document(
        text=(
            "German Surface Water Ordinance (OGewV) sets environmental quality standards for surface waters. "
            "Phosphorus limits: oligotrophic < 0.01 mg/L, mesotrophic 0.01-0.03 mg/L, eutrophic > 0.03 mg/L. "
            "Chlorophyll-a thresholds: good status < 10 µg/L for lakes."
        ),
        source="OGewV",
        category="policy",
    ),
]


def load_policy_documents():
    """Add policy documents to the global knowledge base."""
    knowledge_base.add_documents(POLICY_DOCS)
