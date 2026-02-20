"""
MVP pipeline: HOMOGEN -> PREDIKT.

Runs data harmonization (best-effort), prepares ML-ready data,
then executes a HAB forecast with the trained PREDIKT model.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from swim.agents.homogen.core_pipeline import HOMOGENPipeline
from swim.agents.predikt.data_loader_real import RealDataLoader
from swim.agents.predikt.predikt_agent import PrediktAgent


def run_homogen(project_root: Path, sources: Optional[list] = None) -> None:
    """Run HOMOGEN harmonization; failures are non-fatal for MVP."""
    try:
        pipeline = HOMOGENPipeline(project_root)
        pipeline.run_pipeline(source_names=sources)
        print("HOMOGEN: completed")
    except Exception as exc:
        print(f"HOMOGEN: skipped ({exc})")


def prepare_ml_data(project_root: Path) -> None:
    """Create data/processed/ml_ready_data.csv from raw inputs."""
    loader = RealDataLoader(base_dir=str(project_root))
    loader.load_and_prepare_data()
    print("PREDIKT data: ml_ready_data.csv updated")


def run_predikt(lake_name: Optional[str], horizon_days: int) -> dict:
    """Run forecast with the trained PREDIKT model."""
    agent = PrediktAgent()
    if lake_name:
        if lake_name not in agent.GERMAN_LAKES:
            raise ValueError(
                f"Lake '{lake_name}' not found. Available: {', '.join(agent.GERMAN_LAKES.keys())}"
            )
        lake = agent.GERMAN_LAKES[lake_name]
        location = {"name": lake_name, "latitude": lake["lat"], "longitude": lake["lon"]}
        return agent.predict_bloom_probability(location, horizon_days=horizon_days)
    return agent.predict_german_lakes(horizon_days=horizon_days)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MVP HOMOGEN -> PREDIKT pipeline")
    parser.add_argument("--lake", help="Predict a single lake by name")
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon in days")
    parser.add_argument("--skip-homogen", action="store_true", help="Skip HOMOGEN step")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    if not args.skip_homogen:
        run_homogen(project_root)

    prepare_ml_data(project_root)
    result = run_predikt(args.lake, args.horizon)

    output_dir = project_root / "outputs" / "predikt"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"mvp_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"MVP report saved to {out_path}")


if __name__ == "__main__":
    main()
