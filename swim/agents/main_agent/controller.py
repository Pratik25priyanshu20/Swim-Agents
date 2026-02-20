# Main agent that orchestrates HOMOGEN, CALIBRO, VISIOS, and PREDIKT

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from swim.agents.homogen.core_pipeline import HOMOGENPipeline
from swim.agents.calibro.calibro_core import CalibroAgent
from swim.agents.visios.visios_agent import VisiosAgent
from swim.agents.predikt.predikt_agent import PrediktAgent
from swim.agents.predikt.config import GERMAN_LAKES


class MainAgentController:
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parents[4]
        self.homogen = HOMOGENPipeline(self.project_root)
        self.calibro = CalibroAgent()
        self.visios = VisiosAgent()
        self.predikt = PrediktAgent()

    def run_full_pipeline(
        self,
        location: Dict[str, Any],
        horizon_days: int = 7,
        image_name: Optional[str] = None,
        run_homogen: bool = True,
    ) -> Dict[str, Any]:
        result = {
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "agents": {},
        }

        if run_homogen:
            try:
                homogen_data = self.homogen.run_pipeline()
                result["agents"]["homogen"] = {
                    "status": "success",
                    "datasets": list(homogen_data.keys()) if homogen_data else [],
                }
            except Exception as exc:
                result["agents"]["homogen"] = {"status": "error", "error": str(exc)}
        else:
            result["agents"]["homogen"] = {"status": "skipped"}

        try:
            calibro = self.calibro.get_water_quality_at_location(
                lat=location["latitude"],
                lon=location["longitude"],
                date=result["timestamp"],
            )
            result["agents"]["calibro"] = {"status": "success", "data": calibro}
        except Exception as exc:
            result["agents"]["calibro"] = {"status": "error", "error": str(exc)}

        if image_name:
            visios = self.visios.analyze_image(image_name, include_context=True)
            if "error" in visios:
                result["agents"]["visios"] = {"status": "error", "error": visios["error"]}
            else:
                result["agents"]["visios"] = {"status": "success", "data": visios}
        else:
            result["agents"]["visios"] = {"status": "skipped", "reason": "no image provided"}

        try:
            predikt = self.predikt.predict_bloom_probability(location, horizon_days=horizon_days)
            result["agents"]["predikt"] = {"status": "success", "data": predikt}
        except Exception as exc:
            result["agents"]["predikt"] = {"status": "error", "error": str(exc)}

        output_dir = self.project_root / "outputs" / "main_agent"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        result["report_path"] = str(out_path)
        return result


def _resolve_location(lake_name: Optional[str], lat: Optional[float], lon: Optional[float]) -> Dict[str, Any]:
    if lake_name:
        if lake_name not in GERMAN_LAKES:
            raise ValueError(f"Unknown lake '{lake_name}'.")
        lake = GERMAN_LAKES[lake_name]
        return {"name": lake_name, "latitude": lake["lat"], "longitude": lake["lon"]}

    if lat is None or lon is None:
        raise ValueError("Provide either --lake or both --lat and --lon.")

    return {"name": "custom_location", "latitude": lat, "longitude": lon}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run main agent orchestration")
    parser.add_argument("--lake", help="Lake name (e.g., Bodensee)")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon (days)")
    parser.add_argument("--image", help="Image filename from data/visios_images")
    parser.add_argument("--skip-homogen", action="store_true", help="Skip HOMOGEN step")
    args = parser.parse_args()

    location = _resolve_location(args.lake, args.lat, args.lon)
    controller = MainAgentController()
    result = controller.run_full_pipeline(
        location=location,
        horizon_days=args.horizon,
        image_name=args.image,
        run_homogen=not args.skip_homogen,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
        
