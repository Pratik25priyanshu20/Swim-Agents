import argparse
import asyncio
import json
import sys

from swim.agents.main_agent.controller import MainAgentController, _resolve_location
from swim.agents.predikt.config import GERMAN_LAKES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SWIM platform pipeline")
    parser.add_argument("--lake", help="Lake name (e.g., Bodensee)")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon (days)")
    parser.add_argument("--image", help="Image filename from data/visios_images")
    parser.add_argument("--skip-homogen", action="store_true", help="Skip HOMOGEN step")
    parser.add_argument("--a2a", action="store_true", help="Use A2A protocol instead of direct calls")
    args = parser.parse_args()

    location = _resolve_location(args.lake, args.lat, args.lon)

    if args.a2a:
        from swim.agents.orchestrator.a2a_orchestrator import SWIMOrchestrator

        async def _run():
            orch = SWIMOrchestrator()
            try:
                return await orch.run_pipeline(
                    location=location,
                    horizon_days=args.horizon,
                    image_name=args.image,
                )
            finally:
                await orch.close()

        result = asyncio.run(_run())
    else:
        controller = MainAgentController()
        result = controller.run_full_pipeline(
            location=location,
            horizon_days=args.horizon,
            image_name=args.image,
            run_homogen=not args.skip_homogen,
        )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
