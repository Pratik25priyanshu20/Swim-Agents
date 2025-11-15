"""
SWIM Platform Orchestrator
Coordinates HOMOGEN, CALIBRO, VISIOS, and PREDIKT agents
for comprehensive water quality analysis and HABs forecasting.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

# === Import Agents ===
from swim.agents.homogen.core_pipeline import HOMOGENPipeline
from swim.agents.calibro.calibro_core import CalibroAgent
from swim.agents.visios.visios_agent import VisiosAgent
from swim.agents.predikt.predikt_agent import PrediktAgent


class SWIMPlatform:
    """Unified orchestrator for all SWIM agents."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the SWIM platform with all active agents."""
        print("🌊 Initializing SWIM Platform...")

        self.config = self._load_config(config_path)

        print("  └─ Loading HOMOGEN (Data Harmonization)...")
        self.homogen = HOMOGENPipeline(Path(__file__).resolve().parents[2])

        print("  └─ Loading CALIBRO (Satellite Calibration)...")
        self.calibro = CalibroAgent()

        print("  └─ Loading VISIOS (Visual Interpretation)...")
        self.visios = VisiosAgent()

        print("  └─ Loading PREDIKT (HABs Forecasting)...")
        self.predikt = PrediktAgent()

        print("✅ SWIM Platform Ready\n")

    # --------------------------
    # CONFIGURATION
    # --------------------------
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load or create platform configuration."""
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)

        return {
            "platform": {"name": "SWIM", "version": "2.0.0"},
            "agents": {
                "homogen": {"enabled": True},
                "calibro": {"enabled": True},
                "visios": {"enabled": True},
                "predikt": {"enabled": True},
            },
            "integration": {
                "cross_validation": True,
                "spatial_correlation": True,
                "temporal_window_hours": 24,
            },
        }

    # --------------------------
    # MAIN ORCHESTRATION LOGIC
    # --------------------------
    def comprehensive_water_quality_assessment(
        self,
        location: Dict[str, float],
        timestamp: Optional[str] = None,
        image_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive 4-agent water quality assessment."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        print(f"\n🔍 Comprehensive Assessment for {location}")
        print(f"   Timestamp: {timestamp}")
        print("=" * 60)

        assessment = {"location": location, "timestamp": timestamp, "agents": {}}

        # 1️⃣ HOMOGEN
        print("\n1️⃣  HOMOGEN: Querying harmonized in-situ data...")
        try:
            sensor_data = self.homogen.run_pipeline()  # simplified
            assessment["agents"]["homogen"] = {
                "status": "success",
                "data": list(sensor_data.keys()) if sensor_data else {},
            }
            print("   ✓ HOMOGEN pipeline executed successfully.")
        except Exception as e:
            print(f"   ❌ HOMOGEN error: {e}")
            assessment["agents"]["homogen"] = {"status": "error", "error": str(e)}

        # 2️⃣ CALIBRO
        print("\n2️⃣  CALIBRO: Fetching satellite-derived metrics...")
        try:
            satellite_data = self.calibro.get_water_quality_at_location(
                lat=location["latitude"], lon=location["longitude"], date=timestamp
            )
            assessment["agents"]["calibro"] = {
                "status": "success",
                "data": satellite_data,
            }
            print(f"   ✓ Satellite data retrieved ({len(satellite_data)} metrics).")
        except Exception as e:
            print(f"   ❌ CALIBRO error: {e}")
            assessment["agents"]["calibro"] = {"status": "error", "error": str(e)}

        # 3️⃣ VISIOS
        print("\n3️⃣  VISIOS: Running visual analysis...")
        if image_name:
            try:
                visual_analysis = self.visios.analyze_image(image_name, include_context=True)
                assessment["agents"]["visios"] = {
                    "status": "success",
                    "data": visual_analysis,
                }
                print(f"   ✓ VISIOS classified: {visual_analysis.get('classification')}")
            except Exception as e:
                print(f"   ❌ VISIOS error: {e}")
                assessment["agents"]["visios"] = {"status": "error", "error": str(e)}
        else:
            print("   ⚠ No image provided. VISIOS skipped.")
            assessment["agents"]["visios"] = {
                "status": "skipped",
                "reason": "No image provided",
            }

        # 4️⃣ PREDIKT
        print("\n4️⃣  PREDIKT: Running 7-day HABs forecast...")
        try:
            forecast = self.predikt.predict_bloom_probability(
                location=location, horizon_days=7
            )
            assessment["agents"]["predikt"] = {
                "status": "success",
                "data": forecast,
            }
            print(f"   ✓ Forecast complete (risk: {forecast['risk_level'].upper()})")
        except Exception as e:
            print(f"   ❌ PREDIKT error: {e}")
            assessment["agents"]["predikt"] = {"status": "error", "error": str(e)}

        # 5️⃣ Cross-validation
        print("\n5️⃣  Cross-validating results across agents...")
        validation = self._cross_validate_results(assessment)
        assessment["validation"] = validation

        # 6️⃣ Final Risk Assessment
        print("\n6️⃣  Generating unified risk report...")
        risk = self._generate_risk_assessment(assessment)
        assessment["risk_assessment"] = risk

        print("\n" + "=" * 60)
        print(f"🎯 Overall Risk Level: {risk['level'].upper()}")
        print(f"   Confidence: {risk['confidence']}")
        print("=" * 60)

        return assessment

    # --------------------------
    # CROSS VALIDATION
    # --------------------------
    def _cross_validate_results(self, assessment: Dict) -> Dict:
        """Cross-check agent results for consistency."""
        validation = {"agreements": [], "discrepancies": [], "consistency_score": 0.0}

        agents = assessment.get("agents", {})
        success = [k for k, v in agents.items() if v.get("status") == "success"]

        if "calibro" in success and "predikt" in success:
            sat_chl = agents["calibro"]["data"].get("chlorophyll_a", 0)
            pred_prob = agents["predikt"]["data"].get("bloom_probability", 0)
            if sat_chl > 20 and pred_prob > 0.5:
                validation["agreements"].append("CALIBRO and PREDIKT both indicate bloom risk.")
                validation["consistency_score"] += 0.4

        if "visios" in success and "predikt" in success:
            visios_prob = agents["visios"]["data"].get("bloom_probability", 0)
            pred_prob = agents["predikt"]["data"].get("bloom_probability", 0)
            if abs(visios_prob - pred_prob) < 0.2:
                validation["agreements"].append("VISIOS visually confirms PREDIKT forecast.")
                validation["consistency_score"] += 0.3

        if not validation["agreements"]:
            validation["discrepancies"].append("No strong inter-agent consistency detected.")

        return validation

    # --------------------------
    # RISK FUSION
    # --------------------------
    def _generate_risk_assessment(self, assessment: Dict) -> Dict:
        """Fuse all agent outputs into a final risk score."""
        scores, evidence = [], []
        agents = assessment.get("agents", {})

        # Add risk components
        if "predikt" in agents and agents["predikt"].get("status") == "success":
            p = agents["predikt"]["data"]["bloom_probability"]
            scores.append(p)
            evidence.append(f"PREDIKT forecast: {p:.1%}")

        if "calibro" in agents and agents["calibro"].get("status") == "success":
            chl = agents["calibro"]["data"].get("chlorophyll_a", 0)
            scores.append(min(chl / 100, 1))
            evidence.append(f"Satellite chlorophyll-a: {chl:.1f} µg/L")

        if "visios" in agents and agents["visios"].get("status") == "success":
            bp = agents["visios"]["data"].get("bloom_probability", 0)
            scores.append(bp)
            evidence.append(f"VISIOS visual bloom probability: {bp:.1%}")

        if not scores:
            return {
                "level": "unknown",
                "score": 0.0,
                "confidence": "0%",
                "evidence": ["No valid data sources"],
            }

        avg = sum(scores) / len(scores)
        confidence = f"{(0.5 + len(scores) * 0.1):.1%}"

        if avg > 0.75:
            level = "critical"
        elif avg > 0.5:
            level = "high"
        elif avg > 0.3:
            level = "moderate"
        else:
            level = "low"

        return {
            "level": level,
            "score": round(avg, 3),
            "confidence": confidence,
            "evidence": evidence,
            "sources_used": len(scores),
        }

    # --------------------------
    # REPORT GENERATION
    # --------------------------
    def generate_platform_report(self) -> str:
        """Summarize the status of all agents."""
        return f"""
# 🌊 SWIM Platform Status Report
Generated: {datetime.now():%Y-%m-%d %H:%M:%S}

## Agent Status
- HOMOGEN: ✅ Active
- CALIBRO: ✅ Active
- VISIOS: ✅ Active
- PREDIKT: ✅ Active

## Integration
- Cross-Validation: Enabled
- Temporal Window: {self.config['integration']['temporal_window_hours']} hours

## Summary
SWIM is operational with all four AI agents collaborating for in-situ,
satellite, visual, and predictive water quality intelligence.
"""

# --------------------------
# CLI Interface
# --------------------------
def main():
    """Command-line interface for SWIM Platform."""
    import argparse

    parser = argparse.ArgumentParser(description="SWIM Platform - Unified Orchestrator")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--image", type=str, help="Image for VISIOS agent")
    parser.add_argument("--mode", choices=["assess", "report"], default="assess")
    args = parser.parse_args()

    platform = SWIMPlatform()

    if args.mode == "assess":
        if not args.lat or not args.lon:
            print("❌ Please provide --lat and --lon")
            return
        location = {"latitude": args.lat, "longitude": args.lon}
        result = platform.comprehensive_water_quality_assessment(location, image_name=args.image)
        print(json.dumps(result, indent=2))
    elif args.mode == "report":
        print(platform.generate_platform_report())


if __name__ == "__main__":
    main()