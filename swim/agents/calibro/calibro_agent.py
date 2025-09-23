"""
CALIBRO Agent: Calibrates EO data using in-situ measurements (e.g., chlorophyll-a, turbidity).
- Trains regression models (RF, XGBoost) on satellite-in-situ pairs.
- Computes error metrics and generates predictions.
"""



# agents/calibro/calibro_agent.py

from swim.agents.calibro.core_pipeline import run_calibro_pipeline


def main():
    print("🚀 CALIBRO Agent – Satellite-based HAB Analysis")
    print("Running the full pipeline...")

    results = run_calibro_pipeline()

    print("✅ Pipeline completed for the following lakes:")
    for lake_name, output in results.items():
        print(f" - {lake_name}: {output.get('status', 'unknown')}")

if __name__ == "__main__":
    main()