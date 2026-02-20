# swim/agents/visios/visual_interface.py

import argparse
from datetime import datetime
from pathlib import Path

from swim.agents.visios.visios_agent import VisiosAgent


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def generate_dashboard(output_path: Path) -> Path:
    agent = VisiosAgent()
    summary = agent.summarize_batch()
    history = agent.get_analysis_history(limit=10)

    if "error" in summary:
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>VISIOS Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; background: #f7f7f7; }}
    .card {{ background: #fff; padding: 20px; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>VISIOS Dashboard</h1>
  <div class="card">
    <p>{summary['error']}</p>
  </div>
</body>
</html>"""
        output_path.write_text(html, encoding="utf-8")
        return output_path

    stats = summary.get("statistics", {})
    total = stats.get("total_images", 0)
    avg_score = stats.get("average_bloom_score", 0.0)
    max_score = stats.get("max_bloom_score", 0.0)
    min_score = stats.get("min_bloom_score", 0.0)

    distribution_rows = ""
    for label, count in summary.get("summary", {}).items():
        pct = (count / total) if total else 0.0
        distribution_rows += f"<tr><td>{label}</td><td>{count}</td><td>{_fmt_pct(pct)}</td></tr>"

    history_rows = ""
    for entry in reversed(history):
        history_rows += (
            f"<tr><td>{entry.get('image')}</td>"
            f"<td>{entry.get('classification')}</td>"
            f"<td>{_fmt_pct(entry.get('score', 0.0))}</td>"
            f"<td>{entry.get('timestamp')}</td></tr>"
        )

    high_risk_rows = ""
    for loc in summary.get("high_risk_locations", [])[:10]:
        high_risk_rows += (
            f"<tr><td>{loc.get('image')}</td>"
            f"<td>{loc.get('classification')}</td>"
            f"<td>{_fmt_pct(loc.get('score', 0.0))}</td></tr>"
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>VISIOS Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; background: #f7f7f7; color: #111; }}
    h1 {{ margin-bottom: 6px; }}
    .muted {{ color: #666; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
    .card {{ background: #fff; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #eee; }}
    .pill {{ display: inline-block; padding: 2px 8px; border-radius: 12px; background: #eef; }}
  </style>
</head>
<body>
  <h1>VISIOS Dashboard</h1>
  <div class="muted">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>

  <div class="grid" style="margin-top: 16px;">
    <div class="card">
      <div class="muted">Images analyzed</div>
      <div style="font-size: 24px;">{total}</div>
    </div>
    <div class="card">
      <div class="muted">Average bloom score</div>
      <div style="font-size: 24px;">{_fmt_pct(avg_score)}</div>
    </div>
    <div class="card">
      <div class="muted">Max bloom score</div>
      <div style="font-size: 24px;">{_fmt_pct(max_score)}</div>
    </div>
    <div class="card">
      <div class="muted">Min bloom score</div>
      <div style="font-size: 24px;">{_fmt_pct(min_score)}</div>
    </div>
  </div>

  <div class="card" style="margin-top: 16px;">
    <h3>Classification Distribution</h3>
    <table>
      <thead><tr><th>Class</th><th>Count</th><th>Share</th></tr></thead>
      <tbody>{distribution_rows}</tbody>
    </table>
  </div>

  <div class="card" style="margin-top: 16px;">
    <h3>High Risk Images</h3>
    <table>
      <thead><tr><th>Image</th><th>Class</th><th>Score</th></tr></thead>
      <tbody>{high_risk_rows or "<tr><td colspan='3'>None detected</td></tr>"}</tbody>
    </table>
  </div>

  <div class="card" style="margin-top: 16px;">
    <h3>Recent Analyses</h3>
    <table>
      <thead><tr><th>Image</th><th>Class</th><th>Score</th><th>Timestamp</th></tr></thead>
      <tbody>{history_rows or "<tr><td colspan='4'>No history</td></tr>"}</tbody>
    </table>
  </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VISIOS HTML dashboard")
    parser.add_argument(
        "--output",
        default="outputs/visios/visios_dashboard.html",
        help="Output path for dashboard HTML",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    path = generate_dashboard(output_path)
    print(f"Dashboard saved to {path}")


if __name__ == "__main__":
    main()
