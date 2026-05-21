from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate qualification experiment report.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "data" / "models" / "qualification_window_metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "experiment_report.md",
    )
    return parser.parse_args()


def metric_table(metrics: dict) -> str:
    lines = [
        "| Window | Model | Mag RMSE | Time RMSE (h) | Time Hit Rate |",
        "|:--|:--|--:|--:|--:|",
    ]
    for window, models in metrics.get("window_metrics", {}).items():
        if "legal_risk_fusion" in models:
            rows = {"legal_risk_fusion": models["legal_risk_fusion"]}
            rows.update(models.get("models", {}))
        else:
            rows = models
        for model_name, values in rows.items():
            lines.append(
                "| {window} | {model} | {mag:.4f} | {time:.4f} | {hit:.2%} |".format(
                    window=window,
                    model=model_name,
                    mag=float(values.get("mag_rmse", 0.0)),
                    time=float(values.get("time_hour_rmse", 0.0)),
                    hit=float(values.get("time_hour_hit_rate", 0.0)),
                )
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    metrics_path = resolve_project_path(args.metrics)
    output_path = resolve_project_path(args.output)
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    lines = [
        "# Qualification Experiment Report",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Objective",
        "",
        "This run targets the qualification submission format: T1 (0-24h), "
        "T2 (24-72h), and T3 (72-168h). Each test sequence produces two "
        "prediction files, one for T1/T2 and one for T3.",
        "",
        "## Commands",
        "",
        "```bash",
        "python main.py build-qualification-labels",
        "python main.py train-legal-fusion --model-type both --device cuda",
        "python main.py make-qualification-package --commitment-template /path/to/template",
        "```",
        "",
        "## Metrics",
        "",
    ]
    if metrics:
        lines.append(metric_table(metrics))
    else:
        lines.append("Metrics are not available yet. Run `train-window-baseline` first.")
    lines.extend(
        [
            "",
            "## Packaging",
            "",
            "The generated ZIP contains `predictions/`, `technical_docs/`, "
            "`commitment/`, and `MANIFEST.json`. Large raw data files and model "
            "weights are intentionally excluded from the code snapshot.",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Experiment report saved: {output_path}")


if __name__ == "__main__":
    main()
