#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-data/test_sequences/20010126031640_eq.csv}"
OUTPUT="${2:-submission.csv}"

python -m compileall main.py src scripts
python scripts/train_baseline.py \
  --data data/processed/advanced_features.csv \
  --n-splits 5 \
  --model-type both \
  --use-asymmetric-time-objective \
  --save-dir data/models
python scripts/make_submission.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model-dir data/models \
  --allow-rule-fallback

OUTPUT_PATH="$OUTPUT" python - <<'PY'
import os
import pandas as pd

output_path = os.environ["OUTPUT_PATH"]
df = pd.read_csv(output_path)
required = {"mainshock_id", "predicted_max_mag", "predicted_time_to_max"}
missing = required - set(df.columns)
assert not missing, f"submission missing columns: {missing}"
assert df["predicted_max_mag"].notna().all()
assert df["predicted_time_to_max"].notna().all()
assert (df["predicted_time_to_max"] >= 0).all()
print(f"{output_path} format OK")
PY
