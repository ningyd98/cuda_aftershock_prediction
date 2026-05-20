# Qualification Experiment Report

Generated as the baseline technical report template for the qualification package.

## Objective

The qualification task requires one ZIP package containing prediction files,
technical materials, and the official commitment letter. For every test
mainshock, the prediction output is split into:

- `{YYYYMMDDhhmmss}-T1-T2.csv`: two rows for T1 (0-24h) and T2 (24-72h).
- `{YYYYMMDDhhmmss}-T3.csv`: one row for T3 (72-168h).

Each row is written without a header and follows the official sample format:

```text
mainshock_time longitude latitude mainshock_mag predicted_aftershock_mag (Ms) predicted_aftershock_time_YYYYMMDDhh
```

## Reproduction Commands

```bash
python main.py build-qualification-labels
python main.py train-window-baseline --model-type both --device cuda
python main.py generate-experiment-report
python main.py make-qualification-package --commitment-template /path/to/official_commitment_template
```

## Notes

Full CUDA experiments should be run on the remote GPU host. The generated ZIP
manifest records the sequence count, prediction files, code snapshot, technical
report, and copied official commitment letter.
