# Qualification Experiment Report

Generated at: 2026-05-20T16:24:39.920580+00:00

## Objective

This run targets the qualification submission format: T1 (0-24h), T2 (24-72h), and T3 (72-168h). Each test sequence produces two prediction files, one for T1/T2 and one for T3.

## Commands

```bash
python main.py build-qualification-labels
python main.py train-window-baseline --model-type both --device cuda
python main.py make-qualification-package --commitment-template /path/to/template
```

## Metrics

| Window | Model | Mag RMSE | Time RMSE (h) | Time Hit Rate |
|:--|:--|--:|--:|--:|
| T1 | baseline | 0.1888 | 2.8781 | 84.38% |
| T2 | baseline | 0.3398 | 10.6566 | 72.61% |
| T3 | baseline | 2.0254 | 20.3505 | 77.32% |

## Packaging

The generated ZIP contains `predictions/`, `technical_docs/`, `commitment/`, and `MANIFEST.json`. Large raw data files and model weights are intentionally excluded from the code snapshot.
