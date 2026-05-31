# Qualification Experiment Report

Generated at: 2026-05-20T16:39:40.865901+00:00

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
| T1 | baseline | 0.1888 | 2.8788 | 84.43% |
| T1 | xgboost | 0.1866 | 3.0409 | 84.54% |
| T2 | baseline | 0.3398 | 10.6538 | 72.46% |
| T2 | xgboost | 0.3396 | 9.2521 | 72.84% |
| T3 | baseline | 2.0255 | 20.3528 | 77.32% |
| T3 | xgboost | 1.9889 | 18.4135 | 78.37% |

## Packaging

The generated ZIP contains `predictions/`, `technical_docs/`, `commitment/`, and `MANIFEST.json`. Large raw data files and model weights are intentionally excluded from the code snapshot.
