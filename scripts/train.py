"""Train, validate and (optionally) test the traffic forecaster.

    python -m scripts.train                  # select on validation days, save model
    python -m scripts.train --evaluate-test  # same, then score the held-out test days

Protocol (chronological, no shuffling):
    days  1-40  train
    days 41-48  validation: choose the loss, calibrate the prediction ranges
    days 49-56  test: scored only with --evaluate-test, after everything is fixed

The saved model is refit on days 1-48. The dashboard replays days 49-56, so
every forecast it shows is out-of-sample.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_predictor.forecasting.data import (  # noqa: E402
    OUTAGE_MIN_RUN,
    STEP_MINUTES,
    STEPS_PER_DAY,
    load_dataset,
)
from traffic_predictor.forecasting.features import FEATURE_NAMES, build_features  # noqa: E402
from traffic_predictor.forecasting.model import (  # noqa: E402
    CANDIDATES,
    HORIZONS,
    INTERVAL_LEVEL,
    LEVEL_PERCENTILES,
    Forecaster,
    baseline_predictions,
    horizon_minutes,
    make_estimator,
    interval_scale,
    regression_metrics,
    typical_day_profile,
)

TRAIN_DAYS = 40
VALIDATION_DAYS = 8


def _clean(obj):
    """Make metrics JSON-safe (NaN -> null, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(obj) else round(float(obj), 4)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "traffic-prediction-dataset.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument("--evaluate-test", action="store_true", help="score the held-out test days")
    parser.add_argument(
        "--allow-retest",
        action="store_true",
        help="permit --evaluate-test when the saved metrics already contain test results",
    )
    args = parser.parse_args(argv)
    if not args.data.exists():
        parser.error(
            f"{args.data} not found. Download the dataset first (see README): "
            "kaggle datasets download -d annsanababy/road-traffic-prediction-dataset"
        )

    metrics_path = args.out / "metrics.json"
    previous_test_runs = 0
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        previous_test_runs = int(previous.get("test_evaluation_count", 0))
    if args.evaluate_test and previous_test_runs and not args.allow_retest:
        parser.error(
            f"{metrics_path} already holds test results ({previous_test_runs} run(s)). "
            "Re-scoring the test days after changing the model turns them into validation "
            "data. Pass --allow-retest to do it anyway; the count is recorded."
        )

    started = time.perf_counter()
    ds = load_dataset(args.data)
    counts = ds.counts.to_numpy()
    n_steps, n_junctions = counts.shape
    n_days = n_steps // STEPS_PER_DAY
    train_end = TRAIN_DAYS * STEPS_PER_DAY
    val_end = (TRAIN_DAYS + VALIDATION_DAYS) * STEPS_PER_DAY
    if n_days < TRAIN_DAYS + VALIDATION_DAYS + 1:
        parser.error(f"need at least {TRAIN_DAYS + VALIDATION_DAYS + 1} days of data, got {n_days}")
    print(f"data: {n_steps} steps x {n_junctions} junctions ({n_days} days), "
          f"{int(ds.outage.sum())} outage rows treated as missing")

    frames = {h: build_features(counts, h) for h in HORIZONS}

    def rows(h: int, lo: int, hi: int) -> np.ndarray:
        f = frames[h]
        target_step = f.origin + h
        return (target_step >= lo) & (target_step < hi) & ~np.isnan(f.target)

    # 1. Choose the loss on the validation days (model fit on train days only).
    selection: dict[str, dict[int, dict]] = {}
    val_predictions: dict[str, dict[int, np.ndarray]] = {}
    for name in CANDIDATES:
        selection[name], val_predictions[name] = {}, {}
        for h in HORIZONS:
            f, tr, va = frames[h], rows(h, 0, train_end), rows(h, train_end, val_end)
            est = make_estimator(name).fit(f.X.loc[tr, FEATURE_NAMES], f.target[tr])
            pred = np.clip(est.predict(f.X.loc[va, FEATURE_NAMES]), 0, None)
            val_predictions[name][h] = pred
            selection[name][h] = regression_metrics(f.target[va], pred)
        mean_mae = np.mean([selection[name][h]["mae"] for h in HORIZONS])
        print(f"validation  {name:<20} mean MAE over horizons {mean_mae:.3f}")
    selected = min(CANDIDATES, key=lambda n: np.mean([selection[n][h]["mae"] for h in HORIZONS]))
    print(f"selected: {selected}")

    # 2. Prediction ranges: quantiles of the train-only model's validation residuals,
    #    divided by interval_scale(forecast) so ranges widen at busy times.
    alpha = (1 - INTERVAL_LEVEL) / 2
    interval_offsets: dict[int, list[tuple[float, float]]] = {}
    validation: dict[int, dict] = {}
    for h in HORIZONS:
        f, va = frames[h], rows(h, train_end, val_end)
        val_pred = val_predictions[selected][h]
        resid = (f.target[va] - val_pred) / interval_scale(val_pred)
        offsets = []
        for j in range(n_junctions):
            r = resid[f.junction[va] == j]
            offsets.append((float(np.quantile(r, alpha)), float(np.quantile(r, 1 - alpha))))
        interval_offsets[h] = offsets
        base = baseline_predictions(counts, h)
        ok = va & ~np.isnan(base["persistence"]) & ~np.isnan(base["same_time_yesterday"])
        assert ok.sum() == va.sum(), "baselines missing on validation rows"
        validation[h] = {
            "horizon_minutes": horizon_minutes(h),
            "model": selection[selected][h],
            **{b: regression_metrics(f.target[va], v[va]) for b, v in base.items()},
        }

    # 3. Refit the selected loss on train + validation days.
    models = {}
    for h in HORIZONS:
        f, fit = frames[h], rows(h, 0, val_end)
        models[h] = make_estimator(selected).fit(f.X.loc[fit, FEATURE_NAMES], f.target[fit])

    history = counts[:val_end]
    observed = [history[:, j][~np.isnan(history[:, j])] for j in range(n_junctions)]
    forecaster = Forecaster(
        junctions=list(ds.junctions),
        models=models,
        interval_offsets=interval_offsets,
        level_thresholds=[tuple(float(np.percentile(o, p)) for p in LEVEL_PERCENTILES) for o in observed],
        reference_quantiles=[np.percentile(o, np.arange(101)).tolist() for o in observed],
        typical_day=typical_day_profile(history).T.tolist(),
        fit_end_step=val_end,
        test_start_step=val_end,
        info={
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "estimator": f"HistGradientBoostingRegressor ({selected})",
            "sklearn_version": sklearn.__version__,
            "feature_names": FEATURE_NAMES,
            "fit_days": [1, TRAIN_DAYS + VALIDATION_DAYS],
            "interval_method": "validation residual quantiles, scaled by sqrt(forecast), per junction and horizon",
        },
    )

    # 4. Test days: only on request, after every choice above is fixed.
    test = None
    if args.evaluate_test:
        test = {}
        for h in HORIZONS:
            f, te = frames[h], rows(h, val_end, n_steps)
            pred = forecaster.predict(counts, h).reshape(-1)[te]
            y, junction = f.target[te], f.junction[te]
            lower, upper = forecaster.interval(h, junction, pred)
            inside = (y >= lower) & (y <= upper)
            q1, q2 = np.quantile(pred, [1 / 3, 2 / 3])
            load_bins = {"quiet": pred < q1, "middle": (pred >= q1) & (pred < q2), "busy": pred >= q2}
            base = baseline_predictions(counts, h)
            test[h] = {
                "horizon_minutes": horizon_minutes(h),
                "model": regression_metrics(y, pred),
                **{b: regression_metrics(y, v[te]) for b, v in base.items()},
                "interval": {
                    "nominal_coverage": INTERVAL_LEVEL,
                    "observed_coverage": float(inside.mean()),
                    "coverage_by_forecast_load": {k: float(inside[m].mean()) for k, m in load_bins.items()},
                    "mean_width": float((upper - lower).mean()),
                },
                "model_mae_by_junction": {
                    ds.junctions[j]: float(np.abs(pred[junction == j] - y[junction == j]).mean())
                    for j in range(n_junctions)
                },
            }
            m, p = test[h]["model"]["mae"], test[h]["persistence"]["mae"]
            print(f"test  +{horizon_minutes(h):>2} min  MAE model {m:.2f}  persistence {p:.2f}  "
                  f"coverage {test[h]['interval']['observed_coverage']:.1%}")

    forecaster.save(args.out / "forecaster.joblib")
    test_runs = previous_test_runs + (1 if args.evaluate_test else 0)
    report = {
        "dataset": {
            "file": args.data.name,
            "steps": n_steps,
            "junctions": ds.junctions,
            "days": n_days,
            "step_minutes_inferred": STEP_MINUTES,
            "outage_rule": f"runs of >= {OUTAGE_MIN_RUN} steps with every junction at 0",
            "outage_steps": int(ds.outage.sum()),
        },
        "split_days": {
            "train": [1, TRAIN_DAYS],
            "validation": [TRAIN_DAYS + 1, TRAIN_DAYS + VALIDATION_DAYS],
            "test": [TRAIN_DAYS + VALIDATION_DAYS + 1, n_days],
        },
        "horizons_minutes": [horizon_minutes(h) for h in HORIZONS],
        "selection": {
            "rule": "lowest mean validation MAE over horizons",
            "selected": selected,
            "validation_mae": {n: {horizon_minutes(h): selection[n][h]["mae"] for h in HORIZONS} for n in CANDIDATES},
        },
        "validation": {horizon_minutes(h): v for h, v in validation.items()},
        "final_model": forecaster.info,
        "interval_level": INTERVAL_LEVEL,
        "test_evaluated": test is not None or previous_test_runs > 0,
        "test_evaluation_count": test_runs,
        "test": {horizon_minutes(h): v for h, v in test.items()} if test else (
            previous.get("test") if previous_test_runs else None
        ),
        "training_seconds": round(time.perf_counter() - started, 1),
    }
    if not args.evaluate_test and previous_test_runs:
        report["note"] = (
            "Test results are from an earlier run; the model has been refit since then. "
            "Re-score with --evaluate-test --allow-retest if the model changed."
        )
    metrics_path.write_text(json.dumps(_clean(report), indent=2), encoding="utf-8")
    print(f"saved {args.out / 'forecaster.joblib'} and {metrics_path} "
          f"in {report['training_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
