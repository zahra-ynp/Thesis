"""Paired inference for the monthly predictions produced by notebook 04.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/processed/europe/ml_benchmarks"
FAMILY = "Two-state AR-HMM"
RICH_INFORMATION = "Lag 1 + macro + sentiment"


def compare_monthly_forecasts(
    baseline: pd.DataFrame,
    augmented: pd.DataFrame,
    *,
    hac_lags: int = 3,
    block_length: int = 6,
    bootstrap_samples: int = 20000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Test equal MSE and estimate the uncertainty in relative RMSE reduction.

    DM uses Bartlett/Newey-West covariance (autocovariances divided by n),
    the Harvey-Leybourne-Newbold h=1 correction and Student t(n-1) inference.
    This h=1 denotes non-overlapping monthly targets, including month-end
    nowcasts; it does not imply that target-month news was available earlier.
    The percentile CI resamples paired errors in circular blocks of six months.
    Settings are fixed independently of the resulting p-value. Inference is
    approximate and conditional on fitted forecasts, not on model selection.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    for label, frame in (("baseline", baseline), ("augmented", augmented)):
        if not {"Date", "Actual", "Forecast"}.issubset(frame.columns):
            raise ValueError(f"{label} requires Date, Actual and Forecast columns.")

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame[["Date", "Actual", "Forecast"]].copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
        frame = frame.set_index("Date").sort_index()
        if frame.index.hasnans or frame.index.has_duplicates:
            raise ValueError("Forecast dates must be present and unique.")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Forecasts and actuals must be finite; no rows are dropped.")
        months = frame.index.to_period("M")
        if len(months) > 1 and not np.all(np.diff(months.asi8) == 1):
            raise ValueError("Forecasts must cover consecutive, unique months.")
        return frame

    base, rich = prepare(baseline), prepare(augmented)
    if not base.index.equals(rich.index):
        raise ValueError("Both models must have exactly the same forecast dates.")
    if not np.allclose(base.Actual, rich.Actual, rtol=0, atol=1e-10):
        raise ValueError("Actual inflation values disagree between models.")
    n = len(base)
    if n < 12:
        raise ValueError("At least 12 paired monthly observations are required.")
    for value, name in ((hac_lags, "hac_lags"), (block_length, "block_length"),
                        (bootstrap_samples, "bootstrap_samples")):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer.")
    if not 0 <= hac_lags < n or not 1 <= block_length <= n // 2:
        raise ValueError("Invalid HAC lag or block length for the sample size.")
    if bootstrap_samples < 1000:
        raise ValueError("Use at least 1000 bootstrap samples.")

    base_loss = (base.Actual.to_numpy() - base.Forecast.to_numpy()) ** 2
    rich_loss = (rich.Actual.to_numpy() - rich.Forecast.to_numpy()) ** 2
    base_rmse, rich_rmse = float(np.sqrt(base_loss.mean())), float(np.sqrt(rich_loss.mean()))
    if base_rmse == 0:
        raise ValueError("Relative improvement is undefined for a perfect baseline.")
    differential = base_loss - rich_loss
    mean_difference = float(differential.mean())
    centered = differential - mean_difference
    long_run_variance = float(centered @ centered / n)
    for lag in range(1, hac_lags + 1):
        weight = 1 - lag / (hac_lags + 1)
        long_run_variance += 2 * weight * float(centered[lag:] @ centered[:-lag] / n)
    correction = float(np.sqrt((n - 1) / n))
    if np.array_equal(base_loss, rich_loss):
        statistic, pvalue, status = 0.0, 1.0, "identical_losses"
    elif long_run_variance <= 0 or not np.isfinite(long_run_variance):
        statistic, pvalue, status = None, None, "undefined_loss_variance"
    else:
        statistic = float(correction * mean_difference / np.sqrt(long_run_variance / n))
        pvalue = float(2 * student_t.sf(abs(statistic), df=n - 1))
        status = "ok"

    # Resample the same chronological indices for both models, retaining pairing.
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(bootstrap_samples, int(np.ceil(n / block_length))))
    indices = ((starts[:, :, None] + np.arange(block_length)) % n).reshape(bootstrap_samples, -1)[:, :n]
    boot_base = np.sqrt(base_loss[indices].mean(axis=1))
    boot_rich = np.sqrt(rich_loss[indices].mean(axis=1))
    if np.any(boot_base == 0):
        ci, bootstrap_status = None, "undefined_relative_rmse_in_resamples"
    else:
        improvements = 100 * (1 - boot_rich / boot_base)
        ci = np.quantile(improvements, [alpha / 2, 1 - alpha / 2]).tolist()
        bootstrap_status = "ok"
    acf1_denominator = float(centered @ centered)
    return {
        "test_months": n,
        "test_start": str(base.index.min().date()),
        "test_end": str(base.index.max().date()),
        "baseline_rmse": base_rmse,
        "augmented_rmse": rich_rmse,
        "rmse_improvement_pct": float(100 * (1 - rich_rmse / base_rmse)),
        "mean_squared_loss_difference": mean_difference,
        "loss_difference_lag1_autocorrelation": (
            float(centered[1:] @ centered[:-1] / acf1_denominator)
            if acf1_denominator else None
        ),
        "dm": {
            "statistic": statistic, "pvalue_two_sided": pvalue,
            "degrees_of_freedom": n - 1, "hac_lags": hac_lags,
            "long_run_variance": long_run_variance,
            "hln_correction": correction, "status": status,
            "significant_improvement": bool(pvalue is not None and pvalue < alpha and mean_difference > 0),
        },
        "bootstrap": {
            "method": "paired circular block percentile interval",
            "block_length": block_length, "samples": bootstrap_samples,
            "seed": seed, "confidence_level": 1 - alpha,
            "rmse_improvement_pct_ci": ci, "status": bootstrap_status,
        },
        "alpha": alpha,
    }


def analyse_europe_benchmarks(predictions: pd.DataFrame, *, output_dir: Path | None = None) -> dict:
    """Compare the richer AR-HMM with inflation only and with macro only.

    The first contrast answers the requested combined-input comparison; the
    second isolates the incremental news inputs and explains the earlier 11%.
    Holm correction covers these two exploratory, two-sided loss tests.
    """
    selected = predictions.loc[predictions.family.eq(FAMILY)]
    expected_dates = pd.date_range("2022-01-01", "2025-12-01", freq="MS")
    for information in ("Inflation lag 1", "Lag 1 + macro", RICH_INFORMATION):
        dates = pd.to_datetime(selected.loc[selected.information.eq(information), "Date"]).sort_values()
        if not np.array_equal(dates.to_numpy(), expected_dates.to_numpy()):
            raise ValueError(f"{information} must contain exactly the 48 final-test months.")
    richer = selected.loc[selected.information.eq(RICH_INFORMATION)]
    comparisons = []
    for information in ("Inflation lag 1", "Lag 1 + macro"):
        baseline = selected.loc[selected.information.eq(information)]
        comparison = compare_monthly_forecasts(baseline, richer)
        comparisons.append({
            "family": FAMILY, "baseline_information": information,
            "augmented_information": RICH_INFORMATION, **comparison,
        })
    order = sorted(range(len(comparisons)), key=lambda i: (
        comparisons[i]["dm"]["pvalue_two_sided"]
        if comparisons[i]["dm"]["pvalue_two_sided"] is not None else 1.0
    ))
    previous = 0.0
    for rank, index in enumerate(order):
        item = comparisons[index]
        pvalue = item["dm"]["pvalue_two_sided"]
        adjusted = None if pvalue is None else max(previous, min(1.0, (len(order) - rank) * pvalue))
        if adjusted is not None:
            previous = adjusted
        item["dm"]["pvalue_holm"] = adjusted
        item["dm"]["significant_improvement_holm"] = bool(
            adjusted is not None and adjusted < item["alpha"]
            and item["mean_squared_loss_difference"] > 0
        )
    report = {
        "design": {
            "analysis": "exploratory inference added after inspecting benchmark results",
            "test_period": "2022-01 to 2025-12",
            "null": "equal expected squared prediction loss",
            "alternative": "unequal expected squared prediction loss (two-sided)",
            "positive_difference": "baseline squared error minus augmented squared error",
            "primary_test": "Diebold-Mariano with Bartlett HAC and HLN h=1 correction",
            "multiple_comparisons": "Holm over the two AR-HMM contrasts",
            "intervals": "95% pointwise paired circular-block bootstrap intervals; not simultaneous",
            "limitations": [
                "Macro plus news versus inflation only does not isolate the news contribution.",
                "Target-month Guardian news makes the richer specification a month-end nowcast.",
                "Approximate inference assumes a sufficiently stable, weakly dependent loss process.",
                "Nested regime-switching models, 48 observations and the 2022 shock limit calibration.",
                "Forecasts are held fixed; intervals do not include estimation or model-selection uncertainty.",
                "Non-significance is insufficient evidence of improvement, not proof of no effect.",
            ],
        },
        "software": {"numpy": np.__version__, "pandas": pd.__version__},
        "comparisons": comparisons,
    }
    import scipy
    report["software"]["scipy"] = scipy.__version__
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "significance.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        significance_table(report).to_csv(output_dir / "significance.csv", index=False)
    return report


def significance_table(report: dict) -> pd.DataFrame:
    rows = []
    for item in report["comparisons"]:
        interval = item["bootstrap"]["rmse_improvement_pct_ci"] or [None, None]
        rows.append({
            "Baseline": item["baseline_information"], "Months": item["test_months"],
            "Baseline RMSE": item["baseline_rmse"], "Macro + news RMSE": item["augmented_rmse"],
            "RMSE improvement (%)": item["rmse_improvement_pct"],
            "95% CI lower (%)": interval[0], "95% CI upper (%)": interval[1],
            "DM statistic": item["dm"]["statistic"],
            "DM p-value (two-sided)": item["dm"]["pvalue_two_sided"],
            "DM p-value (Holm)": item["dm"]["pvalue_holm"],
            "Significant improvement (5%, Holm)": item["dm"]["significant_improvement_holm"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=OUTPUT_DIR / "forecasts.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions, parse_dates=["Date"])
    report = analyse_europe_benchmarks(predictions, output_dir=args.output_dir)
    print(significance_table(report).to_string(index=False))


if __name__ == "__main__":
    main()
