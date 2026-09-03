from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def forecast_metrics(actual, forecast) -> dict[str, float]:
    """Calculate the three metrics used throughout the existing thesis."""
    actual_values = np.asarray(actual, dtype=float)
    forecast_values = np.asarray(forecast, dtype=float)
    if actual_values.shape != forecast_values.shape:
        raise ValueError("Actual and forecast arrays must have the same shape.")
    errors = actual_values - forecast_values
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "mean_error_actual_minus_forecast": float(np.mean(errors)),
    }


SAVED_FORECASTS = {
    "AR(1)": ("ar1_forecasts.csv", "Forecast"),
    "ARX + sentiment": ("arx_forecasts.csv", "Forecast"),
    "HMM": ("hmm_forecasts.csv", "Forecast"),
    "AR-HMM": ("ar_hmm_forecasts.csv", "Forecast_ARHMM"),
    "AR-HMM + sentiment": ("ar_hmm_forecasts.csv", "Forecast_ARHMM_With_Sentiment"),
    "LASSO inflation only": ("lasso_forecasts.csv", "Forecast_Inflation_Only"),
    "LASSO inflation + sentiment": ("lasso_forecasts.csv", "Forecast_With_Sentiment"),
}


def load_saved_comparison(results_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load legacy results into one common forecast frame and metric table."""
    results_dir = Path(results_dir)
    if (results_dir / "baseline" / "forecasts.csv").exists():
        results_dir = results_dir / "baseline"
    if (results_dir / "forecasts.csv").exists():
        forecasts = pd.read_csv(results_dir / "forecasts.csv", parse_dates=["Date"])
        models = [c for c in forecasts if c not in ("Date", "Actual")]
        metrics = pd.DataFrame([
            {"model": model, **forecast_metrics(forecasts["Actual"], forecasts[model])}
            for model in models
        ]).sort_values("rmse").reset_index(drop=True)
        return forecasts, metrics
    merged: pd.DataFrame | None = None
    for model, (filename, forecast_column) in SAVED_FORECASTS.items():
        frame = pd.read_csv(results_dir / filename, parse_dates=["Date"])
        selected = frame[["Date", "Actual", forecast_column]].rename(
            columns={forecast_column: model}
        )
        if merged is None:
            merged = selected
        else:
            selected = selected.rename(columns={"Actual": "_actual_check"})
            merged = merged.merge(selected, on="Date", how="inner", validate="one_to_one")
            if not np.allclose(merged["Actual"], merged["_actual_check"]):
                raise ValueError(f"Actual values disagree in saved results for {model}.")
            merged = merged.drop(columns="_actual_check")
    if merged is None:
        raise ValueError("No saved forecasts were configured.")

    rows = []
    for model in SAVED_FORECASTS:
        rows.append({"model": model, **forecast_metrics(merged["Actual"], merged[model])})
    metrics = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return merged, metrics


def verify_saved_result_parity(
    results_dir: str | Path,
    *,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Verify refactoring did not change the metrics represented by saved forecasts."""
    results_dir = Path(results_dir)
    _, calculated = load_saved_comparison(results_dir)
    bundle = results_dir / "baseline" if (results_dir / "baseline" / "metrics.csv").exists() else results_dir
    if (bundle / "metrics.csv").exists():
        expected = pd.read_csv(bundle / "metrics.csv").set_index("model").to_dict("index")
    else:
        with (results_dir / "lasso_metrics.json").open(encoding="utf-8") as file:
            expected = json.load(file)["all_model_common_sample_comparison"]["metrics"]

    records = []
    for row in calculated.to_dict("records"):
        model = row.pop("model")
        differences = {metric: abs(row[metric] - expected[model][metric]) for metric in row}
        records.append({
            "model": model,
            "maximum_absolute_difference": max(differences.values()),
            "matches_saved_result": max(differences.values()) <= tolerance,
        })
    return pd.DataFrame(records).sort_values("model").reset_index(drop=True)


def compare_results(results: Mapping[str, object]) -> pd.DataFrame:
    """Compare new ForecastResult-like objects on their shared forecast dates."""
    common: pd.DataFrame | None = None
    for name, result in results.items():
        frame = result.forecasts[["Date", "Actual", "Forecast"]].rename(
            columns={"Forecast": name}
        )
        if common is None:
            common = frame
        else:
            frame = frame.rename(columns={"Actual": "_actual_check"})
            common = common.merge(frame, on="Date", how="inner", validate="one_to_one")
            if not np.allclose(common["Actual"], common["_actual_check"]):
                raise ValueError(f"Actual values disagree across model results for {name}.")
            common = common.drop(columns="_actual_check")
    if common is None or common.empty:
        raise ValueError("No common forecast dates are available for comparison.")

    rows = [
        {"model": name, **forecast_metrics(common["Actual"], common[name])}
        for name in results
    ]
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def save_result_bundle(results: Mapping[str, object], output_dir: str | Path, *, metadata=None) -> pd.DataFrame:
    """Write three coherent files for an experiment, without per-model file proliferation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined, parameters = None, {}
    for result in results.values():
        part = result.forecasts[["Date", "Actual", "Forecast"]].copy()
        part["Date"] = pd.to_datetime(part["Date"])
        part = part.set_index("Date").sort_index().rename(columns={"Forecast": result.name})
        if part.index.has_duplicates or not np.isfinite(part.to_numpy()).all():
            raise ValueError(f"Invalid forecast rows for {result.name}.")
        if combined is None:
            combined = part
        else:
            if not combined.index.equals(part.index) or not np.allclose(combined["Actual"], part["Actual"]):
                raise ValueError("A result bundle requires identical dates and actual values for every model.")
            combined[result.name] = part[result.name]
        parameters[result.name] = result.parameters
    if combined is None:
        raise ValueError("No results to save.")
    metrics = pd.DataFrame([
        {"model": model, **forecast_metrics(combined["Actual"], combined[model])}
        for model in parameters
    ]).sort_values("rmse").reset_index(drop=True)
    baseline = next((name for name in (
        "Inflation only (lag 1)", "Inflation only (lag 2)",
        "AR-HMM inflation only (lag 1)",
    ) if name in parameters), None)
    if baseline is not None:
        base_rmse = float(metrics.loc[metrics["model"].eq(baseline), "rmse"].iloc[0])
        metrics["rmse_improvement_pct"] = 100 * (base_rmse - metrics["rmse"]) / base_rmse
        metrics["macro_count"] = metrics["model"].map(lambda n: len(parameters[n]["features"]) - 1)
        metrics["features"] = metrics["model"].map(lambda n: ", ".join(parameters[n]["features"]))
        metrics["test_months"] = len(combined)
    combined.reset_index().to_csv(output_dir / "forecasts.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "parameters.json").write_text(json.dumps({
        "models": parameters, "metadata": metadata or {},
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metrics
