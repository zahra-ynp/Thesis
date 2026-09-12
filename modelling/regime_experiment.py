"""Controlled AR-HMM experiments for notebook 02."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from modelling.data_pipeline import load_macro_benchmark_snapshot, MACRO_BENCHMARK_FEATURES
from modelling.models import ForecastResult, fit_expanding_ar_hmm, fit_expanding_macro
from modelling.evaluation import save_result_bundle


def run_lag_order_experiment(
    dataset,
    output_dir,
    *,
    lag_orders=range(1, 7),
    test_start="2017-01-01",
    test_end="2023-12-01",
):
    """Compare expanding AR(p) and two-state AR-HMM(p) on one sample."""
    lag_orders = tuple(sorted(set(lag_orders)))
    if not lag_orders or lag_orders[0] < 1:
        raise ValueError("lag_orders must contain positive integers.")

    data, _, _ = load_macro_benchmark_snapshot(Path(dataset))
    lag_columns = [f"Inflation_Lag{lag}" for lag in range(1, max(lag_orders) + 1)]
    sample = data[["Date", "Inflation_MoM"]].copy()
    for lag, column in enumerate(lag_columns, start=1):
        sample[column] = sample["Inflation_MoM"].shift(lag)
    sample = sample.dropna().loc[lambda x: x["Date"].le(test_end)].copy()
    sample["TargetAssumedAvailableBy"] = sample["Date"] + pd.offsets.MonthEnd(2)

    dates = sample.loc[sample["Date"].between(test_start, test_end), "Date"]
    if not pd.DatetimeIndex(dates).equals(pd.date_range(test_start, test_end, freq="MS")):
        raise ValueError("Incomplete evaluation period.")
    schedule = pd.DataFrame({
        "TargetMonth": dates,
        "ForecastOrigin": dates + pd.offsets.MonthEnd(0),
    })

    results = {}
    for order in lag_orders:
        features = lag_columns[:order]
        ar_name = f"AR({order})"
        arhmm_name = f"AR-HMM(2 states, p={order})"
        results[ar_name] = fit_expanding_macro(
            sample, schedule, features=features, name=ar_name,
        )
        print(f"Fitting {arhmm_name}", flush=True)
        results[arhmm_name] = fit_expanding_ar_hmm(
            sample, schedule, features=features, name=arhmm_name, n_states=2,
        )

    output_dir = Path(output_dir)
    metrics = save_result_bundle(results, output_dir, metadata={
        "experiment": "controlled_ar_vs_two_state_arhmm_lag_order",
        "lag_orders": list(lag_orders),
        "n_states": 2,
        "test_start": test_start,
        "test_end": test_end,
        "common_sample_max_lag": max(lag_orders),
        "timing": "Existing expanding month-end convention",
    })
    metrics["family"] = metrics["model"].str.extract(r"^(AR-HMM|AR)")
    metrics["lag_order"] = metrics["model"].str.extract(r"(?:p=|AR\()(\d+)")[0].astype(int)
    ar_rmse = metrics.loc[metrics["family"].eq("AR")].set_index("lag_order")["rmse"]
    metrics["rmse_difference_vs_ar"] = metrics.apply(
        lambda row: (
            row["rmse"] - ar_rmse[row["lag_order"]]
            if row["family"] == "AR-HMM" else 0.0
        ),
        axis=1,
    )
    metrics["arhmm_improvement_over_ar_pct"] = metrics.apply(
        lambda row: (
            100 * (ar_rmse[row["lag_order"]] - row["rmse"]) / ar_rmse[row["lag_order"]]
            if row["family"] == "AR-HMM" else 0.0
        ),
        axis=1,
    )
    metrics = metrics.sort_values(["lag_order", "family"]).reset_index(drop=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    return metrics


def run_yoy_lag_order_experiment(
    dataset,
    output_dir,
    *,
    lag_orders=range(1, 7),
    test_start="2017-01-01",
    test_end="2023-12-01",
):
    """Compare naive, AR(p), and two-state AR-HMM(p) YoY forecasts."""
    lag_orders = tuple(sorted(set(lag_orders)))
    if not lag_orders or lag_orders[0] < 1:
        raise ValueError("lag_orders must contain positive integers.")

    data, _, _ = load_macro_benchmark_snapshot(Path(dataset))
    sample = data[["Date", "CPIAUCSL"]].copy()
    sample["Inflation_YoY"] = sample["CPIAUCSL"].pct_change(12, fill_method=None) * 100
    lag_columns = [f"Inflation_YoY_Lag{lag}" for lag in range(1, max(lag_orders) + 1)]
    for lag, column in enumerate(lag_columns, start=1):
        sample[column] = sample["Inflation_YoY"].shift(lag)
    sample = sample.dropna().loc[lambda x: x["Date"].le(test_end)].copy()
    sample["TargetAssumedAvailableBy"] = sample["Date"] + pd.offsets.MonthEnd(2)

    dates = sample.loc[sample["Date"].between(test_start, test_end), "Date"]
    if not pd.DatetimeIndex(dates).equals(pd.date_range(test_start, test_end, freq="MS")):
        raise ValueError("Incomplete evaluation period.")
    schedule = pd.DataFrame({
        "TargetMonth": dates,
        "ForecastOrigin": dates + pd.offsets.MonthEnd(0),
    })
    test_rows = sample.set_index("Date").loc[dates]
    naive_name = "Naive YoY(t-1)"
    naive_forecasts = pd.DataFrame({
        "Date": dates.to_numpy(),
        "Actual": test_rows["Inflation_YoY"].to_numpy(),
        "Forecast": test_rows["Inflation_YoY_Lag1"].to_numpy(),
    })
    naive_forecasts["Error"] = naive_forecasts["Actual"] - naive_forecasts["Forecast"]
    results = {
        naive_name: ForecastResult(
            naive_name,
            naive_forecasts,
            {"features": ["Inflation_YoY_Lag1"], "estimator": "last observed YoY"},
        )
    }

    for order in lag_orders:
        features = lag_columns[:order]
        ar_name = f"YoY AR({order})"
        arhmm_name = f"YoY AR-HMM(2 states, p={order})"
        results[ar_name] = fit_expanding_macro(
            sample,
            schedule,
            features=features,
            name=ar_name,
            target="Inflation_YoY",
        )
        print(f"Fitting {arhmm_name}", flush=True)
        results[arhmm_name] = fit_expanding_ar_hmm(
            sample,
            schedule,
            features=features,
            name=arhmm_name,
            target="Inflation_YoY",
            n_states=2,
        )

    output_dir = Path(output_dir)
    metrics = save_result_bundle(results, output_dir, metadata={
        "experiment": "yoy_ar_vs_two_state_arhmm_lag_order",
        "target": "Inflation_YoY",
        "target_definition": "100 * (CPI_t / CPI_t-12 - 1)",
        "lag_orders": list(lag_orders),
        "n_states": 2,
        "test_start": test_start,
        "test_end": test_end,
        "common_sample_max_lag": max(lag_orders),
        "timing": "Existing expanding month-end convention",
    })
    metrics["family"] = metrics["model"].str.extract(r"(AR-HMM|AR)")
    metrics["lag_order"] = pd.to_numeric(
        metrics["model"].str.extract(r"(?:p=|AR\()(\d+)")[0], errors="coerce"
    ).astype("Int64")
    ar_rmse = metrics.loc[metrics["family"].eq("AR")].set_index("lag_order")["rmse"]
    metrics["rmse_difference_vs_ar"] = metrics.apply(
        lambda row: (
            row["rmse"] - ar_rmse[row["lag_order"]]
            if row["family"] == "AR-HMM" else 0.0
        ),
        axis=1,
    )
    metrics["arhmm_improvement_over_ar_pct"] = metrics.apply(
        lambda row: (
            100 * (ar_rmse[row["lag_order"]] - row["rmse"]) / ar_rmse[row["lag_order"]]
            if row["family"] == "AR-HMM" else 0.0
        ),
        axis=1,
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    return metrics.sort_values("rmse").reset_index(drop=True)


def run_regime_experiment(dataset, output_dir, *, test_start="2017-01-01",
                          test_end="2023-12-01", n_states=4):
    """Change only state count, holding the expanding lag-1 sample fixed."""
    if n_states != 4:
        raise ValueError("This four-regime experiment requires n_states=4.")
    data, _, _ = load_macro_benchmark_snapshot(Path(dataset))
    required = ["Inflation_MoM", *MACRO_BENCHMARK_FEATURES]
    sample = data.loc[data[required].notna().all(axis=1) & data.Date.le(test_end)].copy()
    sample["TargetAssumedAvailableBy"] = sample.Date + pd.offsets.MonthEnd(2)
    dates = sample.loc[sample.Date.between(test_start, test_end), "Date"]
    if not pd.DatetimeIndex(dates).equals(pd.date_range(test_start, test_end, freq="MS")):
        raise ValueError("Incomplete evaluation period.")
    schedule = pd.DataFrame({"TargetMonth": dates,
                             "ForecastOrigin": dates + pd.offsets.MonthEnd(0)})
    results = {}
    ar = fit_expanding_macro(sample, schedule, features=["Inflation_Lag1"], name="AR(1)")
    results[ar.name] = ar
    for count in dict.fromkeys([2, n_states]):
        name = f"AR-HMM ({count} states)"
        print(f"Fitting {name} over {len(dates)} months", flush=True)
        results[name] = fit_expanding_ar_hmm(
            sample, schedule, features=["Inflation_Lag1"], name=name, n_states=count,
        )
    output_dir = Path(output_dir)
    metrics = save_result_bundle(results, output_dir, metadata={
        "experiment": "controlled_arhmm_state_count", "n_states": n_states,
        "features": ["Inflation_Lag1"], "test_start": test_start, "test_end": test_end,
        "timing": "Existing expanding lag-1 month-end convention; no forecast date shifting",
        "labels": "Crisis denotes highest conditional inflation, not an independently validated crisis",
    })
    regime_result = results[f"AR-HMM ({n_states} states)"]
    regime_result.forecasts.to_csv(output_dir / "regimes.csv", index=False)
    fig = plot_predicted_regimes(regime_result.forecasts, regime_result.parameters["regime_labels"])
    fig.savefig(output_dir / "predicted_regimes.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    forecasts = pd.read_csv(output_dir / "forecasts.csv", parse_dates=["Date"])
    ax = forecasts.set_index("Date").plot(figsize=(13, 4), ylabel="Monthly inflation (%)",
                                         title="Same inputs and dates: AR(1), two-state and four-state AR-HMM")
    ax.grid(alpha=.2)
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "forecast_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(ax.figure)
    return metrics


def plot_predicted_regimes(frame, labels=("Low", "Moderate", "High", "Crisis")):
    """Plot pre-observation probabilities, never full-sample smoothed regimes."""
    colors = ["#2878b5", "#23956f", "#e6a128", "#c64747"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(frame.Date, frame.Actual, color="#172b4d", label="Actual", lw=1.5)
    axes[0].plot(frame.Date, frame.Forecast, color="#777777", label="Forecast", lw=1.2)
    for label, color in zip(labels, colors):
        selected = frame.PredictedRegime.eq(label)
        axes[0].scatter(frame.loc[selected, "Date"], frame.loc[selected, "Forecast"],
                        color=color, label=label, s=24, zorder=4)
    axes[0].set(ylabel="Monthly inflation (%)", title="Four-state AR-HMM: predicted regime at each target month")
    axes[0].legend(ncol=6, fontsize=9)
    axes[1].stackplot(frame.Date, *[frame[f"Probability_{label}"] for label in labels],
                      labels=labels, colors=colors, alpha=.85)
    axes[1].set(ylabel="Predicted probability", ylim=(0, 1), xlabel="Target month")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(run_regime_experiment(
        root / "data/processed/modelling_datasets/us_nyt_monthly_experiment.csv",
        root / "data/processed/model_results/arhmm_four_states",
    ).to_string(index=False))
