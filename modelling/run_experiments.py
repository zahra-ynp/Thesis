from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from itertools import combinations


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Direct execution starts Python in modelling/, not in the repository root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from modelling.data_pipeline import add_lags, export_macro_benchmark_contract, load_macro_benchmark_snapshot, MACRO_BENCHMARK_FEATURES
from modelling.evaluation import compare_results
from modelling.models import ForecastResult, fit_ar1, fit_ar_hmm, fit_arx_sentiment, fit_hmm, fit_lasso, fit_expanding_macro, fit_expanding_ar_hmm
from modelling.sentiment import (
    ALTERNATIVE_FINBERT_FEATURES,
    add_sentiment_lags,
)


DEFAULT_DATASET = (
    PROJECT_ROOT / "data" / "processed" / "modelling_datasets"  / "us_nyt_monthly_experiment.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "model_results" / "baseline"
DEFAULT_CONTRACT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "modelling_datasets" / "macro_benchmark"

SENTIMENT_AGGREGATION_LABELS = {
    "News_Sentiment": "mean sentiment (baseline)",
    "Sentiment_Median": "median sentiment",
    "Negative_Share": "negative share",
    "Sentiment_Std": "sentiment dispersion",
}


def prepare_model_frame(path: Path) -> pd.DataFrame:
    """Load the common experiment dataset and create existing six-lag inputs."""
    data = pd.read_csv(path, parse_dates=["Date"])
    data, _ = add_lags(data, ["Inflation_MoM"], lags=6)
    data, _ = add_sentiment_lags(data, lags=6)
    return data


def run_selected(data: pd.DataFrame, selected: str, test_start: str) -> dict:
    """Run one existing model family or the complete current comparison."""
    inflation_lags = [f"Inflation_Lag{lag}" for lag in range(1, 7)]
    sentiment_lags = [f"Sentiment_Lag{lag}" for lag in range(1, 7)]
    runners = {
        "ar1": lambda: fit_ar1(data, test_start=test_start),
        "arx": lambda: fit_arx_sentiment(data, test_start=test_start),
        "hmm": lambda: fit_hmm(data, test_start=test_start),
        "arhmm": lambda: fit_ar_hmm(
            data, features=["Inflation_Lag1"], name="AR-HMM", test_start=test_start
        ),
        "arhmm_sentiment": lambda: fit_ar_hmm(
            data,
            features=["Inflation_Lag1", "Sentiment_Lag1"],
            name="AR-HMM + sentiment",
            test_start=test_start,
            maxiter=2500,
            search_reps=40,
        ),
        "lasso": lambda: fit_lasso(
            data, features=inflation_lags, name="LASSO inflation only", test_start=test_start
        ),
        "lasso_sentiment": lambda: fit_lasso(
            data,
            features=[*inflation_lags, *sentiment_lags],
            name="LASSO inflation + sentiment",
            test_start=test_start,
        ),
    }
    names = list(runners) if selected == "all" else [selected]
    return {name: runners[name]() for name in names}


def save_results(results: dict, output_dir: Path, *, metadata=None) -> None:
    """Save one experiment bundle; preserve other models when re-running a subset."""
    from modelling.evaluation import save_result_bundle
    combined = {}
    previous_metadata = {}
    if (output_dir / "forecasts.csv").exists():
        old = pd.read_csv(output_dir / "forecasts.csv", parse_dates=["Date"])
        saved = json.loads((output_dir / "parameters.json").read_text(encoding="utf-8"))
        previous_metadata = saved.get("metadata", {})
        for name, parameters in saved["models"].items():
            combined[name] = ForecastResult(name, old[["Date", "Actual", name]].rename(columns={name: "Forecast"}), parameters)
    combined.update({result.name: result for result in results.values()})
    save_result_bundle(combined, output_dir, metadata={**previous_metadata, **(metadata or {})})


def run_macro_benchmark(path: Path = DEFAULT_DATASET, *, test_start="2017-01-01", test_end="2023-12-01"):
    """Run all sixteen combinations using t-1 inputs and expanding estimation."""
    data, metadata, _ = load_macro_benchmark_snapshot(path)
    features = list(MACRO_BENCHMARK_FEATURES)
    sample = data[["Date", "Inflation_MoM", *features]].dropna().copy()
    sample = sample.loc[sample["Date"] <= pd.Timestamp(test_end)]
    sample["TargetAssumedAvailableBy"] = sample["Date"] + pd.offsets.MonthEnd(2)
    dates = sample.loc[sample["Date"].between(test_start, test_end), "Date"]
    if not pd.DatetimeIndex(dates).equals(pd.date_range(test_start, test_end, freq="MS")):
        raise ValueError("The requested evaluation period is incomplete.")
    schedule = pd.DataFrame({"TargetMonth": dates, "ForecastOrigin": dates + pd.offsets.MonthEnd(0)})
    training = sample.loc[sample["Date"] < pd.Timestamp(test_start)]
    if training.empty:
        raise ValueError("The initial training sample is empty.")
    contract = {
        "experiment": "us_macro_lag1_expanding",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset": Path(path).name,
        "region": metadata["region_code"], "news_source": metadata["news_source"]["source_code"],
        "target": "Inflation_MoM", "baseline_features": features[:1], "macro_features": features[1:],
        "timing": {"predictor_reference_month": "t-1", "calendar_lag": 1,
                   "interpretation": "Historical conditional one-step forecast with t-1 observations assumed available.",
                   "calendar_origin_convention": "End of target month under the assumed publication delay.",
                   "training_target_cutoff": "t-1", "real_time_backtest": False,
                   "vintage_policy": "Saved revised historical data; actual release dates not reconstructed."},
        "sample": {"initial_train_rows": len(training), "initial_train_start": str(training["Date"].min().date()),
                   "initial_train_end": str(training["Date"].max().date()),
                   "test_start": test_start, "test_end": test_end, "test_rows": len(dates)},
    }
    labels = ["Oil", "Unemployment", "Industry", "Fed rate"]
    columns = contract["macro_features"]
    results = {}
    for size in range(5):
        for subset in combinations(range(4), size):
            name = "Inflation only (lag 1)" if not subset else "Inflation + " + " + ".join(labels[i] for i in subset)
            results[name] = fit_expanding_macro(
                sample, schedule, features=["Inflation_Lag1", *[columns[i] for i in subset]], name=name
            )
    return results, contract


def compare_macro_lags(path: Path = DEFAULT_DATASET):
    """Compare lag 1 and lag 2 with identical expanding estimation and target dates.

    Lag 1 uses a later information set (target-month end under the same release
    assumption). It is a historical conditional comparison, not an earlier-origin
    real-time forecast. Existing experiment bundles are not overwritten.
    """
    raw = pd.read_csv(path, parse_dates=["Date"])
    results, descriptions = {}, {}
    for lag in (1, 2):
        frame = raw[["Date", "Inflation_MoM", "Unemployment_Change"]].copy()
        features = [f"Inflation_Lag{lag}", f"Unemployment_Change_Lag{lag}"]
        frame[features[0]] = frame["Inflation_MoM"].shift(lag)
        frame[features[1]] = frame["Unemployment_Change"].shift(lag)
        frame = frame.dropna().reset_index(drop=True)
        frame["TargetAssumedAvailableBy"] = frame["Date"] + pd.offsets.MonthEnd(2)
        dates = frame.loc[frame["Date"].between("2017-01-01", "2023-12-01"), "Date"]
        origins = dates + pd.offsets.MonthEnd(0) if lag == 1 else dates - pd.Timedelta(days=1)
        schedule = pd.DataFrame({"TargetMonth": dates, "ForecastOrigin": origins})
        for use_unemployment in (False, True):
            name = f"Lag {lag}: inflation" + (" + unemployment" if use_unemployment else " only")
            results[name] = fit_expanding_macro(
                frame, schedule, features=features if use_unemployment else features[:1], name=name,
            )
            descriptions[name] = {
                "lag": lag,
                "information_set": "t-1 available (later origin)" if lag == 1 else "t-2 available (before target month)",
            }
    table = compare_results(results)
    for column in ("lag", "information_set"):
        table[column] = table["model"].map(lambda name: descriptions[name][column])
    table["test_months"] = 84
    return table, results


def run_sentiment_comparison(path: Path = DEFAULT_DATASET, *, macro_dir: Path | None = None):
    """Compare models with/without target-month FinBERT sentiment."""
    macro_dir = Path(macro_dir) if macro_dir is not None else DEFAULT_OUTPUT.parent / "macro"
    ranking = pd.read_csv(macro_dir / "metrics.csv").sort_values("rmse")
    saved = json.loads((macro_dir / "parameters.json").read_text(encoding="utf-8"))
    best_name = ranking.loc[ranking["macro_count"] > 0, "model"].iloc[0]
    best_features = saved["models"][best_name]["features"]
    first = saved["metadata"]["sample"]["test_start"]
    last = saved["metadata"]["sample"]["test_end"]
    data, _, _ = load_macro_benchmark_snapshot(path)
    data["Sentiment_Current"] = data["News_Sentiment"]
    common = ["Inflation_MoM", *MACRO_BENCHMARK_FEATURES]
    sample = data.loc[data[common].notna().all(axis=1) & data["Date"].le(last)].copy()
    if sample["Sentiment_Current"].isna().any():
        raise ValueError("Sentiment is missing on the shared macro sample.")
    sample["TargetAssumedAvailableBy"] = sample["Date"] + pd.offsets.MonthEnd(2)
    dates = sample.loc[sample["Date"].between(first, last), "Date"]
    if not pd.DatetimeIndex(dates).equals(pd.date_range(first, last, freq="MS")):
        raise ValueError("The sentiment comparison does not cover the full macro evaluation period.")
    schedule = pd.DataFrame({"TargetMonth": dates, "ForecastOrigin": dates + pd.offsets.MonthEnd(0)})
    specifications = {
        "Inflation only (lag 1)": ["Inflation_Lag1"],
        "Inflation + sentiment": ["Inflation_Lag1", "Sentiment_Current"],
        best_name: best_features,
        best_name + " + sentiment": [*best_features, "Sentiment_Current"],
    }
    results = {name: fit_expanding_macro(sample, schedule, features=features, name=name)
               for name, features in specifications.items()}
    forecasts = None
    for name, result in results.items():
        part = result.forecasts[["Date", "Actual", "Forecast"]].rename(columns={"Forecast": name})
        if forecasts is None:
            forecasts = part
        else:
            forecasts = forecasts.merge(part.drop(columns="Actual"), on="Date", validate="one_to_one")
    return forecasts, compare_results(results), best_name


def run_sentiment_aggregation_comparison(
    path: Path = DEFAULT_DATASET,
    *,
    test_start="2017-01-01",
    test_end="2023-12-01",
):
    """Compare FinBERT monthly aggregations in AR and AR-HMM models.

    Every sentiment representation is tested separately with inflation lag 1.
    Generic mean FinBERT sentiment is the sentiment baseline, and an
    inflation-only specification is retained as the forecasting benchmark.
    """
    data, metadata, _ = load_macro_benchmark_snapshot(path)
    sentiment_features = list(ALTERNATIVE_FINBERT_FEATURES)
    missing = set(sentiment_features) - set(data.columns)
    if missing:
        raise ValueError(
            "The modelling dataset is missing sentiment aggregations: "
            f"{sorted(missing)}"
        )

    required = ["Inflation_MoM", "Inflation_Lag1", *sentiment_features]
    sample = data.loc[data[required].notna().all(axis=1) & data["Date"].le(test_end)].copy()
    sample["TargetAssumedAvailableBy"] = sample["Date"] + pd.offsets.MonthEnd(2)
    dates = sample.loc[sample["Date"].between(test_start, test_end), "Date"]
    expected_dates = pd.date_range(test_start, test_end, freq="MS")
    if not pd.DatetimeIndex(dates).equals(expected_dates):
        raise ValueError("The sentiment aggregation evaluation period is incomplete.")
    schedule = pd.DataFrame({
        "TargetMonth": dates,
        "ForecastOrigin": dates + pd.offsets.MonthEnd(0),
    })

    ar_specifications = {"Inflation only (lag 1)": ["Inflation_Lag1"]}
    arhmm_specifications = {"AR-HMM inflation only (lag 1)": ["Inflation_Lag1"]}
    for feature, label in SENTIMENT_AGGREGATION_LABELS.items():
        ar_specifications[f"Inflation + {label}"] = ["Inflation_Lag1", feature]
        arhmm_specifications[f"AR-HMM + {label}"] = ["Inflation_Lag1", feature]

    ar_results = {
        name: fit_expanding_macro(sample, schedule, features=features, name=name)
        for name, features in ar_specifications.items()
    }
    arhmm_results = {}
    for name, features in arhmm_specifications.items():
        arhmm_results[name] = fit_expanding_ar_hmm(
            sample, schedule, features=features, name=name
        )
        print(f"Completed {name}")

    metadata_common = {
        "dataset": Path(path).name,
        "region": metadata["region_code"],
        "news_source": metadata["news_source"]["source_code"],
        "target": "Inflation_MoM",
        "test_start": test_start,
        "test_end": test_end,
        "test_rows": len(dates),
        "predictor_timing": "month t sentiment observed by month end is used for month t inflation",
        "sentiment_lag_months": 0,
        "sentiment_features": SENTIMENT_AGGREGATION_LABELS,
        "sentiment_baseline": "News_Sentiment (monthly mean)",
        "comparison_policy": "one sentiment aggregation per model",
    }
    return (
        ar_results,
        arhmm_results,
        {**metadata_common, "experiment": "us_ar_sentiment_aggregations"},
        {**metadata_common, "experiment": "us_arhmm_sentiment_aggregations"},
    )


def run_expanding_arhmm_benchmark(
    path: Path = DEFAULT_DATASET,
    *,
    test_start="2017-01-01",
    test_end="2023-12-01",
):
    """Run expanding-window AR-HMM macro and sentiment comparisons."""
    data, metadata, _ = load_macro_benchmark_snapshot(path)
    benchmark_features = list(MACRO_BENCHMARK_FEATURES)
    required = ["Inflation_MoM", *benchmark_features]
    sample = data.loc[
        data[required].notna().all(axis=1) & data["Date"].le(test_end)
    ].copy()
    sample["TargetAssumedAvailableBy"] = sample["Date"] + pd.offsets.MonthEnd(2)
    dates = sample.loc[sample["Date"].between(test_start, test_end), "Date"]
    expected_dates = pd.date_range(test_start, test_end, freq="MS")
    if not pd.DatetimeIndex(dates).equals(expected_dates):
        raise ValueError("The expanding AR-HMM evaluation period is incomplete.")
    schedule = pd.DataFrame({
        "TargetMonth": dates,
        "ForecastOrigin": dates + pd.offsets.MonthEnd(0),
    })
    initial_train = sample.loc[sample["Date"] < pd.Timestamp(test_start)]
    if initial_train.empty:
        raise ValueError("The initial expanding AR-HMM training sample is empty.")

    labels = ["Oil", "Unemployment", "Industry", "Fed rate"]
    macro_columns = benchmark_features[1:]
    baseline_name = "AR-HMM inflation only (lag 1)"
    feature_map = {}
    macro_results = {}
    for size in range(5):
        for subset in combinations(range(4), size):
            name = (
                baseline_name
                if not subset
                else "AR-HMM + " + " + ".join(labels[index] for index in subset)
            )
            features = [
                "Inflation_Lag1",
                *[macro_columns[index] for index in subset],
            ]
            feature_map[name] = features
            macro_results[name] = fit_expanding_ar_hmm(
                sample, schedule, features=features, name=name
            )
            print(f"Completed {name}")

    macro_metrics = compare_results(macro_results)
    macro_metrics["macro_count"] = macro_metrics["model"].map(
        lambda name: len(feature_map[name]) - 1
    )
    best_macro_name = macro_metrics.loc[
        macro_metrics["macro_count"] > 0, "model"
    ].iloc[0]

    sentiment_sample = sample.copy()
    sentiment_sample["Sentiment_Current"] = sentiment_sample["News_Sentiment"]
    if sentiment_sample["Sentiment_Current"].isna().any():
        raise ValueError("Sentiment is missing on the expanding AR-HMM sample.")
    sentiment_name = "AR-HMM inflation + sentiment"
    best_sentiment_name = best_macro_name + " + sentiment"
    sentiment_results = {
        baseline_name: macro_results[baseline_name],
        sentiment_name: fit_expanding_ar_hmm(
            sentiment_sample,
            schedule,
            features=["Inflation_Lag1", "Sentiment_Current"],
            name=sentiment_name,
        ),
        best_macro_name: macro_results[best_macro_name],
        best_sentiment_name: fit_expanding_ar_hmm(
            sentiment_sample,
            schedule,
            features=[*feature_map[best_macro_name], "Sentiment_Current"],
            name=best_sentiment_name,
        ),
    }
    metadata_common = {
        "dataset": Path(path).name,
        "region": metadata["region_code"],
        "news_source": metadata["news_source"]["source_code"],
        "target": "Inflation_MoM",
        "test_start": test_start,
        "test_end": test_end,
        "test_rows": len(dates),
        "initial_train_start": str(initial_train["Date"].min().date()),
        "initial_train_end": str(initial_train["Date"].max().date()),
        "initial_train_rows": len(initial_train),
        "predictor_lag": 1,
        "estimator": "expanding-window two-state AR-HMM",
    }
    macro_metadata = {
        **metadata_common,
        "experiment": "us_arhmm_macro_lag1_expanding",
        "baseline_features": ["Inflation_Lag1"],
        "macro_features": macro_columns,
    }
    sentiment_metadata = {
        **metadata_common,
        "experiment": "us_arhmm_sentiment_expanding",
        "best_macro_model": best_macro_name,
        "sentiment_feature": "Sentiment_Current",
    }
    return macro_results, sentiment_results, macro_metadata, sentiment_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["all", "ar1", "arx", "hmm", "arhmm", "arhmm_sentiment", "lasso", "lasso_sentiment"],
        default="all",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--macro", action="store_true", help="Run all 16 macro combinations and save one result bundle.")
    parser.add_argument(
        "--arhmm-expanding",
        action="store_true",
        help="Run expanding AR-HMM macro and sentiment comparisons and save both bundles.",
    )
    parser.add_argument(
        "--sentiment-aggregations",
        action="store_true",
        help="Compare mean, median, negative share, and dispersion in AR and AR-HMM.",
    )
    parser.add_argument("--test-start", default="2017-01-01")
    parser.add_argument("--test-end", default="2023-12-01", help="End of Part A evaluation sample.")
    parser.add_argument("--prepare-macro-contract", action="store_true", help="Audit and export Part A only; fits no models.")
    parser.add_argument("--contract-output", type=Path, default=DEFAULT_CONTRACT_OUTPUT)
    parser.add_argument("--timing-policy", choices=["month_end_next_month_forecast"], default="month_end_next_month_forecast",
                        help="Part A calendar convention: next-month forecast with reference-month lag 2.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs and show the intended run without fitting models.",
    )
    args = parser.parse_args()

    if args.sentiment_aggregations:
        if args.macro or args.arhmm_expanding or args.prepare_macro_contract or args.check_only:
            parser.error("Use --sentiment-aggregations on its own.")
        from modelling.evaluation import save_result_bundle
        ar_results, arhmm_results, ar_metadata, arhmm_metadata = (
            run_sentiment_aggregation_comparison(
                args.data, test_start=args.test_start, test_end=args.test_end
            )
        )
        destination = args.output or DEFAULT_OUTPUT.parent
        ar_metrics = save_result_bundle(
            ar_results, destination / "ar_sentiment_aggregations", metadata=ar_metadata
        )
        arhmm_metrics = save_result_bundle(
            arhmm_results,
            destination / "arhmm_sentiment_aggregations",
            metadata=arhmm_metadata,
        )
        print("\nAR sentiment aggregation comparison:\n")
        print(ar_metrics.to_string(index=False))
        print("\nAR-HMM sentiment aggregation comparison:\n")
        print(arhmm_metrics.to_string(index=False))
        return

    if args.arhmm_expanding:
        if args.macro or args.prepare_macro_contract or args.check_only:
            parser.error("Use --arhmm-expanding on its own to generate both AR-HMM bundles.")
        from modelling.evaluation import save_result_bundle
        macro_results, sentiment_results, macro_metadata, sentiment_metadata = (
            run_expanding_arhmm_benchmark(
                args.data, test_start=args.test_start, test_end=args.test_end
            )
        )
        destination = args.output or DEFAULT_OUTPUT.parent
        macro_metrics = save_result_bundle(
            macro_results, destination / "arhmm_macro", metadata=macro_metadata
        )
        sentiment_metrics = save_result_bundle(
            sentiment_results,
            destination / "arhmm_sentiment",
            metadata=sentiment_metadata,
        )
        print("\nExpanding AR-HMM macro comparison:\n")
        print(macro_metrics.to_string(index=False))
        print("\nExpanding AR-HMM sentiment comparison:\n")
        print(sentiment_metrics.to_string(index=False))
        return

    if args.macro:
        from modelling.evaluation import save_result_bundle
        if args.prepare_macro_contract or args.check_only:
            parser.error("Use --macro on its own to generate the forecast comparison.")
        results, contract = run_macro_benchmark(args.data, test_start=args.test_start, test_end=args.test_end)
        destination = args.output or DEFAULT_OUTPUT.parent / "macro"
        metrics = save_result_bundle(results, destination, metadata=contract)
        print(metrics[["model", "rmse", "mae", "rmse_improvement_pct"]].to_string(index=False))
        return

    if args.prepare_macro_contract:
        if args.check_only:
            parser.error("Use --prepare-macro-contract by itself; it audits and exports without fitting models.")
        paths = export_macro_benchmark_contract(
            args.data, args.contract_output, test_start=args.test_start, test_end=args.test_end,
            timing_policy=args.timing_policy,
        )
        contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
        print(json.dumps({"status": contract["status"], "sample": contract["sample"],
                          "outputs": {key: str(value) for key, value in paths.items()}}, indent=2))
        return

    data = prepare_model_frame(args.data)
    if args.check_only:
        print(f"Dataset ready: {len(data)} monthly rows from {data['Date'].min().date()} to {data['Date'].max().date()}")
        print(f"Selected existing model set: {args.model}")
        return
    results = run_selected(data, args.model, args.test_start)
    save_results(results, args.output or DEFAULT_OUTPUT, metadata={
        "dataset": str(args.data), "test_start": args.test_start,
        "last_rerun_models": [result.name for result in results.values()],
        "timing": "Original static-fit lag-1 experiment; compare separately from the macro benchmark.",
    })
    print(compare_results(results).to_string(index=False))


if __name__ == "__main__":
    main()
