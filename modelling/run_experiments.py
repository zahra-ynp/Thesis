from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_pipeline import add_lags
from evaluation import compare_results
from models import fit_ar1, fit_ar_hmm, fit_arx_sentiment, fit_hmm, fit_lasso
from sentiment import add_sentiment_lags


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    PROJECT_ROOT / "data" / "processed" / "modelling_datasets"  / "us_nyt_monthly_experiment.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "model_results"


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


def save_results(results: dict, output_dir: Path) -> None:
    """Save standardized forecasts, parameters, and one comparison table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, result in results.items():
        result.forecasts.to_csv(output_dir / f"{key}_forecasts.csv", index=False)
        with (output_dir / f"{key}_parameters.json").open("w", encoding="utf-8") as file:
            json.dump(result.parameters, file, indent=2)
    compare_results(results).to_csv(output_dir / "comparison_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["all", "ar1", "arx", "hmm", "arhmm", "arhmm_sentiment", "lasso", "lasso_sentiment"],
        default="all",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-start", default="2017-01-01")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs and show the intended run without fitting models.",
    )
    args = parser.parse_args()

    data = prepare_model_frame(args.data)
    if args.check_only:
        print(f"Dataset ready: {len(data)} monthly rows from {data['Date'].min().date()} to {data['Date'].max().date()}")
        print(f"Selected existing model set: {args.model}")
        return
    results = run_selected(data, args.model, args.test_start)
    save_results(results, args.output)
    print(compare_results(results).to_string(index=False))


if __name__ == "__main__":
    main()
