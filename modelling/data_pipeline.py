from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT_NAME = "inflation_sentiment_forecasting"
REQUIRED_MACRO_COLUMNS = (
    "CPIAUCSL",
    "UNRATE",
    "FEDFUNDS",
    "INDPRO",
    "MCOILWTICO",
)


def load_source_registry(engine) -> pd.DataFrame:
    """Return available news sources and their database coverage."""
    from sqlalchemy import text

    query = text("""
        SELECT
            s.source_code,
            s.source_name,
            CAST(s.is_active AS INT) AS is_active,
            COUNT(n.article_id) AS article_count,
            MIN(n.news_date) AS first_news_date,
            MAX(n.news_date) AS last_news_date
        FROM core.news_sources s
        LEFT JOIN core.daily_news n ON n.source_id = s.source_id
        GROUP BY s.source_code, s.source_name, s.is_active
        ORDER BY s.source_code
    """)
    return pd.read_sql(query, engine, parse_dates=["first_news_date", "last_news_date"])


def load_monthly_macro_from_database(
    engine,
    *,
    region_code: str = "US",
) -> pd.DataFrame:
    """Load the current experiment macro series and reshape to one row per month."""
    from sqlalchemy import text

    query = text("""
        SELECT m.month_date, i.indicator_code, m.indicator_value
        FROM core.monthly_macro_indicators m
        INNER JOIN core.source_agencies a ON m.agency_id = a.agency_id
        INNER JOIN core.source_regions r ON a.region_id = r.region_id
        INNER JOIN core.indicators i ON m.indicator_id = i.indicator_id
        WHERE r.region_code = :region_code
          AND i.indicator_code IN (
              'CPIAUCSL', 'UNRATE', 'FEDFUNDS', 'INDPRO', 'MCOILWTICO'
          )
        ORDER BY m.month_date, i.indicator_code
    """)
    long_frame = pd.read_sql(
        query,
        engine,
        params={"region_code": region_code},
        parse_dates=["month_date"],
    )
    if long_frame.empty:
        raise ValueError(f"No macroeconomic rows found for region {region_code!r}.")

    wide = (
        long_frame.pivot(
            index="month_date",
            columns="indicator_code",
            values="indicator_value",
        )
        .rename_axis(columns=None)
        .reset_index()
        .rename(columns={"month_date": "Date"})
    )
    return wide


def load_article_sentiment_from_database(
    engine,
    *,
    source_code: str,
    region_code: str = "US",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load article scores for exactly one named source, regardless of active status."""
    from sqlalchemy import text

    source_query = text("""
        SELECT source_code, source_name, CAST(is_active AS INT) AS is_active
        FROM core.news_sources
        WHERE source_code = :source_code
    """)
    source = pd.read_sql(source_query, engine, params={"source_code": source_code})
    if len(source) != 1:
        raise ValueError(f"Expected one registered source for {source_code!r}; found {len(source)}.")

    news_query = text("""
        SELECT
            n.article_id,
            n.external_article_id,
            n.news_date,
            n.sentiment_score,
            n.sentiment_label
        FROM core.daily_news n
        INNER JOIN core.news_sources s ON n.source_id = s.source_id
        INNER JOIN core.source_regions r ON n.region_id = r.region_id
        WHERE s.source_code = :source_code
          AND r.region_code = :region_code
        ORDER BY n.news_date, n.article_id
    """)
    articles = pd.read_sql(
        news_query,
        engine,
        params={"source_code": source_code, "region_code": region_code},
        parse_dates=["news_date"],
    )
    if articles.empty:
        raise ValueError(
            f"No news rows found for source {source_code!r} and region {region_code!r}."
        )

    source_record = source.iloc[0].to_dict()
    source_record["is_active"] = bool(source_record["is_active"])
    return articles, source_record


def prepare_experiment_frame(
    macro: pd.DataFrame,
    monthly_news: pd.DataFrame,
    *,
    source_code: str,
) -> pd.DataFrame:
    """Create the common monthly frame and leakage-safe one-month lags."""
    if "Date" not in macro or "Date" not in monthly_news:
        raise ValueError("Both macro and monthly news data must contain a Date column.")

    missing_macro = set(REQUIRED_MACRO_COLUMNS) - set(macro.columns)
    if missing_macro:
        raise ValueError(f"Macro data are missing required series: {sorted(missing_macro)}")

    macro_frame = macro.copy()
    news_frame = monthly_news.copy()
    macro_frame["Date"] = pd.to_datetime(macro_frame["Date"]).dt.to_period("M").dt.to_timestamp()
    news_frame["Date"] = pd.to_datetime(news_frame["Date"]).dt.to_period("M").dt.to_timestamp()
    macro_frame = macro_frame.sort_values("Date").drop_duplicates("Date", keep="last")
    news_frame = news_frame.sort_values("Date").drop_duplicates("Date", keep="last")

    for column in REQUIRED_MACRO_COLUMNS:
        macro_frame[column] = pd.to_numeric(macro_frame[column], errors="coerce")

    data = macro_frame.merge(news_frame, on="Date", how="left", validate="one_to_one")
    data.insert(1, "Experiment_News_Source", source_code.upper())

    data["Inflation_MoM"] = data["CPIAUCSL"].pct_change(1, fill_method=None) * 100
    data["Inflation_YoY"] = data["CPIAUCSL"].pct_change(12, fill_method=None) * 100
    data["Unemployment_Change"] = data["UNRATE"].diff()
    data["Fed_Funds_Change"] = data["FEDFUNDS"].diff()
    data["IP_Growth_MoM"] = data["INDPRO"].pct_change(1, fill_method=None) * 100
    positive_oil = data["MCOILWTICO"].where(data["MCOILWTICO"] > 0)
    data["Oil_Price_Growth"] = np.log(positive_oil).diff() * 100

    lag_columns = {
        "Inflation_MoM": "Inflation_Lag1",
        "News_Sentiment": "News_Sentiment_Lag1",
        "Unemployment_Change": "Unemployment_Change_Lag1",
        "Fed_Funds_Change": "Fed_Funds_Change_Lag1",
        "IP_Growth_MoM": "IP_Growth_MoM_Lag1",
        "Oil_Price_Growth": "Oil_Price_Growth_Lag1",
    }
    for original, lagged in lag_columns.items():
        if original in data:
            data[lagged] = data[original].shift(1)

    return data.sort_values("Date").reset_index(drop=True)


def validate_experiment_frame(data: pd.DataFrame) -> dict[str, Any]:
    """Run structural checks and return a compact coverage report."""
    if data.empty:
        raise ValueError("The experiment frame is empty.")
    if data["Date"].duplicated().any():
        raise ValueError("The experiment frame contains duplicate months.")
    if not data["Date"].is_monotonic_increasing:
        raise ValueError("The experiment frame is not ordered chronologically.")

    expected_dates = pd.date_range(data["Date"].min(), data["Date"].max(), freq="MS")
    missing_months = expected_dates.difference(pd.DatetimeIndex(data["Date"]))
    missing_macro = {
        column: int(data[column].isna().sum()) for column in REQUIRED_MACRO_COLUMNS
    }
    if any(missing_macro.values()):
        raise ValueError(f"Macroeconomic coverage is incomplete: {missing_macro}")

    return {
        "row_count": int(len(data)),
        "first_month": str(data["Date"].min().date()),
        "last_month": str(data["Date"].max().date()),
        "missing_calendar_months": [str(value.date()) for value in missing_months],
        "months_with_sentiment": int(data["News_Sentiment"].notna().sum()),
        "months_without_sentiment": int(data["News_Sentiment"].isna().sum()),
        "article_count_total": int(data["Article_Count"].fillna(0).sum()),
        "macro_missing_values": missing_macro,
    }


def build_experiment_metadata(
    data: pd.DataFrame,
    *,
    source_record: dict[str, Any],
    coverage: dict[str, Any],
    test_start: str,
    region_code: str = "US",
) -> dict[str, Any]:
    """Build auditable metadata shared by all forecasting models."""
    test_timestamp = pd.Timestamp(test_start)
    train = data.loc[data["Date"] < test_timestamp]
    test = data.loc[data["Date"] >= test_timestamp]
    if train.empty or test.empty:
        raise ValueError("test_start must leave observations in both train and test periods.")

    return {
        "experiment": EXPERIMENT_NAME,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "region_code": region_code,
        "news_source": {
            "source_code": str(source_record["source_code"]),
            "source_name": str(source_record["source_name"]),
            "was_active_at_export": bool(source_record["is_active"]),
            "selection_rule": "explicit source_code; experiment does not depend on active view",
        },
        "coverage": coverage,
        "target_definitions": {
            "primary": "Inflation_MoM = 100 * monthly percentage change in CPIAUCSL",
            "secondary": "Inflation_YoY = 100 * twelve-month percentage change in CPIAUCSL",
        },
        "predictor_timing": {
            "forecast_horizon_months": 1,
            "policy": "Only lag-1 predictors are used to forecast month t.",
        },
        "split": {
            "test_start": str(test_timestamp.date()),
            "train_first_month": str(train["Date"].min().date()),
            "train_last_month": str(train["Date"].max().date()),
            "test_first_month": str(test["Date"].min().date()),
            "test_last_month": str(test["Date"].max().date()),
        },
        "macro_transformations": {
            "UNRATE": "first difference",
            "FEDFUNDS": "first difference",
            "INDPRO": "monthly percentage growth",
            "MCOILWTICO": "monthly log growth",
        },
    }


def export_experiment_dataset(
    data: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save the common experiment data and its metadata side by side."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_code = metadata["news_source"]["source_code"].lower()
    region_code = metadata["region_code"].lower()
    stem = f"{region_code}_{source_code}_monthly_experiment"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}_metadata.json"
    data.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return csv_path, json_path


def add_lags(
    data: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    lags: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Add consistently named lags for any existing monthly columns."""
    if lags < 1:
        raise ValueError("lags must be at least 1.")
    frame = data.copy()
    created: list[str] = []
    for column in columns:
        if column not in frame:
            raise ValueError(f"Cannot lag missing column {column!r}.")
        base = column.removesuffix("_MoM")
        for lag in range(1, lags + 1):
            name = f"{base}_Lag{lag}"
            frame[name] = frame[column].shift(lag)
            created.append(name)
    return frame, created


MACRO_BENCHMARK_FEATURES = (
    "Inflation_Lag1", "Oil_Price_Growth_Lag1", "Unemployment_Change_Lag1",
    "IP_Growth_MoM_Lag1", "Fed_Funds_Change_Lag1",
)
MACRO_TRANSFORM_COLUMNS = (
    "Inflation_MoM", "Inflation_YoY", "Oil_Price_Growth",
    "Unemployment_Change", "IP_Growth_MoM", "Fed_Funds_Change",
)


def load_macro_benchmark_snapshot(
    path: str | Path,
    metadata_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Audit the saved US snapshot without repairing, filling, or sorting its rows."""
    path = Path(path)
    metadata_path = Path(metadata_path) if metadata_path else path.with_name(
        path.stem + "_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data = pd.read_csv(path)
    required = {
        "Date", "Experiment_News_Source", *REQUIRED_MACRO_COLUMNS,
        *MACRO_TRANSFORM_COLUMNS, *MACRO_BENCHMARK_FEATURES,
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Snapshot is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Snapshot is empty.")
    data["Date"] = pd.to_datetime(data["Date"], errors="raise")
    dates = data["Date"]
    if dates.isna().any() or not dates.eq(dates.dt.to_period("M").dt.to_timestamp()).all():
        raise ValueError("Dates must be nonmissing month-start dates.")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be unique and ordered.")
    expected_dates = pd.date_range(dates.min(), dates.max(), freq="MS")
    if len(expected_dates) != len(data):
        raise ValueError("Snapshot has missing calendar months; row shifts would be invalid.")
    if metadata.get("region_code") != "US":
        raise ValueError("This benchmark contract currently supports the US only.")
    source = metadata["news_source"]["source_code"]
    for column in ("Experiment_News_Source", "News_Source"):
        if column in data and not data[column].eq(source).all():
            raise ValueError(f"{column} does not agree with the named metadata source.")
    coverage = metadata["coverage"]
    observed = {"row_count": len(data), "first_month": str(dates.min().date()),
                "last_month": str(dates.max().date())}
    if any(coverage.get(key) != value for key, value in observed.items()):
        raise ValueError("Snapshot dates or row count disagree with metadata coverage.")
    numeric = [*REQUIRED_MACRO_COLUMNS, *MACRO_TRANSFORM_COLUMNS, *MACRO_BENCHMARK_FEATURES]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if np.isinf(data[column].to_numpy()).any():
            raise ValueError(f"Infinite values in {column}.")
    if data[list(REQUIRED_MACRO_COLUMNS)].isna().any().any():
        raise ValueError("Missing raw macroeconomic values.")
    for column in ("CPIAUCSL", "INDPRO", "MCOILWTICO"):
        if data[column].le(0).any():
            raise ValueError(f"Nonpositive values invalidate the specified transformation: {column}.")

    # Reuse the original transformation implementation, independently of saved derived values.
    recalculated = prepare_experiment_frame(
        data[["Date", *REQUIRED_MACRO_COLUMNS]], data[["Date"]], source_code=source
    )
    checks = []
    for column in (*MACRO_TRANSFORM_COLUMNS, *MACRO_BENCHMARK_FEATURES):
        actual = data[column].to_numpy(dtype=float)
        expected = recalculated[column].to_numpy(dtype=float)
        if not np.allclose(actual, expected, rtol=1e-9, atol=1e-10, equal_nan=True):
            raise ValueError(f"Saved transformation or lag does not reproduce: {column}.")
        finite = np.isfinite(expected)
        checks.append({
            "column": column, "matches_recomputed": True,
            "missing_count": int(np.isnan(actual).sum()),
            "maximum_absolute_difference": float(np.max(np.abs(actual[finite] - expected[finite])))
            if finite.any() else 0.0,
        })
    return data, metadata, checks


def build_macro_benchmark_contract(
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    test_start: str = "2017-01-01",
    test_end: str = "2023-12-01",
    timing_policy: str = "month_end_next_month_forecast",
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Prepare Part A only. No estimator is fitted and no forecasts are generated.

    Forecast month t at the end of t-1, using reference-month t-2 inputs.
    Release availability is an explicit conservative assumption, not a vintage audit.
    """
    if timing_policy != "month_end_next_month_forecast":
        raise ValueError("Supported timing_policy: month_end_next_month_forecast.")
    path = Path(path)
    metadata_path = Path(metadata_path) if metadata_path else path.with_name(
        path.stem + "_metadata.json"
    )
    data, metadata, checks = load_macro_benchmark_snapshot(path, metadata_path)
    first, last = pd.Timestamp(test_start), pd.Timestamp(test_end)
    if any(pd.isna(x) or x != x.to_period("M").to_timestamp() for x in (first, last)) or first > last:
        raise ValueError("Test boundaries must be ordered month-start dates.")
    # Audit the old lag-1 snapshot above, then align a separate benchmark frame.
    # A reference-month t-1 CPI is not yet published at the end of t-1.
    aligned_columns = {
        "Inflation_MoM": "Inflation_Lag2",
        "Oil_Price_Growth": "Oil_Price_Growth_Lag2",
        "Unemployment_Change": "Unemployment_Change_Lag2",
        "IP_Growth_MoM": "IP_Growth_MoM_Lag2",
        "Fed_Funds_Change": "Fed_Funds_Change_Lag2",
    }
    for original, aligned in aligned_columns.items():
        data[aligned] = data[original].shift(2)
    features = list(aligned_columns.values())
    required = ["Inflation_MoM", *features]
    eligible = np.isfinite(data[required]).all(axis=1)
    in_scope = data["Date"].le(last)
    sample = data.loc[eligible & in_scope, ["Date", *required]].copy()
    sample["TargetAssumedAvailableBy"] = sample["Date"] + pd.offsets.MonthEnd(2)
    first_origin = first - pd.Timedelta(days=1)
    training = sample.loc[sample["TargetAssumedAvailableBy"] <= first_origin]
    test = sample.loc[sample["Date"].between(first, last)]
    expected_test = pd.date_range(first, last, freq="MS")
    if training.empty:
        raise ValueError("No complete initial training sample.")
    if not pd.DatetimeIndex(test["Date"]).equals(expected_test):
        missing = expected_test.difference(pd.DatetimeIndex(test["Date"]))
        raise ValueError(f"Evaluation sample is incomplete: {missing.strftime('%Y-%m-%d').tolist()}")
    excluded = []
    for index in data.index[~eligible | ~in_scope]:
        reasons = [f"missing {c}" for c in required if not np.isfinite(data.at[index, c])]
        if not in_scope.at[index]:
            reasons.append("after evaluation end")
        excluded.append({"date": str(data.at[index, "Date"].date()), "reasons": reasons})
    sample["Sample"] = np.select(
        [sample["TargetAssumedAvailableBy"] <= first_origin, sample["Date"] >= first],
        ["initial_train", "evaluation"], default="pretest_target_unreleased_at_first_origin",
    )
    schedule = pd.DataFrame({"TargetMonth": test["Date"].to_numpy()})
    schedule["ForecastOrigin"] = schedule["TargetMonth"] - pd.Timedelta(days=1)
    schedule["OriginConvention"] = "end of day, America/New_York"
    schedule["LatestPredictorMonth"] = schedule["TargetMonth"] - pd.DateOffset(months=2)
    schedule["TrainingTargetEnd"] = schedule["LatestPredictorMonth"]
    schedule["PredictorsAssumedAvailableBy"] = schedule["LatestPredictorMonth"] + pd.offsets.MonthEnd(2)
    schedule["EligibleTrainingRows"] = [
        int(sample["TargetAssumedAvailableBy"].le(origin).sum()) for origin in schedule["ForecastOrigin"]
    ]
    timing = {
        "policy": timing_policy,
        "description": "Forecast month t at the end of t-1, using all five predictors from reference month t-2.",
        "reference_month_step": 2,
        "calendar_target_months_ahead": 1,
        "origin_timezone": "America/New_York",
        "origin_time": "end of day",
        "availability_rule": "Assume each monthly observation is available by the end of the following month; all five predictors use the same conservative two-month reference lag.",
        "training_rule": "Use common complete rows whose TargetAssumedAvailableBy is no later than ForecastOrigin. A pre-test reference month is not an eligible label until its assumed release date.",
        "historical_release_dates_verified": False,
        "availability_status": "assumption; exceptional release delays have not been exhaustively audited",
        "baseline_interpretation": "Direct regression on inflation at calendar lag 2. This is neither the old lag-1 AR(1) nor an AR(2) containing both lags, and is not a recursive two-step AR(1) forecast.",
        "live_use": "Require actual publication timestamps and saved vintages; reject or explicitly revise the information set if a required observation is not released. Do not fill from later releases or partial monthly values.",
    }
    contract = {
        "schema_version": 1, "experiment": "us_macro_benchmark_part_a",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "part_a_complete_under_documented_assumptions",
        "ready_for_part_b": True,
        "models_fitted": False,
        "dataset": {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "metadata_filename": metadata_path.name,
                    "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                    "snapshot_created_at_utc": metadata.get("created_at_utc"),
                    "region": metadata["region_code"], "news_source": metadata["news_source"],
                    "news_used_as_predictor": False},
        "target": "Inflation_MoM", "target_units": "percent monthly CPI change",
        "baseline_features": features[:1], "macro_features": features[1:],
        "planned_subset_count_including_baseline": 16,
        "transformations": {"Inflation_MoM": "100 * CPI monthly percentage change",
                            **metadata["macro_transformations"]},
        "timing": timing,
        "vintage_policy": {
            "classification": "historical pseudo-out-of-sample using the saved, potentially revised data snapshot",
            "vintages_present": False, "real_time_backtest": False,
            "evidence": "Snapshot has one value per series/month; metadata contains no observation-level release or vintage dates.",
            "limitation": "Chronological lagging does not remove revisions made after a historical forecast origin.",
        },
        "sample": {"source_rows": len(data), "common_rows": len(sample),
                   "initial_train_rows": len(training), "initial_train_start": str(training["Date"].min().date()),
                   "initial_train_end": str(training["Date"].max().date()),
                   "test_rows": len(test), "test_start": str(first.date()), "test_end": str(last.date()),
                   "excluded_rows": excluded,
                   "pretest_unreleased_at_first_origin": sample.loc[
                       sample["Sample"].eq("pretest_target_unreleased_at_first_origin"), "Date"
                   ].dt.strftime("%Y-%m-%d").tolist(),
                   "eligibility_rule": "Finite target and all five candidate predictors, shared by every subset; no imputation.",
                   "evaluation_dates": test["Date"].dt.strftime("%Y-%m-%d").tolist()},
        "validation": {"ordered_unique_complete_calendar": True, "raw_macro_complete_finite": True,
                       "transformation_checks": checks},
        "selection_policy": "Pre-specify delayed-information autoregression and full macro references; select smaller subsets using pre-2017 chronological validation only. Test-period rankings are exploratory.",
    }
    return sample.reset_index(drop=True), contract, schedule


def export_macro_benchmark_contract(
    path: str | Path,
    output_dir: str | Path,
    **kwargs,
) -> dict[str, Path]:
    """Save the audited sample, metadata contract, and assumed origin schedule."""
    sample, contract, schedule = build_macro_benchmark_contract(path, **kwargs)
    policy_path = Path(__file__).resolve().parents[1] / "database" / "metadata" / "macro_availability_policy.json"
    contract["availability_audit"] = json.loads(policy_path.read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"sample": output_dir / "common_sample.csv", "contract": output_dir / "experiment_contract.json",
             "schedule": output_dir / "forecast_schedule.csv", "report": output_dir / "CONTRACT.md"}
    sample.to_csv(paths["sample"], index=False)
    schedule.to_csv(paths["schedule"], index=False)
    paths["contract"].write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = contract["sample"]
    lines = [
        "# Macroeconomic benchmark: Part A contract", "",
        f"Generated: {contract['created_at_utc']}", "",
        f"Status: **{contract['status']}**. No models fitted.", "",
        "## Dataset and common sample", "",
        "| Item | Value |", "|---|---|",
        f"| Source snapshot | {contract['dataset']['filename']} |",
        f"| Region / named source | {contract['dataset']['region']} / {contract['dataset']['news_source']['source_code']} |",
        f"| Source rows / eligible rows | {summary['source_rows']} / {summary['common_rows']} |",
        f"| Initial training | {summary['initial_train_start']} through {summary['initial_train_end']}: {summary['initial_train_rows']} months |",
        f"| Evaluation | {summary['test_start']} through {summary['test_end']}: {summary['test_rows']} months |",
        "", "All sixteen planned subsets use the same finite target and predictor sample. Sentiment is not required for eligibility.",
        "", "## Timing and vintage interpretation", "", contract["timing"]["description"], "",
        contract["timing"]["availability_rule"], "",
        contract["timing"]["baseline_interpretation"], "",
        f"Pre-test target months not yet available at the first origin: {', '.join(summary['pretest_unreleased_at_first_origin']) or 'none'}. They remain in the sample and become eligible only at later origins.", "",
        "Historical release dates have not been reconstructed. The input is a potentially revised historical snapshot, not historical vintages. A row lag does not eliminate revision look-ahead.",
        "", "## Excluded observations", "",
    ]
    lines.extend(f"- {row['date']}: {', '.join(row['reasons'])}." for row in summary["excluded_rows"])
    lines.extend(["", "## Series availability audit", ""])
    for item in contract["availability_audit"]["series"]:
        references = " ".join(f"[Source {i + 1}]({url})" for i, url in enumerate(item["source_urls"]))
        lines.append(f"- **{item['series']}:** {item['availability']} {item['revision']} {references}")
    lines.extend(["", "## Reproducibility", "",
                  f"Dataset SHA-256: `{contract['dataset']['sha256']}`.", "",
                  "See [the machine-readable contract](experiment_contract.json), [common sample](common_sample.csv), and [origin schedule](forecast_schedule.csv).",
                  "The schedule uses assumed availability dates, never observed release dates. Live forecasts require a separate actual-release check.", ""])
    paths["report"].write_text("\n".join(lines), encoding="utf-8")
    return paths
