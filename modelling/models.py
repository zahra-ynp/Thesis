from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """Standard result returned by every forecasting function."""

    name: str
    forecasts: pd.DataFrame
    parameters: dict[str, Any]
    fitted_model: Any = field(default=None, repr=False)


def chronological_split(
    data: pd.DataFrame,
    *,
    target: str,
    features: Sequence[str],
    test_start: str = "2017-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the shared complete-case chronological train/test split."""
    required = ["Date", target, *features]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing model columns: {sorted(missing)}")

    sample = data[required].copy()
    sample["Date"] = pd.to_datetime(sample["Date"])
    sample = sample.dropna().sort_values("Date").reset_index(drop=True)
    cutoff = pd.Timestamp(test_start)
    train = sample.loc[sample["Date"] < cutoff].copy()
    test = sample.loc[sample["Date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("test_start must leave observations in both train and test periods.")
    return train, test


def fit_ols_forecast(
    data: pd.DataFrame,
    *,
    features: Sequence[str],
    name: str,
    target: str = "Inflation_MoM",
    test_start: str = "2017-01-01",
) -> ForecastResult:
    """Fit the existing static AR(1) or ARX specification."""
    import statsmodels.api as sm

    train, test = chronological_split(
        data, target=target, features=features, test_start=test_start
    )
    fitted = sm.OLS(train[target], sm.add_constant(train[list(features)])).fit()
    forecast = fitted.predict(sm.add_constant(test[list(features)], has_constant="add"))
    output = pd.DataFrame({
        "Date": test["Date"].to_numpy(),
        "Actual": test[target].to_numpy(),
        "Forecast": np.asarray(forecast),
    })
    output["Error"] = output["Actual"] - output["Forecast"]
    return ForecastResult(name, output, fitted.params.to_dict(), fitted)


def fit_ar1(data: pd.DataFrame, *, test_start: str = "2017-01-01") -> ForecastResult:
    return fit_ols_forecast(
        data, features=["Inflation_Lag1"], name="AR(1)", test_start=test_start
    )


def fit_expanding_macro(
    sample: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    features: Sequence[str],
    name: str,
) -> ForecastResult:
    """Direct OLS forecasts using only targets available at each scheduled origin."""
    frame = sample.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["TargetAssumedAvailableBy"] = pd.to_datetime(frame["TargetAssumedAvailableBy"])
    rows, coefficients = [], []
    for item in schedule.itertuples(index=False):
        origin, target_date = pd.Timestamp(item.ForecastOrigin), pd.Timestamp(item.TargetMonth)
        train = frame.loc[frame["TargetAssumedAvailableBy"] <= origin]
        target = frame.loc[frame["Date"] == target_date]
        if len(target) != 1 or len(train) <= len(features) + 1:
            raise ValueError(f"Insufficient data for {name} at {target_date.date()}.")
        x = np.column_stack([np.ones(len(train)), train[list(features)].to_numpy(dtype=float)])
        y = train["Inflation_MoM"].to_numpy(dtype=float)
        x_next = np.r_[1.0, target.iloc[0][list(features)].to_numpy(dtype=float)]
        if not all(np.isfinite(v).all() for v in (x, y, x_next)):
            raise ValueError(f"Nonfinite data for {name} at {target_date.date()}.")
        beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        if rank < x.shape[1]:
            raise ValueError(f"Rank-deficient regression for {name} at {target_date.date()}.")
        prediction = float(x_next @ beta)
        actual = float(target.iloc[0]["Inflation_MoM"])
        rows.append({"Date": target_date, "Actual": actual, "Forecast": prediction,
                     "Error": actual - prediction})
        coefficients.append({"target_month": str(target_date.date()), "origin": str(origin.date()),
                             "training_end": str(train["Date"].max().date()), "training_rows": len(train),
                             "coefficients": dict(zip(["intercept", *features], beta.tolist()))})
    return ForecastResult(name, pd.DataFrame(rows), {
        "features": list(features), "estimator": "expanding-window ordinary least squares",
        "coefficients_by_origin": coefficients,
    })


def fit_arx_sentiment(
    data: pd.DataFrame, *, test_start: str = "2017-01-01"
) -> ForecastResult:
    return fit_ols_forecast(
        data,
        features=["Inflation_Lag1", "News_Sentiment"],
        name="ARX + sentiment",
        test_start=test_start,
    )


def _normal_density(value: float, means: np.ndarray, variance: float) -> np.ndarray:
    return np.exp(-0.5 * ((value - means) ** 2) / variance) / np.sqrt(
        2.0 * np.pi * variance
    )


def fit_hmm(
    data: pd.DataFrame,
    *,
    target: str = "Inflation_MoM",
    test_start: str = "2017-01-01",
    n_states: int = 2,
    random_seed: int = 42,
    maxiter: int = 2000,
    search_reps: int = 30,
) -> ForecastResult:
    """Fit and roll forward the existing Markov-switching mean model."""
    import statsmodels.api as sm

    train, test = chronological_split(data, target=target, features=[], test_start=test_start)
    np.random.seed(random_seed)
    fitted = sm.tsa.MarkovRegression(
        train[target], k_regimes=n_states, trend="c", switching_variance=False
    ).fit(disp=False, maxiter=1000, search_reps=20)

    means = np.array([fitted.params[f"const[{state}]"] for state in range(n_states)])
    variance = float(fitted.params["sigma2"])
    transition = np.asarray(fitted.regime_transition[:, :, 0])
    filtered = np.asarray(fitted.filtered_marginal_probabilities.iloc[-1], dtype=float)
    forecasts = []
    for actual in test[target].to_numpy():
        predicted_probabilities = transition @ filtered
        forecasts.append(float(predicted_probabilities @ means))
        filtered = predicted_probabilities * _normal_density(actual, means, variance)
        filtered /= filtered.sum()

    output = pd.DataFrame({"Date": test["Date"], "Actual": test[target], "Forecast": forecasts})
    output["Error"] = output["Actual"] - output["Forecast"]
    parameters = {
        "state_means": means.tolist(),
        "variance": variance,
        "transition_matrix": transition.tolist(),
    }
    return ForecastResult("HMM", output.reset_index(drop=True), parameters, fitted)


def fit_ar_hmm(
    data: pd.DataFrame,
    *,
    features: Sequence[str],
    name: str,
    target: str = "Inflation_MoM",
    test_start: str = "2017-01-01",
    n_states: int = 2,
    maxiter: int = 2000,
    search_reps: int = 30,
    random_seed: int = 42,
) -> ForecastResult:
    """Fit the existing AR-HMM with state-specific predictor coefficients."""
    import statsmodels.api as sm

    train, test = chronological_split(
        data, target=target, features=features, test_start=test_start
    )
    np.random.seed(random_seed)
    fitted = sm.tsa.MarkovRegression(
        train[target],
        k_regimes=n_states,
        trend="c",
        exog=train[list(features)],
        switching_exog=True,
        switching_variance=False,
    ).fit(disp=False, maxiter=maxiter, search_reps=search_reps)

    constants = np.array([fitted.params[f"const[{state}]"] for state in range(n_states)])
    coefficients = np.array([
        [fitted.params[f"x{feature_index + 1}[{state}]"] for feature_index in range(len(features))]
        for state in range(n_states)
    ])
    variance = float(fitted.params["sigma2"])
    transition = np.asarray(fitted.regime_transition[:, :, 0])
    filtered = np.asarray(fitted.filtered_marginal_probabilities.iloc[-1], dtype=float)
    forecasts = []
    for (_, row), actual in zip(test.iterrows(), test[target].to_numpy()):
        predicted_probabilities = transition @ filtered
        state_forecasts = constants + coefficients @ row[list(features)].to_numpy(dtype=float)
        forecasts.append(float(predicted_probabilities @ state_forecasts))
        filtered = predicted_probabilities * _normal_density(actual, state_forecasts, variance)
        filtered /= filtered.sum()

    output = pd.DataFrame({"Date": test["Date"], "Actual": test[target], "Forecast": forecasts})
    output["Error"] = output["Actual"] - output["Forecast"]
    parameters = {
        "state_constants": constants.tolist(),
        "state_coefficients": {
            feature: coefficients[:, index].tolist() for index, feature in enumerate(features)
        },
        "variance": variance,
        "transition_matrix": transition.tolist(),
    }
    return ForecastResult(name, output.reset_index(drop=True), parameters, fitted)


def fit_lasso(
    data: pd.DataFrame,
    *,
    features: Sequence[str],
    name: str,
    target: str = "Inflation_MoM",
    test_start: str = "2017-01-01",
) -> ForecastResult:
    """Fit the existing standardized LASSO with time-series cross-validation."""
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train, test = chronological_split(
        data, target=target, features=features, test_start=test_start
    )
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("lasso", Lasso(max_iter=100000)),
    ])
    fitted = GridSearchCV(
        pipeline,
        {"lasso__alpha": np.logspace(-4, 0, 60)},
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    ).fit(train[list(features)], train[target])
    forecasts = fitted.predict(test[list(features)])
    output = pd.DataFrame({"Date": test["Date"], "Actual": test[target], "Forecast": forecasts})
    output["Error"] = output["Actual"] - output["Forecast"]
    estimator = fitted.best_estimator_.named_steps["lasso"]
    parameters = {
        "alpha": float(fitted.best_params_["lasso__alpha"]),
        "standardized_coefficients": dict(zip(features, estimator.coef_.tolist())),
    }
    return ForecastResult(name, output.reset_index(drop=True), parameters, fitted)
