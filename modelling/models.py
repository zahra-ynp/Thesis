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
    target: str = "Inflation_MoM",
) -> ForecastResult:
    """Direct OLS forecasts using only targets available at each scheduled origin."""
    frame = sample.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["TargetAssumedAvailableBy"] = pd.to_datetime(frame["TargetAssumedAvailableBy"])
    rows, coefficients = [], []
    for item in schedule.itertuples(index=False):
        origin, target_date = pd.Timestamp(item.ForecastOrigin), pd.Timestamp(item.TargetMonth)
        train = frame.loc[frame["TargetAssumedAvailableBy"] <= origin]
        target_row = frame.loc[frame["Date"] == target_date]
        if len(target_row) != 1 or len(train) <= len(features) + 1:
            raise ValueError(f"Insufficient data for {name} at {target_date.date()}.")
        x = np.column_stack([np.ones(len(train)), train[list(features)].to_numpy(dtype=float)])
        y = train[target].to_numpy(dtype=float)
        x_next = np.r_[1.0, target_row.iloc[0][list(features)].to_numpy(dtype=float)]
        if not all(np.isfinite(v).all() for v in (x, y, x_next)):
            raise ValueError(f"Nonfinite data for {name} at {target_date.date()}.")
        beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        if rank < x.shape[1]:
            raise ValueError(f"Rank-deficient regression for {name} at {target_date.date()}.")
        prediction = float(x_next @ beta)
        actual = float(target_row.iloc[0][target])
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


def _filtered_probability_update(
    actual: float,
    state_forecasts: np.ndarray,
    variance: float | np.ndarray,
    predicted_probabilities: np.ndarray,
) -> np.ndarray:
    """Bayes update in log space to avoid all-zero regime likelihoods."""
    variance = np.maximum(np.asarray(variance, dtype=float), np.finfo(float).tiny)
    log_weights = (
        np.log(np.clip(predicted_probabilities, np.finfo(float).tiny, None))
        - 0.5 * ((float(actual) - state_forecasts) ** 2) / variance
        - 0.5 * np.log(2.0 * np.pi * variance)
    )
    weights = np.exp(log_weights - np.max(log_weights))
    return weights / weights.sum()


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
        filtered = _filtered_probability_update(
            actual, means, variance, predicted_probabilities
        )

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
    switching_intercept: bool = True,
    switching_ar: bool = True,
    switching_variance: bool = False,
) -> ForecastResult:
    """Fit the existing AR-HMM with state-specific predictor coefficients."""
    import statsmodels.api as sm

    train, test = chronological_split(
        data, target=target, features=features, test_start=test_start
    )
    model = sm.tsa.MarkovRegression(
        train[target],
        k_regimes=n_states,
        trend="c",
        exog=train[list(features)],
        switching_trend=switching_intercept,
        switching_exog=switching_ar,
        switching_variance=switching_variance,
    )
    fitted = model.fit(
        disp=False, maxiter=maxiter, search_reps=search_reps, rng=random_seed
    )

    constants = (np.array([fitted.params[f"const[{state}]"] for state in range(n_states)])
                 if switching_intercept
                 else np.repeat(float(fitted.params["const"]), n_states))
    coefficients = np.array([
        [
            (fitted.params[f"x{feature_index + 1}[{state}]"] if switching_ar else
             fitted.params[next(parameter for parameter in fitted.params.index
                                if parameter == f"x{feature_index + 1}" or
                                parameter.startswith(f"x{feature_index + 1}["))])
            for feature_index in range(len(features))
        ]
        for state in range(n_states)
    ])
    variances = (np.array([fitted.params[f"sigma2[{state}]"] for state in range(n_states)])
                 if switching_variance
                 else np.repeat(float(fitted.params["sigma2"]), n_states))
    transition = np.asarray(fitted.regime_transition[:, :, 0])
    filtered = np.asarray(fitted.filtered_marginal_probabilities.iloc[-1], dtype=float)
    reference_features = train[list(features)].mean().to_numpy(dtype=float)
    reference_levels = constants + coefficients @ reference_features
    variance_only = (not switching_intercept and not switching_ar and switching_variance)
    if variance_only and n_states == 2:
        order = np.argsort(variances, kind="stable")
        regime_labels = ["Low volatility", "High volatility"]
        regime_label_rule = "ascending regime-specific innovation variance"
    else:
        order = np.argsort(reference_levels, kind="stable")
        regime_labels = (["Low", "Moderate", "High", "Crisis"] if n_states == 4
                         else [f"Level {i + 1}" for i in range(n_states)])
        regime_label_rule = "ascending conditional inflation at initial-training mean predictors"
    rows = []
    for (_, row), actual in zip(test.iterrows(), test[target].to_numpy()):
        predicted_probabilities = transition @ filtered
        state_forecasts = constants + coefficients @ row[list(features)].to_numpy(dtype=float)
        ordered_probabilities = predicted_probabilities[order]
        forecast = float(predicted_probabilities @ state_forecasts)
        rows.append({
            "Date": row["Date"],
            "Actual": float(actual),
            "Forecast": forecast,
            "Error": float(actual) - forecast,
            "PredictedRegime": regime_labels[int(ordered_probabilities.argmax())],
            **{f"Probability_{label}": float(probability)
               for label, probability in zip(regime_labels, ordered_probabilities)},
        })
        filtered = _filtered_probability_update(
            actual, state_forecasts, variances, predicted_probabilities
        )

    output = pd.DataFrame(rows)
    if output.empty or not np.isfinite(output[["Actual", "Forecast"]].to_numpy()).all():
        raise RuntimeError(f"Static AR-HMM produced nonfinite forecasts for {name}.")
    parameters = {
        "features": list(features),
        "estimator": f"static-coefficient {n_states}-state AR-HMM",
        "n_states": n_states,
        "state_constants": constants.tolist(),
        "state_coefficients": {
            feature: coefficients[:, index].tolist() for index, feature in enumerate(features)
        },
        "state_variances": variances.tolist(),
        "transition_matrix": transition.tolist(),
        "switching_intercept": switching_intercept,
        "switching_ar": switching_ar,
        "switching_variance": switching_variance,
        "log_likelihood": float(fitted.llf),
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
        "regime_labels": regime_labels,
        "regime_label_rule": regime_label_rule,
        "regime_reference_features": dict(zip(features, reference_features.tolist())),
        "reference_state_levels": reference_levels[order].tolist(),
        "raw_state_order": order.tolist(),
        "probability_timing": "predicted before observing the target month inflation",
        "converged": bool(fitted.mle_retvals.get("converged", False)),
        "initialization": "original seeded static fit",
        "random_seed": random_seed,
    }
    return ForecastResult(name, output.reset_index(drop=True), parameters, fitted)


def fit_expanding_ar_hmm(
    sample: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    features: Sequence[str],
    name: str,
    target: str = "Inflation_MoM",
    n_states: int = 2,
    maxiter: int = 2000,
    search_reps: int = 30,
    random_seed: int = 42,
) -> ForecastResult:
    """Refit an AR-HMM at every scheduled origin using all available targets."""
    required = {"Date", "TargetAssumedAvailableBy", target, *features}
    missing = required - set(sample.columns)
    if missing:
        raise ValueError(f"Missing expanding AR-HMM columns: {sorted(missing)}")
    schedule_missing = {"ForecastOrigin", "TargetMonth"} - set(schedule.columns)
    if schedule_missing:
        raise ValueError(f"Missing schedule columns: {sorted(schedule_missing)}")

    import statsmodels.api as sm

    frame = sample.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["TargetAssumedAvailableBy"] = pd.to_datetime(
        frame["TargetAssumedAvailableBy"]
    )
    rows, diagnostics = [], []
    start_params = None
    fitted = None
    reference_features = None
    regime_labels = (["Low", "Moderate", "High", "Crisis"] if n_states == 4
                     else [f"Level {i + 1}" for i in range(n_states)])

    for item in schedule.itertuples(index=False):
        origin = pd.Timestamp(item.ForecastOrigin)
        target_date = pd.Timestamp(item.TargetMonth)
        train = frame.loc[
            frame["TargetAssumedAvailableBy"] <= origin,
            ["Date", target, *features],
        ].dropna()
        target_row = frame.loc[frame["Date"] == target_date, [target, *features]]
        if len(target_row) != 1 or len(train) <= len(features) + n_states:
            raise ValueError(
                f"Insufficient data for expanding AR-HMM {name} "
                f"at {target_date.date()}."
            )
        if not np.isfinite(train[[target, *features]].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite training data for {name} at {target_date.date()}.")
        next_features = target_row.iloc[0][list(features)].to_numpy(dtype=float)
        if not np.isfinite(next_features).all():
            raise ValueError(f"Nonfinite predictors for {name} at {target_date.date()}.")
        if reference_features is None:
            # Freeze the naming reference at the initial training sample; no test data.
            reference_features = train[list(features)].mean().to_numpy(dtype=float)

        model = sm.tsa.MarkovRegression(
            train[target],
            k_regimes=n_states,
            trend="c",
            exog=train[list(features)],
            switching_exog=True,
            switching_variance=False,
        )
        # Boundary transition probabilities can make a warm start impossible to
        # untransform. Retry a fresh initialization using this training window only.
        attempts = ([('warm', random_seed)] if start_params is not None else [])
        attempts += [('fresh', random_seed + offset) for offset in range(3)]
        attempts += [('fresh-lbfgs', random_seed), ('fresh-powell', random_seed)]
        fitted = None
        for initialization, seed in attempts:
            np.random.seed(seed)
            try:
                candidate = model.fit(
                    disp=False, maxiter=maxiter,
                    method=('lbfgs' if initialization == 'fresh-lbfgs' else
                            'powell' if initialization == 'fresh-powell' else 'bfgs'),
                    search_reps=0 if initialization == 'warm' else search_reps,
                    **({'start_params': start_params} if initialization == 'warm' else {}),
                )
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                continue
            if candidate.mle_retvals.get('converged', False) and np.isfinite(candidate.params).all():
                fitted = candidate
                break
        if fitted is None:
            raise RuntimeError(f"AR-HMM did not converge for {name} at {target_date.date()} after fresh starts.")
        start_params = np.asarray(fitted.params)

        constants = np.array([
            fitted.params[f"const[{state}]"] for state in range(n_states)
        ])
        coefficients = np.array([
            [
                fitted.params[f"x{feature_index + 1}[{state}]"]
                for feature_index in range(len(features))
            ]
            for state in range(n_states)
        ])
        transition = np.asarray(fitted.regime_transition[:, :, 0])
        filtered = np.asarray(
            fitted.filtered_marginal_probabilities.iloc[-1], dtype=float
        )
        predicted_probabilities = transition @ filtered
        state_forecasts = constants + coefficients @ next_features
        reference_levels = constants + coefficients @ reference_features
        order = np.argsort(reference_levels, kind="stable")
        ordered_probabilities = predicted_probabilities[order]
        prediction = float(predicted_probabilities @ state_forecasts)
        actual = float(target_row.iloc[0][target])
        rows.append({
            "Date": target_date,
            "Actual": actual,
            "Forecast": prediction,
            "Error": actual - prediction,
            "PredictedRegime": regime_labels[int(ordered_probabilities.argmax())],
            **{f"Probability_{label}": float(probability)
               for label, probability in zip(regime_labels, ordered_probabilities)},
        })
        diagnostics.append({
            "target_month": str(target_date.date()),
            "origin": str(origin.date()),
            "training_end": str(train["Date"].max().date()),
            "training_rows": int(len(train)),
            "converged": True,
            "initialization": initialization,
            "random_seed": seed,
            "raw_state_order": order.tolist(),
            "reference_state_levels": reference_levels[order].tolist(),
            "training_state_mass": np.asarray(
                fitted.smoothed_marginal_probabilities
            ).sum(axis=0)[order].tolist(),
        })

    if fitted is None:
        raise ValueError("The expanding AR-HMM schedule is empty.")
    return ForecastResult(name, pd.DataFrame(rows), {
        "features": list(features),
        "estimator": f"expanding-window {n_states}-state AR-HMM",
        "n_states": n_states,
        "regime_labels": regime_labels,
        "regime_label_rule": "ascending conditional inflation at initial-training mean predictors",
        "regime_reference_features": dict(zip(features, reference_features.tolist())),
        "probability_timing": "predicted before observing the target month inflation",
        "fits_by_origin": diagnostics,
    }, fitted)


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
        n_jobs=1,
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
