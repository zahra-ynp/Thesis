"""Chapter 4 figures drawn from archived European results, without model fitting."""

import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
STATE_COLORS = ["#0072B2", "#D55E00"]


def plot_state_lag_selection(structure, *, ylim=None):
    """Compare the unrestricted state-and-lag grid on its selection period."""
    candidates = [
        item for item in structure["validation_candidates"]
        if item["family"] == "AR-HMM" and set(item["settings"]) == {"n_states"}
    ]
    states = sorted({item["settings"]["n_states"] for item in candidates})
    selected = structure["selected_structure"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 2.9))
        for state_count, color, marker in zip(states, STATE_COLORS, ["o", "s"]):
            rows = sorted(
                (item for item in candidates if item["settings"]["n_states"] == state_count),
                key=lambda item: len(item["features"]),
            )
            ax.plot(
                [len(item["features"]) for item in rows],
                [item["metrics"]["rmse"] for item in rows],
                marker=marker, color=color, label=f"{state_count} states", lw=1.6,
            )
        ax.scatter(
            [len(selected["features"])], [selected["metrics"]["rmse"]],
            s=120, facecolors="none", edgecolors=STATE_COLORS[states.index(selected["settings"]["n_states"])],
            zorder=4,
        )
        values = [item["metrics"]["rmse"] for item in candidates]
        padding = max(max(values) - min(values), 0.01) * 0.25
        ax.set(
            xlabel="Number of inflation lags",
            ylabel="Validation RMSE (percentage points)",
            xticks=sorted({len(item["features"]) for item in candidates}),
            ylim=ylim if ylim is not None else (min(values) - padding, max(values) + padding),
        )
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
    return fig


def plot_european_regimes(model):
    """Show the saved two-state estimates and pre-observation probabilities."""
    frame = pd.DataFrame(model["forecasts"])
    frame["Date"] = pd.to_datetime(frame.Date)
    labels = model["parameters"]["regime_labels"]
    probabilities = frame[[f"Probability_{label}" for label in labels]].to_numpy()
    if len(labels) != 2 or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Expected two states with probabilities summing to one.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("State probabilities must be finite and between zero and one.")
    assigned = np.array(labels)[probabilities.argmax(axis=1)]
    if not np.array_equal(assigned, frame.PredictedRegime.to_numpy()):
        raise ValueError("Saved regime labels disagree with the predicted probabilities.")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            2, 1, figsize=(6.8, 4.6), sharex=True,
            gridspec_kw={"height_ratios": [1.65, 1]},
        )
        axes[0].plot(frame.Date, frame.Actual, color="#252525", lw=1.3, label="Observed inflation")
        axes[0].plot(frame.Date, frame.Forecast, color="#888888", lw=1.1, label="Model estimate")
        for i, (label, color) in enumerate(zip(labels, STATE_COLORS)):
            mask = frame.PredictedRegime.eq(label)
            axes[0].scatter(
                frame.loc[mask, "Date"], frame.loc[mask, "Forecast"],
                color=color, s=21, label=f"State {i + 1}", zorder=4,
            )
        axes[0].set_ylabel("Monthly HICP inflation (%)")
        axes[0].legend(loc="upper right", ncol=2, frameon=False, fontsize=9)
        axes[1].stackplot(frame.Date, *probabilities.T, colors=STATE_COLORS, alpha=0.8)
        axes[1].set(
            ylabel="Predicted state probability", ylim=(0, 1),
            xlabel="Target month", yticks=[0, 0.5, 1],
        )
        axes[1].legend(
            handles=[Patch(color=color, label=f"State {i + 1}") for i, color in enumerate(STATE_COLORS)],
            frameon=False, ncol=2, loc="upper right", fontsize=9,
        )
        axes[1].xaxis.set_major_locator(mdates.YearLocator())
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for ax in axes:
            ax.grid(axis="y", alpha=0.2)
            ax.set_xlim(frame.Date.min(), frame.Date.max())
        fig.tight_layout(pad=0.8)
    return fig


def plot_evaluation_timeline(results):
    """Read calendar ranges from the archived structure design and fold labels."""
    design = results["structure_validation"]["design"]
    selection_start, selection_end = [pd.Timestamp(value) for value in design["selection_period"].split(" to ")]
    final_start, final_end = [pd.Timestamp(value) for value in design["final_test_period"].split(" to ")]
    folds = results["macro_experiment"]["rolling_validation"]["folds"]
    rows = []
    training_starts = []
    for i, fold in enumerate(folds, start=1):
        # Archived labels have the form '2007-2016 → 2017-2018' or '2007-2020 → 2021'.
        years = [int(value) for value in re.findall(r"\d{4}", fold["fold"])]
        if len(years) not in (3, 4):
            raise ValueError(f"Unrecognised validation fold label: {fold['fold']}")
        train_start, train_end, evaluation_start = years[:3]
        training_starts.append(train_start)
        rows.append((f"Validation block {i}", train_end, evaluation_start, years[-1]))
    first_year = min(training_starts)
    rows.insert(0, ("Initial structure selection", selection_start.year - 1, selection_start.year, selection_end.year))
    rows.append(("Final European test", final_start.year - 1, final_start.year, final_end.year))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.8, 3.4))
        for i, (_, train_end, start, end) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.broken_barh([(first_year, train_end - first_year + 1)], (y - 0.28, 0.56), facecolors="#527a75")
            ax.broken_barh([(start, end - start + 1)], (y - 0.28, 0.56), facecolors=STATE_COLORS[1] if i == len(rows) - 1 else STATE_COLORS[0])
        ax.set(
            yticks=range(len(rows)), yticklabels=[row[0] for row in rows[::-1]],
            xlim=(first_year, final_end.year + 1),
            xticks=sorted({first_year, first_year + 4, selection_start.year - 1, selection_start.year + 2, final_start.year, final_end.year}),
            xlabel="Calendar year",
        )
        ax.legend(
            handles=[Patch(color=color, label=label) for color, label in [
                ("#527a75", "Training"), (STATE_COLORS[0], "Validation"), (STATE_COLORS[1], "Final test"),
            ]],
            loc="upper left", bbox_to_anchor=(0, 1.2), ncol=3, frameon=False, fontsize=10,
        )
        ax.grid(axis="x", alpha=0.15)
        fig.tight_layout()
    return fig
