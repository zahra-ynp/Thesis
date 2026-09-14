"""Controlled AR-HMM experiments for notebook 02."""
from pathlib import Path
from itertools import combinations
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modelling.data_pipeline import MACRO_BENCHMARK_FEATURES
from modelling.models import ForecastResult, fit_expanding_ar_hmm, fit_expanding_macro


def run_notebook_results(dataset, cache_path, *, test_start='2022-01-01',
                         test_end='2025-11-01', rerun=False):
    """Run every notebook comparison with one input and one resumable results file."""
    from modelling.models import fit_ols_forecast, fit_hmm, fit_ar_hmm, fit_lasso
    from modelling.evaluation import forecast_metrics

    dataset, cache_path = Path(dataset), Path(cache_path)
    frame = pd.read_csv(dataset, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    if frame.Date.duplicated().any() or not pd.DatetimeIndex(frame.Date).equals(
        pd.date_range(frame.Date.min(), frame.Date.max(), freq='MS')
    ):
        raise ValueError('The input must contain one row per calendar month.')
    for target in ['Inflation_MoM', 'Inflation_YoY']:
        for lag in range(1, 7):
            frame[f'{target}_Lag{lag}'] = frame[target].shift(lag)
    for lag in range(1, 7):
        frame[f'Sentiment_Lag{lag}'] = frame.News_Sentiment.shift(lag)
    macro_columns = list(MACRO_BENCHMARK_FEATURES[1:])
    sentiments = ['News_Sentiment', 'Sentiment_Median', 'Negative_Share', 'Sentiment_Std']
    common = ['Inflation_MoM', 'Inflation_YoY', *macro_columns, *sentiments,
              *[f'{target}_Lag{lag}' for target in ['Inflation_MoM', 'Inflation_YoY'] for lag in range(1, 7)],
              *[f'Sentiment_Lag{lag}' for lag in range(1, 7)]]
    # Lag on the full calendar first. Never shift across a missing month.
    sample = frame.loc[frame[common].notna().all(axis=1) & frame.Date.le(test_end)].copy()
    if not pd.DatetimeIndex(sample.Date).equals(pd.date_range(sample.Date.min(), sample.Date.max(), freq='MS')):
        raise ValueError('An internal training gap requires a missing-observation state filter.')
    dates = sample.loc[sample.Date.ge(test_start), 'Date']
    if dates.empty or not sample.Date.lt(test_start).any():
        raise ValueError('The requested split must contain training and test observations.')
    sample['TargetAssumedAvailableBy'] = sample.Date + pd.offsets.MonthEnd(2)
    schedule = pd.DataFrame({'TargetMonth': dates, 'ForecastOrigin': dates + pd.offsets.MonthEnd(0)})
    signature = {'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                 'test_start': test_start, 'test_end': test_end, 'version': 3,
                 'main_estimation': 'static'}
    saved = json.loads(cache_path.read_text()) if cache_path.exists() and not rerun else {}
    if saved.get('signature') != signature:
        saved = {'signature': signature, 'models': {}}
    saved['groups'] = {}
    saved['sample'] = {
        'training_start': str(sample.Date.min().date()),
        'initial_training_end': str(sample.loc[sample.Date.lt(test_start), 'Date'].max().date()),
        'scored_start': str(dates.min().date()), 'scored_end': str(dates.max().date()),
        'scored_months': len(dates),
        'unavailable_months': pd.date_range(test_start, test_end, freq='MS').difference(dates).strftime('%Y-%m').tolist(),
        'timing': ('Static coefficients estimated through December 2021. Hidden-state '
                   'probabilities update during testing; target-month sentiment is a nowcast input.'),
        'data_vintage': 'Latest revised FRED data, not a real-time vintage backtest.',
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        temporary = cache_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(saved, allow_nan=False), encoding='utf-8')
        temporary.replace(cache_path)

    def fit(group, target, family, features, *, states=2, mode='static', label=None):
        key = '|'.join([target, family, str(states), mode, *features])
        saved['groups'].setdefault(group, []).append(key)
        if key in saved['models']:
            return key
        name = label or f'{target.removeprefix("Inflation_")} {family}'
        print(f'Fitting {name} ({group})', flush=True)
        if family == 'Naive':
            test = sample.loc[sample.Date.isin(dates)]
            result = ForecastResult(name, pd.DataFrame({'Date': test.Date, 'Actual': test[target],
                'Forecast': test[features[0]]}), {'features': features})
        elif mode == 'static':
            if family == 'HMM':
                result = fit_hmm(sample, target=target, test_start=test_start)
            elif family == 'AR-HMM':
                try:
                    result = fit_ar_hmm(sample, features=features, name=name, target=target,
                                        test_start=test_start, n_states=states)
                except RuntimeError:
                    print(f'Retrying {name} with training-only predictor scaling', flush=True)
                    training = sample.loc[sample.Date.lt(test_start), features]
                    center, scale = training.mean(), training.std().replace(0, 1)
                    scaled_sample = sample.copy()
                    scaled_sample[features] = (sample[features] - center) / scale
                    result = fit_ar_hmm(scaled_sample, features=features, name=name,
                                        target=target, test_start=test_start, n_states=states)
                    result.parameters['predictor_scaling'] = {
                        'center': center.to_dict(), 'scale': scale.to_dict(),
                        'policy': 'Fixed initial-training scaling after unscaled convergence failure',
                    }
            elif family == 'LASSO':
                result = fit_lasso(sample, features=features, name=name, target=target, test_start=test_start)
            else:
                result = fit_ols_forecast(sample, features=features, name=name, target=target, test_start=test_start)
        else:
            fitter = fit_expanding_ar_hmm if family == 'AR-HMM' else fit_expanding_macro
            try:
                result = fitter(sample, schedule, features=features, name=name, target=target,
                                **({'n_states': states} if family == 'AR-HMM' else {}))
            except RuntimeError:
                if family != 'AR-HMM':
                    raise
                print(f'Retrying {name} with training-only predictor scaling', flush=True)
                training = sample.loc[sample.Date.lt(test_start), features]
                center, scale = training.mean(), training.std().replace(0, 1)
                scaled_sample = sample.copy()
                scaled_sample[features] = (sample[features] - center) / scale
                result = fitter(scaled_sample, schedule, features=features, name=name,
                                target=target, n_states=states)
                result.parameters['predictor_scaling'] = {
                    'center': center.to_dict(), 'scale': scale.to_dict(),
                    'policy': 'Fixed initial-training scaling after unscaled convergence failure',
                }
        output = result.forecasts.copy()
        if not pd.DatetimeIndex(output.Date).equals(pd.DatetimeIndex(dates)):
            raise ValueError(f'Unexpected forecast dates for {name}.')
        if not np.isfinite(output[['Actual', 'Forecast']].to_numpy()).all():
            raise ValueError(f'Nonfinite forecasts for {name}.')
        output['Date'] = pd.to_datetime(output.Date).dt.strftime('%Y-%m-%d')
        saved['models'][key] = {'name': name, 'target': target, 'family': family,
            'features': features, 'states': states if family in ('AR-HMM', 'HMM') else None,
            'mode': mode, 'metrics': forecast_metrics(output.Actual, output.Forecast),
            'forecasts': output.to_dict('records'), 'parameters': result.parameters}
        checkpoint()
        return key

    mom = 'Inflation_MoM'
    lag1 = [f'{mom}_Lag1']
    for family, features, name in [
        ('AR', lag1, 'Static AR(1)'), ('AR', [*lag1, 'News_Sentiment'], 'Static ARX + sentiment'),
        ('HMM', [], 'Static HMM'), ('AR-HMM', lag1, 'Static AR-HMM'),
        ('AR-HMM', [*lag1, 'Sentiment_Lag1'], 'Static AR-HMM + lagged sentiment'),
        ('LASSO', [f'{mom}_Lag{i}' for i in range(1, 7)], 'Static LASSO'),
        ('LASSO', [*[f'{mom}_Lag{i}' for i in range(1, 7)], *[f'Sentiment_Lag{i}' for i in range(1, 7)]], 'Static LASSO + sentiment'),
    ]:
        fit('Static baselines', mom, family, features, mode='static', label=name)

    for family in ['AR', 'AR-HMM']:
        for mode in ['static', 'expanding']:
            fit('Static vs expanding', mom, family, lag1, mode=mode,
                label=f'{mode.title()} {family}(1)'
                + (' · 2 states' if family == 'AR-HMM' else ''))

    for target, macro_order in [(mom, 1), ('Inflation_YoY', 2)]:
        prefix = target.removeprefix('Inflation_')
        for order in range(1, 7):
            features = [f'{target}_Lag{i}' for i in range(1, order+1)]
            for family in ['AR', 'AR-HMM']:
                fit(f'{prefix} lag orders', target, family, features, mode='static',
                    label=f'{prefix} {family}({order})' + (' · 2 states' if family == 'AR-HMM' else ''))
        fit(f'{prefix} lag orders', target, 'Naive', [f'{target}_Lag1'], mode='static',
            label=f'{prefix} naive')
        if target == mom:
            fit('MoM state counts', target, 'AR', lag1, mode='static', label='MoM AR(1)')
            for count in [2, 3, 4]:
                fit('MoM state counts', target, 'AR-HMM', lag1, states=count, mode='static',
                    label=f'MoM AR-HMM(1) · {count} states')
        base = [f'{target}_Lag{i}' for i in range(1, macro_order+1)]
        macro_keys = {'AR': [], 'AR-HMM': []}
        for size in range(5):
            for subset in combinations(range(4), size):
                suffix = ' + '.join(['Oil', 'Unemployment', 'Industry', 'Fed rate'][i] for i in subset)
                features = [*base, *[macro_columns[i] for i in subset]]
                for family in macro_keys:
                    key = fit(f'{prefix} macro', target, family, features, mode='static',
                        label=f'{prefix} {family}({macro_order})' + (f' + {suffix}' if suffix else ''))
                    if subset:
                        macro_keys[family].append(key)
        for family in macro_keys:
            best_key = min(macro_keys[family], key=lambda key: saved['models'][key]['metrics']['rmse'])
            best = saved['models'][best_key]
            for features, name in [(base, f'{prefix} {family}({macro_order})'), (best['features'], best['name'])]:
                fit(f'{prefix} sentiment', target, family, features, mode='static', label=name)
                fit(f'{prefix} sentiment', target, family, [*features, 'News_Sentiment'],
                    mode='static', label=name+' + sentiment')
        if target == mom:
            for family in ['AR', 'AR-HMM']:
                fit('MoM sentiment aggregations', target, family, base, mode='static',
                    label=f'MoM {family}(1)')
                for feature in sentiments:
                    fit('MoM sentiment aggregations', target, family, [*base, feature], mode='static',
                        label=f'MoM {family}(1) + {feature}')
    checkpoint()
    print(f'Completed {len(saved["models"])} model specifications.', flush=True)
    return saved


def notebook_result_table(results, group):
    """Present a comparison from the single notebook cache."""
    return pd.DataFrame([{'key': key, 'model': results['models'][key]['name'],
        'family': results['models'][key]['family'], 'estimation': results['models'][key]['mode'],
        **results['models'][key]['metrics']}
        for key in dict.fromkeys(results['groups'][group])]).sort_values('rmse').reset_index(drop=True)


def plot_notebook_forecasts(results, keys, title):
    """Plot selected models on the requested calendar, including unavailable months."""
    keys = list(dict.fromkeys(keys))
    targets = {results['models'][key]['target'] for key in keys}
    if len(targets) != 1:
        raise ValueError('Plot monthly and YoY inflation separately.')
    calendar = pd.date_range(results['signature']['test_start'], results['signature']['test_end'], freq='MS')
    plotted = pd.DataFrame(index=calendar)
    for key in keys:
        model = results['models'][key]
        frame = pd.DataFrame(model['forecasts']).set_index('Date')
        frame.index = pd.to_datetime(frame.index)
        if 'Actual' not in plotted:
            plotted['Actual'] = frame.Actual
        plotted[model['name']] = frame.Forecast
    ax = plotted.plot(figsize=(14, 4), title=title, ylabel='Inflation (%)')
    unavailable_start = pd.Timestamp(results['sample']['scored_end']) + pd.offsets.MonthBegin(1)
    if unavailable_start <= calendar.max():
        ax.axvspan(unavailable_start, calendar.max(), color='#aaaaaa', alpha=.2,
                   label='Unavailable for common comparison')
    ax.set_xlim(calendar.min(), calendar.max())
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    plt.tight_layout()
    return ax



def plot_predicted_regimes(frame, labels=("Low", "Moderate", "High", "Crisis")):
    """Plot pre-observation probabilities, never full-sample smoothed regimes."""
    colors = (["#2878b5", "#23956f", "#e6a128", "#c64747"] if len(labels) == 4
              else [plt.get_cmap('viridis')(i / max(1, len(labels) - 1)) for i in range(len(labels))])
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(frame.Date, frame.Actual, color="#172b4d", label="Actual", lw=1.5)
    axes[0].plot(frame.Date, frame.Forecast, color="#777777", label="Forecast", lw=1.2)
    for label, color in zip(labels, colors):
        selected = frame.PredictedRegime.eq(label)
        axes[0].scatter(frame.loc[selected, "Date"], frame.loc[selected, "Forecast"],
                        color=color, label=label, s=24, zorder=4)
    axes[0].set(ylabel="Monthly inflation (%)", title=f"{len(labels)}-state AR-HMM: predicted regime at each target month")
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
    run_notebook_results(
        root / 'data/processed/modelling_datasets/extended_2025/us_nyt_monthly_experiment.csv',
        root / 'data/processed/model_results/results_static_2022_2025.json',
    )
