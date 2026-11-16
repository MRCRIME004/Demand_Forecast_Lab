"""Shared analysis engine for local Python and browser Pyodide. No network access."""
from pathlib import Path
from datetime import date, timedelta
from statistics import NormalDist
import base64
import csv
import hashlib
import io
import json
import math
import runpy
import time
import warnings
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.signal import lfilter, lfiltic
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import chi2, t as student_t, skew as scipy_skew, kurtosis as scipy_kurtosis

try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
    SKLEARN_IMPORT_ERROR = None
except ImportError as exc:
    RandomForestRegressor = None
    HAS_SKLEARN = False
    SKLEARN_IMPORT_ERROR = str(exc)

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as SMExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX as SMSARIMAX
    from statsmodels.tsa.seasonal import STL as SMSTL
    from statsmodels.tsa.stattools import (
        adfuller as sm_adfuller,
        kpss as sm_kpss,
        acf as sm_acf,
        pacf as sm_pacf,
    )
    HAS_STATSMODELS = True
    STATSMODELS_IMPORT_ERROR = None
except ImportError as exc:
    SMExponentialSmoothing = None
    SMSARIMAX = None
    SMSTL = None
    sm_adfuller = None
    sm_kpss = None
    sm_acf = None
    sm_pacf = None
    HAS_STATSMODELS = False
    STATSMODELS_IMPORT_ERROR = str(exc)

LABELS = {
    'SES': 'Exponential smoothing',
    'SMA': 'Moving average',
    'Weekday_DR': 'Weekday regression',
    'Weekday_Naive': 'Weekday naive',
    'Holt': "Holt's linear trend",
    'Weekday_Ridge': 'Ridge regression',
    'Weekday_RF': 'Random Forest regression',
    'Holt_Winters': 'Holt-Winters seasonal',
    'SARIMA': 'SARIMA (seasonal)',
    'Ensemble': 'Top-3 validation ensemble',
}
MODEL_COLORS = {
    'SMA': '#D97706',
    'SES': '#0D9488',
    'Weekday_DR': '#2563EB',
    'Weekday_Naive': '#9333EA',
    'Holt': '#DC2626',
    'Weekday_Ridge': '#0284C7',
    'Weekday_RF': '#65A30D',
    'Holt_Winters': '#E11D48',
    'SARIMA': '#4F46E5',
    'Ensemble': '#0F766E',
}
MODEL_STYLES = {
    'SMA': dict(ls='--', marker='s'),
    'SES': dict(ls='-', marker='o'),
    'Weekday_DR': dict(ls='-.', marker='^'),
    'Weekday_Naive': dict(ls=':', marker='D'),
    'Holt': dict(ls='--', marker='v'),
    'Weekday_Ridge': dict(ls='-', marker='P'),
    'Weekday_RF': dict(ls='-.', marker='X'),
    'Holt_Winters': dict(ls='--', marker='d'),
    'SARIMA': dict(ls=':', marker='p'),
    'Ensemble': dict(ls='-', marker='*'),
}
GRAPH_ACCENTS = {
    '01_demand': '#1D4ED8',
    '02_evaluation': '#2563EB',
    '03_future': '#7C3AED',
    '04_SES': '#0D9488',
    '04_SMA': '#D97706',
    '04_Weekday_DR': '#4338CA',
    '04_Weekday_Naive': '#9333EA',
    '05_weekday': '#0891B2',
    '06_errors': '#DC2626',
    '07_residuals': '#EA580C',
    '08_acf': '#4F46E5',
    '09_lead_errors': '#B45309',
    '10_policy': '#059669',
    '11_cost': '#BE185D',
    '12_service': '#0369A1',
    '13_coverage': '#0284C7',
    '14_cv_folds': '#C026D3',
    '15_dm_test': '#0F766E',
    '16_feature_importance': '#7E22CE',
    '17_stl_decomposition': '#0E7490',
    '18_stationarity': '#B91C1C',
    '19_distribution': '#15803D',
    '20_demand_acf_pacf': '#6D28D9',
}
BASE_METHOD_ORDER = ['SMA', 'SES', 'Weekday_DR', 'Weekday_Naive', 'Holt', 'Weekday_Ridge']
METHOD_ORDER = (
    BASE_METHOD_ORDER
    + (['Weekday_RF'] if HAS_SKLEARN else [])
    + (['Holt_Winters', 'SARIMA'] if HAS_STATSMODELS else [])
    + ['Ensemble']
)
MAX_OBSERVATIONS = 10000
MAX_PLOT_POINTS = 1000
MAX_BAR_POINTS = 250
WEEKDAY_NAMES = {2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday', 6: 'Friday', 7: 'Saturday', 8: 'Sunday'}
WEEK_LENGTH = 5
DAY_CODES = tuple(range(2, 7))

# Rule-of-thumb constants (labeled in UI, tables, and documentation; configurable via settings where practical)
DEFAULT_MIN_INTERVAL_ERRORS = 30        # Rule of thumb: minimum backtest errors per horizon step to display prediction intervals
DEFAULT_CV_FOLDS = 5                    # Default number of expanding-window cross-validation folds (user selectable 3..10)
MIN_DM_TEST_SAMPLES = 15                # Rule of thumb: minimum test observations for a reliable Diebold-Mariano test
COVERAGE_TOLERANCE_PCT = 10.0           # Rule of thumb: percentage-point tolerance for judging empirical coverage close to nominal
DEFAULT_MAX_MISSING_SHARE_PCT = 10.0    # Rule of thumb: maximum allowed percentage of blank demand values before rejecting the file
DEFAULT_OUTLIER_Z_THRESHOLD = 3.2       # Rule of thumb: causal residual z-score threshold, justified by bundled simulation tests
DEFAULT_FAST_MODE_ROW_THRESHOLD = 500   # Rule of thumb: observation count above which Fast mode turns on by default
STL_SEASONAL_PERIOD = 5                 # 5-day Monday-Friday operating week seasonal cycle
OUTLIER_WARMUP = 25                     # Rule of thumb: observations used before any outlier can be flagged
OUTLIER_FIT_WINDOW = 100                # Rule of thumb: recent observations used for causal trend/weekday fitting
OUTLIER_SCALE_WINDOW = 200              # Rule of thumb: prior one-step residuals used for robust scale
DEMAND_ACF_MAX_LAG = 20                 # Maximum lag for demand ACF and PACF exploration
RF_PERM_REPEATS = 5                     # Validation permutation importance repeats
ENSEMBLE_TOP_K = 3                      # Equal-weight ensemble averages top 3 models by validation MAE
RIDGE_LAMBDA_GRID = [float(x) for x in np.logspace(-3, 3, 10)]  # At most 10 log-spaced penalty values
RF_PARAM_GRID = [
    dict(n_estimators=ne, max_depth=md, min_samples_leaf=msl, random_state=42)
    for ne in (50, 100)
    for md in (3, 6)
    for msl in (1, 3)
]  # At most 8 Random Forest settings
FIXED_RF_PARAMS = dict(n_estimators=50, max_depth=5, min_samples_leaf=2, random_state=42)
FIXED_RIDGE_LAMBDA = 1.0
SARIMA_CANDIDATES = [
    ((p, d, q), (P, 0, Q, STL_SEASONAL_PERIOD))
    for p in (0, 1)
    for d in (0, 1)
    for q in (0, 1)
    for (P, Q) in ((0, 0), (1, 1))
]  # At most 16 SARIMA candidates


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def score_with_scale(actual, forecast, naive_scale=None):
    """Compute MAE, RMSE, WAPE_pct, Bias, Median/IQR/SD AE, plus MASE and sMAPE_pct (excluding 0/0 pairs)."""
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    if actual.shape != forecast.shape or actual.size == 0:
        raise ValueError('Metric arrays must be nonempty and have equal shape')
    error = actual - forecast
    absolute = np.abs(error)
    denom = np.abs(actual).sum()
    mae = float(absolute.mean())
    mase = float(mae / naive_scale) if (naive_scale is not None and math.isfinite(naive_scale) and naive_scale > 1e-12) else np.nan
    pair_denom = np.abs(actual) + np.abs(forecast)
    nonzero_mask = pair_denom > 0.0
    zero_excluded = int(np.sum(~nonzero_mask))
    if np.any(nonzero_mask):
        smape_pct = float(np.mean(200.0 * absolute[nonzero_mask] / pair_denom[nonzero_mask]))
    else:
        smape_pct = np.nan
    bias_val = float(-error.mean())
    bias_pct = float(100.0 * (-error.sum()) / denom) if denom else np.nan
    tracking_signal = float(np.sum(-error) / mae) if mae > 1e-12 else 0.0
    return dict(
        N=len(actual),
        MAE=mae,
        RMSE=float(np.sqrt(np.mean(error ** 2))),
        WAPE_pct=float(100.0 * absolute.sum() / denom) if denom else np.nan,
        MASE=mase,
        sMAPE_pct=smape_pct,
        sMAPE_zero_excluded=zero_excluded,
        Bias=bias_val,
        Bias_pct=bias_pct,
        Tracking_Signal=tracking_signal,
        Median_AE=float(np.median(absolute)),
        Q1_AE=float(np.quantile(absolute, .25)),
        Q3_AE=float(np.quantile(absolute, .75)),
        SD_AE=float(np.std(absolute, ddof=1)) if len(actual) > 1 else 0.,
    )


def diebold_mariano_pair(e_model, e_bench, model_key, bench_key, loss='absolute', min_samples=MIN_DM_TEST_SAMPLES):
    """Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) small-sample correction (h=1, two-sided Student-t)."""
    e1 = np.asarray(e_model, dtype=float)
    e2 = np.asarray(e_bench, dtype=float)
    T = len(e1)
    if loss == 'squared':
        d = (e1 ** 2) - (e2 ** 2)
        loss_label = 'Squared error (MSE)'
    else:
        d = np.abs(e1) - np.abs(e2)
        loss_label = 'Absolute error (MAE)'
    mean_d = float(np.mean(d))
    short_note = (
        f" (Rule-of-thumb note: test length T={T} is below {min_samples} observations; interpret p-value cautiously.)"
        if T < min_samples else
        (f" (Small-sample HLN correction applied for T={T} test observations.)" if T < 30 else "")
    )
    if model_key == bench_key or np.allclose(d, 0.0, atol=1e-12):
        return dict(
            Model=model_key,
            Model_Label=LABELS.get(model_key, model_key),
            Benchmark=bench_key,
            Benchmark_Label=LABELS.get(bench_key, bench_key),
            Loss=loss_label,
            Test_N=T,
            Mean_Loss_Diff=0.0,
            DM_HLN_Stat=0.0,
            p_value=1.0,
            Better_Model='Tie (identical)',
            Significant_5pct=False,
            Conclusion=f"{LABELS.get(model_key, model_key)} and {LABELS.get(bench_key, bench_key)} have identical test errors.{short_note}",
        )
    gamma0 = float(np.mean((d - mean_d) ** 2))
    if gamma0 <= 1e-20 or T < 2:
        return dict(
            Model=model_key,
            Model_Label=LABELS.get(model_key, model_key),
            Benchmark=bench_key,
            Benchmark_Label=LABELS.get(bench_key, bench_key),
            Loss=loss_label,
            Test_N=T,
            Mean_Loss_Diff=mean_d,
            DM_HLN_Stat=np.nan,
            p_value=np.nan,
            Better_Model=LABELS.get(model_key, model_key) if mean_d < 0 else LABELS.get(bench_key, bench_key),
            Significant_5pct=False,
            Conclusion=f"Loss differences have near-zero variance; DM statistic is undefined.{short_note}",
        )
    dm_raw = mean_d / math.sqrt(gamma0 / T)
    hln_factor = math.sqrt((T + 1.0 - 2.0 * 1.0 + 1.0 * 0.0 / T) / T)
    dm_hln = float(dm_raw * hln_factor)
    p_val = float(2.0 * student_t.sf(abs(dm_hln), df=T - 1))
    better = LABELS.get(model_key, model_key) if mean_d < 0 else LABELS.get(bench_key, bench_key)
    sig = bool(p_val < 0.05)
    if sig:
        conclusion = (
            f"{better} is significantly more accurate at the 5% level "
            f"(HLN-corrected DM = {dm_hln:+.3f}, two-sided p = {p_val:.4f} < 0.05).{short_note}"
        )
    else:
        conclusion = (
            f"{better} has lower sample {loss_label.lower()}, but the difference is not statistically significant "
            f"at the 5% level (HLN-corrected DM = {dm_hln:+.3f}, two-sided p = {p_val:.4f} >= 0.05).{short_note}"
        )
    return dict(
        Model=model_key,
        Model_Label=LABELS.get(model_key, model_key),
        Benchmark=bench_key,
        Benchmark_Label=LABELS.get(bench_key, bench_key),
        Loss=loss_label,
        Test_N=T,
        Mean_Loss_Diff=mean_d,
        DM_HLN_Stat=dm_hln,
        p_value=p_val,
        Better_Model=better,
        Significant_5pct=sig,
        Conclusion=conclusion,
    )


def fit_ses(values):
    """Same objective as the report, with the recurrence evaluated in compiled code."""
    y = np.asarray(values, dtype=float)
    n = len(y)
    ar = np.arange(n, dtype=float)
    y_pad = np.r_[0., y[:-1]]
    def profile(alpha):
        if alpha <= 0.:
            weight = np.ones(n)
            part = np.zeros(n)
        elif alpha >= 1.:
            weight = np.zeros(n); weight[0] = 1.
            part = y_pad
        else:
            b = 1. - alpha
            m = min(n, int(-700. / math.log(b)) + 1)
            weight = np.zeros(n)
            weight[:m] = b ** ar[:m]
            part = lfilter([alpha], [1., -b], y_pad)
        r = y - part
        ww = float(weight @ weight)
        wr = float(weight @ r)
        initial = max(0., wr / ww)
        error = r - weight * initial
        return float(error @ error), initial
    grid = np.linspace(0, 1, 101)
    idx = min(range(101), key=lambda i: profile(grid[i])[0])
    lo, hi = grid[max(0, idx-1)], grid[min(100, idx+1)]
    opt = minimize_scalar(lambda a: profile(a)[0], bounds=(lo, hi),
                          method='bounded', options={'xatol': 1e-10})
    if not opt.success:
        raise ValueError('Exponential smoothing could not converge. Check the demand values.')
    alpha = min([0., 1., float(opt.x)], key=lambda a: profile(a)[0])
    sse, initial = profile(alpha)
    level = initial
    for actual in y:
        level = alpha*actual+(1-alpha)*level
    return dict(alpha=alpha, initial=initial, level=float(level), SSE=sse)


def fit_holt(values):
    """Holt's linear trend method with level and trend smoothing parameters estimated on history."""
    y = np.asarray(values, dtype=float)
    n = len(y)
    l0 = float(np.mean(y[:min(5, n)]))
    b0 = float((np.mean(y[5:min(10, n)]) - l0) / 5.0) if n >= 10 else 0.0
    alphas = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
    betas = [0.01, 0.03, 0.08, 0.15, 0.3]
    best_sse = float('inf')
    best_state = (0.1, 0.03, l0, b0)
    for alpha in alphas:
        for beta in betas:
            a_poly = [1.0, -(2.0 - alpha - alpha * beta), (1.0 - alpha)]
            b_sub = [alpha * (1.0 + beta), -alpha]
            yhat0 = l0 + b0
            e0 = y[0] - yhat0
            l_0 = yhat0 + alpha * e0
            b_0 = b0 + alpha * beta * e0
            if n == 1:
                sse = float(e0 * e0)
                if sse < best_sse:
                    best_sse = sse
                    best_state = (alpha, beta, float(l_0), float(b_0))
                continue
            yhat1 = l_0 + b_0
            zi = lfiltic(b_sub, a_poly, y=[yhat1, yhat0], x=[y[0]])
            yhat_rest, _ = lfilter(b_sub, a_poly, y[1:], zi=zi)
            yhat = np.r_[yhat0, yhat1, yhat_rest[:-1]]
            err = y - yhat
            sse = float(err @ err)
            if sse < best_sse:
                best_sse = sse
                tr_end = float(b0 + (alpha * beta) * np.sum(err))
                lv_end = float(yhat[-1] + alpha * err[-1])
                best_state = (alpha, beta, lv_end, tr_end)
    alpha, beta, level, trend = best_state
    return dict(alpha=alpha, beta=beta, level=level, trend=trend, SSE=best_sse)


def advance_business_days(start_dt, business_days):
    """Advance by the active five- or seven-day operating calendar."""
    if WEEK_LENGTH == 7:
        return start_dt + timedelta(days=int(business_days))
    cur = start_dt
    while cur.weekday() > 4:
        cur += timedelta(days=1)
    remaining = int(business_days)
    while remaining > 0:
        cur += timedelta(days=1)
        if cur.weekday() <= 4:
            remaining -= 1
    return cur


def generate_business_dates(start_dt, count):
    """Return dates from the active five- or seven-day operating calendar."""
    if WEEK_LENGTH == 7:
        return [start_dt + timedelta(days=i) for i in range(int(count))]
    cur = start_dt
    while cur.weekday() > 4:
        cur += timedelta(days=1)
    out = []
    while len(out) < count:
        if cur.weekday() <= 4:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def is_month_end_bday(dt):
    """Return 1.0 if dt is the final active operating day of its month."""
    nxt = advance_business_days(dt, 1)
    return 1.0 if nxt.month != dt.month else 0.0


def rich_feature_names(has_real_dates=False):
    base = [
        'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5',
        'Roll_Mean_5', 'Roll_Mean_10',
        *(WEEKDAY_NAMES[d][:3] for d in DAY_CODES[:-1]),
        'Trend',
    ]
    if has_real_dates:
        base.extend(['Week_of_Month', 'Month_End'])
    return base


def build_rich_feature_row(hist_y, target_day, target_idx, target_date=None, has_real_dates=False):
    """
    Build the richer feature vector for step target_idx (0-based) strictly from prior observations hist_y[:target_idx].
    Never accesses hist_y[target_idx] or any future observation.
    """
    n_hist = len(hist_y)
    l1 = float(hist_y[-1])
    l2 = float(hist_y[-2]) if n_hist >= 2 else l1
    l3 = float(hist_y[-3]) if n_hist >= 3 else l2
    l4 = float(hist_y[-4]) if n_hist >= 4 else l3
    l5 = float(hist_y[-5]) if n_hist >= 5 else l4
    rm5 = float(np.mean(hist_y[-min(5, n_hist):]))
    rm10 = float(np.mean(hist_y[-min(10, n_hist):]))
    d = int(target_day)
    row = [
        l1, l2, l3, l4, l5,
        rm5, rm10,
        *(float(d == code) for code in DAY_CODES[:-1]),
        float(target_idx + 1),
    ]
    if has_real_dates:
        if target_date is not None:
            wom = float((target_date.day - 1) // 7 + 1)
            me = is_month_end_bday(target_date)
        else:
            wom = 1.0
            me = 0.0
        row.extend([wom, me])
    return np.asarray(row, dtype=float)


def build_rich_design_matrix(y_sub, days_sub, dates_sub=None, has_real_dates=False):
    """Build (X, target, keep_indices) on y_sub using only history prior to each row t."""
    y_arr = np.asarray(y_sub, dtype=float)
    d_arr = np.asarray(days_sub, dtype=int)
    n = len(y_arr)
    start_t = 10 if n >= 16 else max(1, min(5, n - 5))
    keep = [
        t for t in range(start_t, n)
        if int(d_arr[t]) == ((int(d_arr[t - 1]) - 1) % WEEK_LENGTH + 2)
    ]
    if len(keep) < 4:
        keep = list(range(start_t, n))
    rows = []
    targets = []
    for t in keep:
        dt_t = dates_sub[t] if (has_real_dates and dates_sub is not None and t < len(dates_sub)) else None
        rows.append(build_rich_feature_row(y_arr[:t], d_arr[t], t, target_date=dt_t, has_real_dates=has_real_dates))
        targets.append(float(y_arr[t]))
    p_cols = len(rich_feature_names(has_real_dates))
    X = np.vstack(rows) if rows else np.empty((0, p_cols), dtype=float)
    target = np.asarray(targets, dtype=float)
    return X, target, np.asarray(keep, dtype=int)


def fit_standardized_ridge(X, target, lam=1.0):
    """
    Fit L2-penalized Ridge regression on standardized predictors Z = (X - mu) / sd with unpenalized intercept.
    Returns dict(intercept, beta_std, mu, sd, raw_beta).
    """
    if len(X) == 0:
        p = X.shape[1]
        return dict(intercept=0.0, beta_std=np.zeros(p), mu=np.zeros(p), sd=np.ones(p))
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd_safe
    y_mean = float(np.mean(target))
    y_cent = target - y_mean
    p = Z.shape[1]
    A = Z.T @ Z + float(lam) * np.eye(p)
    beta_std = np.linalg.solve(A, Z.T @ y_cent)
    return dict(intercept=y_mean, beta_std=beta_std, mu=mu, sd=sd_safe, lam=float(lam))


def predict_standardized_ridge_row(ridge_model, x_row):
    z = (x_row - ridge_model['mu']) / ridge_model['sd']
    return max(0.0, float(ridge_model['intercept'] + z @ ridge_model['beta_std']))


def predict_forest_row(rf_model, x_row):
    """Predict one row from a fitted sklearn forest without repeated validation/joblib overhead."""
    estimators = getattr(rf_model, 'estimators_', None)
    if not estimators:
        return float(rf_model.predict(np.asarray(x_row, dtype=float)[None, :])[0])
    row = np.asarray(x_row, dtype=float)
    total = 0.0
    for estimator in estimators:
        tree = estimator.tree_
        node = 0
        while tree.children_left[node] != tree.children_right[node]:
            feature_idx = tree.feature[node]
            node = tree.children_left[node] if row[feature_idx] <= tree.threshold[node] else tree.children_right[node]
        total += float(tree.value[node].reshape(-1)[0])
    return total / len(estimators)


def recursive_rich_predict(y_sub, days_sub, future_days, dates_sub=None, has_real_dates=False, predictor_fn=None):
    """Recursively predict future_days from origin len(y_sub) using only history available at the origin."""
    hist_y = list(np.asarray(y_sub, dtype=float))
    prev_day = int(days_sub[-1])
    cur_date = dates_sub[-1] if (has_real_dates and dates_sub is not None and len(dates_sub) > 0) else None
    output = []
    for day in future_days:
        day = int(day)
        count = (day - prev_day) % WEEK_LENGTH or WEEK_LENGTH
        for _ in range(count):
            prev_day = (prev_day - 1) % WEEK_LENGTH + 2
            if has_real_dates and cur_date is not None:
                cur_date = advance_business_days(cur_date, 1)
            x_row = build_rich_feature_row(
                hist_y, prev_day, len(hist_y),
                target_date=cur_date, has_real_dates=has_real_dates
            )
            pred_val = max(0.0, float(predictor_fn(x_row)))
            hist_y.append(pred_val)
        output.append(hist_y[-1])
    return np.asarray(output, dtype=float)


def compute_stl_decomposition(y, period=STL_SEASONAL_PERIOD):
    """
    Compute STL decomposition (Observed = Trend + Seasonal + Residual) with period=5.
    Uses statsmodels.tsa.seasonal.STL when available, with a deterministic robust Loess/seasonal fallback.
    Also computes the Iglewicz-Hoaglin modified z-score on the STL residuals.
    """
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)
    if HAS_STATSMODELS and SMSTL is not None and n >= 2 * period:
        res = SMSTL(y_arr, period=period, robust=True).fit()
        trend = np.asarray(res.trend, dtype=float)
        seasonal = np.asarray(res.seasonal, dtype=float)
        resid = np.asarray(res.resid, dtype=float)
    else:
        # Robust Loess + period-5 seasonal-trend decomposition
        trend_init = np.zeros(n, dtype=float)
        half_w = max(2, period // 2)
        for i in range(n):
            lo = max(0, i - half_w)
            hi = min(n, i + half_w + 1)
            trend_init[i] = float(np.median(y_arr[lo:hi]))
        detrended = y_arr - trend_init
        phase_meds = np.zeros(period, dtype=float)
        for ph in range(period):
            vals = detrended[ph::period]
            phase_meds[ph] = float(np.median(vals)) if len(vals) else 0.0
        phase_meds -= float(np.mean(phase_meds))
        seasonal = np.array([phase_meds[i % period] for i in range(n)], dtype=float)
        deseason = y_arr - seasonal
        trend = np.zeros(n, dtype=float)
        bw = max(5, min(11, n // 4))
        x_idx = np.arange(n, dtype=float)
        for i in range(n):
            dist = np.abs(x_idx - i)
            order = np.argsort(dist)[:bw]
            max_d = max(float(dist[order[-1]]), 1.0)
            u = np.clip(dist[order] / (max_d * 1.05), 0.0, 1.0)
            w = (1.0 - u ** 3) ** 3
            X_loc = np.column_stack([np.ones(len(order)), x_idx[order] - i])
            XW = X_loc * w[:, None]
            beta_loc = np.linalg.lstsq(XW, deseason[order] * w, rcond=None)[0]
            trend[i] = float(beta_loc[0])
        resid = y_arr - trend - seasonal

    fitted = trend + seasonal
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    if mad > 1e-12:
        mod_z = 0.6745 * (resid - med) / mad
    else:
        sd_r = float(np.std(resid, ddof=1)) if n > 1 else 1.0
        mod_z = (resid - med) / sd_r if sd_r > 1e-12 else np.zeros(n, dtype=float)
    return dict(
        observed=y_arr,
        trend=trend,
        seasonal=seasonal,
        resid=resid,
        fitted=fitted,
        med_resid=med,
        mad_resid=mad,
        mod_z=mod_z,
    )


def apply_causal_outlier_treatment(y_raw, train_end, threshold=DEFAULT_OUTLIER_Z_THRESHOLD, treatment='keep', excluded_indices=None):
    """
    Flag outliers with one causal one-step trend + weekday rule for every scored observation.
    The fit for observation i uses only observations before i. Its robust center and scale use only
    earlier residuals produced by the same rule. The first OUTLIER_WARMUP observations are never flagged.
    """
    y_arr = np.asarray(y_raw, dtype=float)
    n = len(y_arr)
    causal_trend = np.zeros(n, dtype=float)
    causal_seasonal = np.zeros(n, dtype=float)
    causal_resid = np.full(n, np.nan, dtype=float)
    causal_mod_z = np.zeros(n, dtype=float)
    flagged_mask = np.zeros(n, dtype=bool)
    detection_y = y_arr.copy()
    excluded = {int(i) for i in (excluded_indices or [])}

    def design(indices):
        idx = np.asarray(indices, dtype=float)
        phase = np.asarray(indices, dtype=int) % STL_SEASONAL_PERIOD
        return np.column_stack([
            np.ones(len(idx)), idx,
            *(phase == p for p in range(STL_SEASONAL_PERIOD - 1)),
        ]).astype(float)

    for i in range(10, n):
        start = max(0, i - OUTLIER_FIT_WINDOW)
        hist_idx = np.arange(start, i)
        X = design(hist_idx)
        target = detection_y[hist_idx]
        beta = np.linalg.lstsq(X, target, rcond=None)[0]
        fit_errors = target - X @ beta
        fit_center = float(np.median(fit_errors))
        fit_mad = float(np.median(np.abs(fit_errors - fit_center)))
        if fit_mad > 1e-12:
            robust_keep = np.abs(fit_errors - fit_center) <= 3.5 * fit_mad / 0.6745
            if int(np.sum(robust_keep)) >= X.shape[1] + 2:
                beta = np.linalg.lstsq(X[robust_keep], target[robust_keep], rcond=None)[0]
        fit_i = float(design([i])[0] @ beta)
        trend_i = float(beta[0] + beta[1] * i)
        res_i = float(y_arr[i] - fit_i)
        causal_trend[i] = trend_i
        causal_seasonal[i] = fit_i - trend_i
        causal_resid[i] = res_i
        if i in excluded:
            causal_resid[i] = np.nan
            causal_mod_z[i] = 0.0
            detection_y[i] = max(0.0, fit_i)
            continue
        prior_res = causal_resid[max(10, i - OUTLIER_SCALE_WINDOW):i]
        prior_res = prior_res[np.isfinite(prior_res)]
        if len(prior_res) >= 10:
            med_i = float(np.median(prior_res))
            mad_i = float(np.median(np.abs(prior_res - med_i)))
            scale_i = (mad_i / 0.6745) if mad_i > 1e-12 else max(float(np.std(prior_res, ddof=1)), 1e-9)
            causal_mod_z[i] = (res_i - med_i) / scale_i
            if i >= OUTLIER_WARMUP and abs(causal_mod_z[i]) > float(threshold):
                flagged_mask[i] = True
                detection_y[i] = max(0.0, fit_i)
    y_treated = y_arr.copy()
    records = []
    for i in range(n):
        if not flagged_mask[i]:
            continue
        fit_val = max(0.0, float(causal_trend[i] + causal_seasonal[i]))
        prior_res = causal_resid[max(10, i - OUTLIER_SCALE_WINDOW):i]
        prior_res = prior_res[np.isfinite(prior_res)]
        med_i = float(np.median(prior_res))
        mad_i = float(np.median(np.abs(prior_res - med_i)))
        scale_i = (mad_i / 0.6745) if mad_i > 1e-12 else max(float(np.std(prior_res, ddof=1)), 1e-9)
        max_dev = float(threshold) * scale_i
        sign_r = 1.0 if causal_resid[i] >= med_i else -1.0
        cap_val = max(0.0, float(fit_val + med_i + sign_r * max_dev))
        if treatment == 'cap':
            new_val = cap_val
        elif treatment == 'replace':
            new_val = fit_val
        else:
            new_val = float(y_arr[i])
        y_treated[i] = new_val
        records.append(dict(
            Observation=i + 1,
            Raw_Demand=float(y_arr[i]),
            STL_Fitted=fit_val,
            STL_Residual=float(causal_resid[i]),
            Modified_Z_Score=float(causal_mod_z[i]),
            Threshold=float(threshold),
            Treatment_Action=treatment,
            Treated_Demand=float(new_val),
        ))
    return dict(
        y_treated=y_treated,
        flagged_mask=flagged_mask,
        outlier_count=int(np.sum(flagged_mask)),
        outlier_positions=[int(i + 1) for i in np.flatnonzero(flagged_mask)],
        outlier_records=records,
        causal_trend=causal_trend,
        causal_seasonal=causal_seasonal,
        causal_resid=causal_resid,
        causal_mod_z=causal_mod_z,
    )


def _adf_fallback(series):
    """Augmented Dickey-Fuller test (with constant, AIC lag selection on common sample) when statsmodels is absent."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    dx = np.diff(x)
    max_p = min(int(12.0 * ((n / 100.0) ** 0.25)), max(0, (n - 6) // 4))
    best_aic = float('inf')
    best_p = 0
    # Evaluate all candidate lags p on the common sample starting at max_p (matching statsmodels autolag='AIC')
    y_common = dx[max_p:]
    T_common = len(y_common)
    for p in range(max_p + 1):
        cols = [np.ones(T_common), x[max_p:n - 1]]
        for j in range(1, p + 1):
            cols.append(dx[max_p - j:n - 1 - j])
        X_c = np.column_stack(cols)
        beta_c, _, _, _ = np.linalg.lstsq(X_c, y_common, rcond=None)
        resid_c = y_common - X_c @ beta_c
        sse_c = float(resid_c @ resid_c)
        aic = T_common * math.log(max(sse_c / T_common, 1e-18)) + 2.0 * X_c.shape[1]
        if aic < best_aic:
            best_aic = aic
            best_p = p
    # Compute final ADF statistic on the full available sample for best_p
    y_dep = dx[best_p:]
    T_eff = len(y_dep)
    cols = [np.ones(T_eff), x[best_p:n - 1]]
    for j in range(1, best_p + 1):
        cols.append(dx[best_p - j:n - 1 - j])
    X = np.column_stack(cols)
    beta, _, _, _ = np.linalg.lstsq(X, y_dep, rcond=None)
    resid = y_dep - X @ beta
    sse = float(resid @ resid)
    sigma2 = sse / max(1, T_eff - X.shape[1])
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se_1 = math.sqrt(max(float(cov[1, 1]), 1e-18))
    best_stat = float(beta[1] / se_1)
    crit_5 = -2.86154 - 2.732 / max(n, 10)
    # Approximate MacKinnon p-value interpolation for constant-only ADF
    tau_grid = np.array([-4.5, -3.96, -3.43, -3.12, -2.86, -2.57, -1.95, -1.0, 0.0, 1.0])
    p_grid = np.array([0.0002, 0.001, 0.01, 0.025, 0.05, 0.10, 0.30, 0.75, 0.95, 0.99])
    p_val = float(np.interp(best_stat, tau_grid, p_grid))
    return best_stat, p_val, best_p, crit_5


def _kpss_fallback(series):
    """KPSS stationarity test (level stationarity 'c', Bartlett kernel) when statsmodels is absent."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    e = x - float(np.mean(x))
    S = np.cumsum(e)
    eta = float(np.sum(S ** 2)) / (n ** 2)
    nlags = min(n - 1, max(1, int(4.0 * ((n / 100.0) ** 0.25))))
    gamma0 = float(np.mean(e ** 2))
    s2 = gamma0
    for l in range(1, nlags + 1):
        w = 1.0 - l / (nlags + 1.0)
        cov_l = float(np.sum(e[l:] * e[:-l])) / n
        s2 += 2.0 * w * cov_l
    kpss_stat = eta / max(s2, 1e-18)
    crit_5 = 0.463
    stat_grid = np.array([0.0, 0.347, 0.463, 0.574, 0.739, 1.5])
    p_grid = np.array([0.10, 0.10, 0.05, 0.025, 0.01, 0.01])
    p_val = float(np.interp(kpss_stat, stat_grid, p_grid))
    return kpss_stat, p_val, nlags, crit_5


def compute_stationarity_tests(y, data_label='Demand'):
    """Run ADF and KPSS tests on demand level and its first difference, with a combined plain-language verdict."""
    y_arr = np.asarray(y, dtype=float)
    dy_arr = np.diff(y_arr)
    rows = []
    verdicts = {}
    for s_key, s_label, arr in [
        ('level', f'{data_label} (level)', y_arr),
        ('diff1', f'First difference of {data_label.lower()} (Δy)', dy_arr),
    ]:
        if HAS_STATSMODELS and sm_adfuller is not None and sm_kpss is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                adf_res = sm_adfuller(arr, autolag='AIC')
                adf_stat, adf_p, adf_lags = float(adf_res[0]), float(adf_res[1]), int(adf_res[2])
                adf_crit5 = float(adf_res[4].get('5%', -2.86))
                kpss_res = sm_kpss(arr, regression='c', nlags='auto')
                kpss_stat, kpss_p, kpss_lags = float(kpss_res[0]), float(kpss_res[1]), int(kpss_res[2])
                kpss_crit5 = float(kpss_res[3].get('5%', 0.463))
        else:
            adf_stat, adf_p, adf_lags, adf_crit5 = _adf_fallback(arr)
            kpss_stat, kpss_p, kpss_lags, kpss_crit5 = _kpss_fallback(arr)

        adf_rejects = bool(adf_p < 0.05)
        kpss_rejects = bool(kpss_p < 0.05)
        if adf_rejects and not kpss_rejects:
            verdict = "Stationary: ADF rejects a unit root (p < 0.05) and KPSS does not reject stationarity (p >= 0.05)."
            short_verdict = "Stationary"
        elif (not adf_rejects) and kpss_rejects:
            verdict = "Non-stationary (unit root): ADF fails to reject a unit root (p >= 0.05) and KPSS rejects stationarity (p < 0.05); differencing is recommended."
            short_verdict = "Non-stationary (unit root)"
        elif adf_rejects and kpss_rejects:
            verdict = "Trend-stationary / regime shift: ADF rejects a unit root (p < 0.05) while KPSS rejects level stationarity (p < 0.05)."
            short_verdict = "Trend-stationary / shift"
        else:
            verdict = "Inconclusive / weak evidence: neither ADF nor KPSS rejects at the 5% level on this sample size."
            short_verdict = "Inconclusive"

        verdicts[s_key] = dict(
            series_label=s_label,
            adf_stat=adf_stat, adf_p=adf_p, adf_lags=adf_lags, adf_crit5=adf_crit5, adf_rejects=adf_rejects,
            kpss_stat=kpss_stat, kpss_p=kpss_p, kpss_lags=kpss_lags, kpss_crit5=kpss_crit5, kpss_rejects=kpss_rejects,
            verdict=verdict, short_verdict=short_verdict,
        )
        rows.append(dict(
            Series=s_label,
            Test='ADF (Augmented Dickey-Fuller)',
            Statistic=adf_stat,
            p_value=adf_p,
            Lags_Used=adf_lags,
            Critical_5pct=adf_crit5,
            Null_Hypothesis='Unit root (non-stationary)',
            Test_Decision_5pct='Reject unit root (stationary)' if adf_rejects else 'Fail to reject unit root',
            Combined_Verdict=verdict,
        ))
        rows.append(dict(
            Series=s_label,
            Test='KPSS (Kwiatkowski-Phillips-Schmidt-Shin)',
            Statistic=kpss_stat,
            p_value=kpss_p,
            Lags_Used=kpss_lags,
            Critical_5pct=kpss_crit5,
            Null_Hypothesis='Level stationary',
            Test_Decision_5pct='Reject stationarity (non-stationary)' if kpss_rejects else 'Fail to reject stationarity (stationary)',
            Combined_Verdict=verdict,
        ))
    combined_summary = (
        f"Level series is {verdicts['level']['short_verdict'].lower()} "
        f"(ADF stat = {verdicts['level']['adf_stat']:+.3f}, p = {verdicts['level']['adf_p']:.3f}; "
        f"KPSS stat = {verdicts['level']['kpss_stat']:.3f}, p = {verdicts['level']['kpss_p']:.3f}). "
        f"After first differencing, Δy is {verdicts['diff1']['short_verdict'].lower()} "
        f"(ADF stat = {verdicts['diff1']['adf_stat']:+.3f}, p = {verdicts['diff1']['adf_p']:.3f}; "
        f"KPSS stat = {verdicts['diff1']['kpss_stat']:.3f}, p = {verdicts['diff1']['kpss_p']:.3f})."
    )
    return pd.DataFrame(rows), verdicts, combined_summary


def compute_demand_acf_pacf(y, max_lag=DEMAND_ACF_MAX_LAG):
    """Compute ACF and PACF of the demand series itself up to lag 20 with 95% confidence bands."""
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)
    nlags = min(int(max_lag), max(1, n // 2 - 1))
    ci_95 = 1.96 / math.sqrt(n)
    if HAS_STATSMODELS and sm_acf is not None and sm_pacf is not None:
        acf_vals = np.asarray(sm_acf(y_arr, nlags=nlags, fft=False), dtype=float)[1:nlags + 1]
        pacf_vals = np.asarray(sm_pacf(y_arr, nlags=nlags, method='ywm'), dtype=float)[1:nlags + 1]
    else:
        c = y_arr - float(np.mean(y_arr))
        denom = float(c @ c)
        if denom <= 1e-18:
            acf_vals = np.zeros(nlags, dtype=float)
            pacf_vals = np.zeros(nlags, dtype=float)
        else:
            r = np.array([1.0] + [float(c[k:] @ c[:-k]) / denom for k in range(1, nlags + 1)], dtype=float)
            acf_vals = r[1:]
            # Durbin-Levinson recursion for Yule-Walker PACF
            pacf_vals = np.zeros(nlags, dtype=float)
            phi = np.zeros((nlags + 1, nlags + 1), dtype=float)
            phi[1, 1] = r[1]
            pacf_vals[0] = r[1]
            v = max(1.0 - r[1] ** 2, 1e-12)
            for k in range(2, nlags + 1):
                num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
                phi_kk = float(np.clip(num / max(v, 1e-12), -0.999, 0.999))
                phi[k, k] = phi_kk
                for j in range(1, k):
                    phi[k, j] = phi[k - 1, j] - phi_kk * phi[k - 1, k - j]
                v = max(v * (1.0 - phi_kk ** 2), 1e-12)
                pacf_vals[k - 1] = phi_kk

    rows = []
    for k in range(1, nlags + 1):
        ak = float(acf_vals[k - 1])
        pk = float(pacf_vals[k - 1])
        rows.append(dict(
            Lag=k,
            ACF=ak,
            PACF=pk,
            CI_95_Lower=-ci_95,
            CI_95_Upper=ci_95,
            Significant_ACF=bool(abs(ak) > ci_95),
            Significant_PACF=bool(abs(pk) > ci_95),
        ))
    return pd.DataFrame(rows), ci_95


def compute_distribution_summary(y, data_label='Demand', unit_label='units'):
    """Compute histogram bins and distribution moments (mean, median, skewness, kurtosis, SD, IQR)."""
    y_arr = np.asarray(y, dtype=float)
    mean_v = float(np.mean(y_arr))
    med_v = float(np.median(y_arr))
    sd_v = float(np.std(y_arr, ddof=1)) if len(y_arr) > 1 else 0.0
    skew_v = float(scipy_skew(y_arr, bias=False)) if (len(y_arr) > 2 and sd_v > 1e-12) else 0.0
    kurt_v = float(scipy_kurtosis(y_arr, bias=False)) if (len(y_arr) > 3 and sd_v > 1e-12) else 0.0
    q25, q75 = float(np.quantile(y_arr, 0.25)), float(np.quantile(y_arr, 0.75))
    n_bins = min(15, max(6, int(math.sqrt(len(y_arr)))))
    counts, edges = np.histogram(y_arr, bins=n_bins)
    rows = []
    for b_idx in range(len(counts)):
        rows.append(dict(
            Bin=b_idx + 1,
            Bin_Start=float(edges[b_idx]),
            Bin_End=float(edges[b_idx + 1]),
            Count=int(counts[b_idx]),
            Share_pct=float(100.0 * counts[b_idx] / len(y_arr)),
            Series_Mean=mean_v,
            Series_Median=med_v,
            Series_SD=sd_v,
            Series_Skewness=skew_v,
            Series_Kurtosis=kurt_v,
            Series_IQR=q75 - q25,
        ))
    stats_dict = dict(
        mean=mean_v, median=med_v, sd=sd_v, skewness=skew_v, kurtosis=kurt_v,
        q25=q25, q75=q75, iqr=q75 - q25, min=float(np.min(y_arr)), max=float(np.max(y_arr)),
        bins=n_bins, counts=counts.tolist(), edges=edges.tolist(),
    )
    return pd.DataFrame(rows), stats_dict


def fit_holt_winters_state(values, seasonal='add', damped=False, period=STL_SEASONAL_PERIOD):
    """
    Fit Holt-Winters with period=5 via statsmodels.tsa.holtwinters.ExponentialSmoothing
    and return the state dictionary for fast recursive/rolling state updates without refitting inside a block.
    """
    y = np.asarray(values, dtype=float)
    n = len(y)
    use_mul = (seasonal == 'mul') and bool(np.all(y > 0.0))
    seas_mode = 'mul' if use_mul else 'add'
    if HAS_STATSMODELS and SMExponentialSmoothing is not None and n >= 2 * period:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                model = SMExponentialSmoothing(
                    y,
                    trend='add',
                    damped_trend=bool(damped),
                    seasonal=seas_mode,
                    seasonal_periods=period,
                    initialization_method='estimated',
                )
                res = model.fit(optimized=True)
                p = res.params
                alpha = float(p.get('smoothing_level', 0.2))
                beta = float(p.get('smoothing_trend', 0.05))
                gamma = float(p.get('smoothing_seasonal', 0.1))
                phi = float(p.get('damping_trend', 0.9 if damped else 1.0))
                if not math.isfinite(alpha): alpha = 0.2
                if not math.isfinite(beta): beta = 0.05
                if not math.isfinite(gamma): gamma = 0.1
                if not math.isfinite(phi) or not damped: phi = 1.0
                lv = float(res.level[-1])
                tr = float(res.trend[-1])
                # last `period` seasonal components in chronological order (oldest to newest)
                s_tail = [float(x) for x in res.season[-period:]]
                sse = float(np.sum((y - np.asarray(res.fittedvalues, dtype=float)) ** 2))
                return dict(
                    alpha=alpha, beta=beta, gamma=gamma, phi=phi,
                    level=lv, trend=tr, season_tail=s_tail,
                    seasonal=seas_mode, damped=bool(damped), period=period, SSE=sse,
                )
            except Exception:
                pass

    # Deterministic fallback when series is very short or optimization fails
    l0 = float(np.mean(y[:min(period, n)]))
    b0 = float((np.mean(y[period:min(2 * period, n)]) - l0) / period) if n >= 2 * period else 0.0
    if seas_mode == 'mul':
        l0 = max(l0, 1e-3)
        s_init = [max(0.1, float(np.mean(y[ph::period]) / l0)) for ph in range(period)]
        s_mean = float(np.mean(s_init))
        s_init = [v / s_mean for v in s_init]
    else:
        s_init = [float(np.mean(y[ph::period]) - l0) for ph in range(period)]
        s_mean = float(np.mean(s_init))
        s_init = [v - s_mean for v in s_init]
    alpha, beta, gamma = 0.25, 0.05, 0.15
    phi = 0.90 if damped else 1.0
    lv, tr = l0, b0
    s_buf = list(s_init)
    sse = 0.0
    for t in range(n):
        s_old = s_buf[-period]
        if seas_mode == 'mul':
            yhat = (lv + phi * tr) * s_old
            new_lv = alpha * (y[t] / max(s_old, 1e-6)) + (1.0 - alpha) * (lv + phi * tr)
            new_tr = beta * (new_lv - lv) + (1.0 - beta) * phi * tr
            new_s = gamma * (y[t] / max(lv + phi * tr, 1e-6)) + (1.0 - gamma) * s_old
        else:
            yhat = lv + phi * tr + s_old
            new_lv = alpha * (y[t] - s_old) + (1.0 - alpha) * (lv + phi * tr)
            new_tr = beta * (new_lv - lv) + (1.0 - beta) * phi * tr
            new_s = gamma * (y[t] - lv - phi * tr) + (1.0 - gamma) * s_old
        sse += float((y[t] - yhat) ** 2)
        lv, tr = float(new_lv), float(new_tr)
        s_buf.append(float(new_s))
    return dict(
        alpha=alpha, beta=beta, gamma=gamma, phi=phi,
        level=lv, trend=tr, season_tail=s_buf[-period:],
        seasonal=seas_mode, damped=bool(damped), period=period, SSE=sse,
    )


def hw_forecast_from_state(hw_state, steps):
    """Forecast `steps` ahead from a Holt-Winters state dict."""
    lv = hw_state['level']
    tr = hw_state['trend']
    phi = hw_state['phi']
    damped = hw_state['damped']
    seas_mode = hw_state['seasonal']
    period = hw_state['period']
    s_tail = hw_state['season_tail']
    out = []
    phi_sum = 0.0
    for h in range(1, steps + 1):
        phi_sum = (phi_sum + (phi ** h)) if damped else float(h)
        s_h = float(s_tail[(h - 1) % period])
        if seas_mode == 'mul':
            val = (lv + phi_sum * tr) * s_h
        else:
            val = lv + phi_sum * tr + s_h
        out.append(max(0.0, float(val)))
    return np.asarray(out, dtype=float)


def hw_update_state_with_obs(hw_state, new_obs_iterable):
    """Advance Holt-Winters state across new observations without refitting parameters."""
    lv = float(hw_state['level'])
    tr = float(hw_state['trend'])
    alpha = float(hw_state['alpha'])
    beta = float(hw_state['beta'])
    gamma = float(hw_state['gamma'])
    phi = float(hw_state['phi'])
    seas_mode = hw_state['seasonal']
    period = int(hw_state['period'])
    s_buf = list(hw_state['season_tail'])
    for obs in new_obs_iterable:
        obs = float(obs)
        s_old = s_buf[-period]
        if seas_mode == 'mul':
            denom_s = s_old if abs(s_old) > 1e-6 else 1e-6
            new_lv = alpha * (obs / denom_s) + (1.0 - alpha) * (lv + phi * tr)
            new_tr = beta * (new_lv - lv) + (1.0 - beta) * phi * tr
            denom_l = (lv + phi * tr) if abs(lv + phi * tr) > 1e-6 else 1e-6
            new_s = gamma * (obs / denom_l) + (1.0 - gamma) * s_old
        else:
            new_lv = alpha * (obs - s_old) + (1.0 - alpha) * (lv + phi * tr)
            new_tr = beta * (new_lv - lv) + (1.0 - beta) * phi * tr
            new_s = gamma * (obs - lv - phi * tr) + (1.0 - gamma) * s_old
        lv, tr = float(new_lv), float(new_tr)
        s_buf.append(float(new_s))
    new_state = dict(hw_state)
    new_state['level'] = lv
    new_state['trend'] = tr
    new_state['season_tail'] = s_buf[-period:]
    return new_state


def select_holt_winters_config(y_train, y_val, hw_seasonal_opt='auto', hw_damped_opt='auto', fast_mode=False):
    """
    Select Holt-Winters (seasonal in {'add', 'mul'}, damped in {False, True}) on training & validation split only.
    In Fast mode or when user explicitly sets options, skips searching.
    """
    pos_ok = bool(np.all(y_train > 0.0) and np.all(y_val > 0.0))
    if hw_seasonal_opt in ('add', 'mul'):
        seas_candidates = ['mul' if (hw_seasonal_opt == 'mul' and pos_ok) else 'add']
    elif fast_mode:
        seas_candidates = ['add']
    else:
        seas_candidates = ['add', 'mul'] if pos_ok else ['add']

    if hw_damped_opt in (True, False, 'true', 'false'):
        damped_candidates = [hw_damped_opt in (True, 'true')]
    elif fast_mode:
        damped_candidates = [False]
    else:
        damped_candidates = [False, True]

    best_cfg = (seas_candidates[0], damped_candidates[0])
    best_mae = float('inf')
    rows = []
    for s_mode in seas_candidates:
        for d_flag in damped_candidates:
            st = fit_holt_winters_state(y_train, seasonal=s_mode, damped=d_flag, period=STL_SEASONAL_PERIOD)
            preds = []
            cur_st = st
            for obs in y_val:
                preds.append(float(hw_forecast_from_state(cur_st, 1)[0]))
                cur_st = hw_update_state_with_obs(cur_st, [obs])
            mae = float(np.mean(np.abs(y_val - np.asarray(preds, dtype=float))))
            rows.append(dict(Seasonal=s_mode, Damped_Trend=bool(d_flag), Validation_MAE=mae))
            if mae < best_mae:
                best_mae = mae
                best_cfg = (s_mode, bool(d_flag))
    return best_cfg[0], best_cfg[1], rows


def search_sarima_order_by_aic(y_train):
    """
    Search SARIMA(p,d,q)(P,0,Q)_5 once on the training part over at most 16 candidates by AIC.
    Skips any candidate that fails to converge and reports how many were skipped.
    Never touches validation or test data.
    """
    y_tr = np.asarray(y_train, dtype=float)
    rows = []
    skipped_count = 0
    best_aic = float('inf')
    best_spec = ((1, 0, 0), (0, 0, 0, STL_SEASONAL_PERIOD))
    best_res = None

    for order, seasonal_order in SARIMA_CANDIDATES:
        if not (HAS_STATSMODELS and SMSARIMAX is not None):
            skipped_count += 1
            rows.append(dict(
                Order_pdq=str(order),
                Seasonal_PDQs=str(seasonal_order),
                Converged=False,
                AIC=np.nan,
                BIC=np.nan,
                Selected=False,
                Status='Skipped (statsmodels not loaded)',
            ))
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mod = SMSARIMAX(
                    y_tr,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = mod.fit(disp=False, maxiter=50)
                retvals = getattr(res, 'mle_retvals', {}) or {}
                converged = bool(retvals.get('converged', True))
                aic_val = float(res.aic)
                bic_val = float(getattr(res, 'bic', np.nan))
                if (not converged) or (not math.isfinite(aic_val)):
                    skipped_count += 1
                    rows.append(dict(
                        Order_pdq=str(order),
                        Seasonal_PDQs=str(seasonal_order),
                        Converged=False,
                        AIC=aic_val if math.isfinite(aic_val) else np.nan,
                        BIC=bic_val if math.isfinite(bic_val) else np.nan,
                        Selected=False,
                        Status='Skipped (failed to converge)',
                    ))
                    continue
                rows.append(dict(
                    Order_pdq=str(order),
                    Seasonal_PDQs=str(seasonal_order),
                    Converged=True,
                    AIC=aic_val,
                    BIC=bic_val,
                    Selected=False,
                    Status='Converged',
                ))
                if aic_val < best_aic:
                    best_aic = aic_val
                    best_spec = (order, seasonal_order)
                    best_res = res
        except Exception as exc:
            skipped_count += 1
            rows.append(dict(
                Order_pdq=str(order),
                Seasonal_PDQs=str(seasonal_order),
                Converged=False,
                AIC=np.nan,
                BIC=np.nan,
                Selected=False,
                Status=f'Skipped ({type(exc).__name__})',
            ))

    # Fallback if all 16 failed to converge
    if best_res is None and HAS_STATSMODELS and SMSARIMAX is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                best_res = SMSARIMAX(
                    y_tr, order=(1, 0, 0), seasonal_order=(0, 0, 0, STL_SEASONAL_PERIOD),
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False, maxiter=50)
                best_aic = float(best_res.aic)
            except Exception:
                best_res = None

    for r in rows:
        if r['Converged'] and r['Order_pdq'] == str(best_spec[0]) and r['Seasonal_PDQs'] == str(best_spec[1]):
            r['Selected'] = True
            break

    return dict(
        order=best_spec[0],
        seasonal_order=best_spec[1],
        best_aic=best_aic if math.isfinite(best_aic) else None,
        candidates_tried=len(SARIMA_CANDIDATES),
        skipped_count=skipped_count,
        converged_count=len(SARIMA_CANDIDATES) - skipped_count,
        table_df=pd.DataFrame(rows),
        fitted_train_res=best_res,
    )


def fit_sarima_block(y_hist, order, seasonal_order):
    """Fit chosen SARIMA order once on y_hist at the start of a block."""
    if not (HAS_STATSMODELS and SMSARIMAX is not None):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            mod = SMSARIMAX(
                np.asarray(y_hist, dtype=float),
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            return mod.fit(disp=False, maxiter=50)
        except Exception:
            return None


def sarima_forecast_with_state_update(base_res, y_prefix, steps):
    """
    Update SARIMA state with new observations y_prefix without refitting parameters inside a block,
    then forecast `steps` ahead.
    """
    y_arr = np.asarray(y_prefix, dtype=float)
    if base_res is None:
        return np.full(steps, float(np.mean(y_arr[-min(5, len(y_arr)):])), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            n_base = len(base_res.model.endog)
            if len(y_arr) == n_base:
                fc = np.asarray(base_res.forecast(steps=steps), dtype=float)
            elif len(y_arr) > n_base and hasattr(base_res, 'append'):
                updated = base_res.append(y_arr[n_base:], refit=False)
                fc = np.asarray(updated.forecast(steps=steps), dtype=float)
            elif hasattr(base_res, 'apply'):
                updated = base_res.apply(y_arr, refit=False)
                fc = np.asarray(updated.forecast(steps=steps), dtype=float)
            else:
                fc = np.asarray(base_res.forecast(steps=steps), dtype=float)
            fc = np.where(np.isfinite(fc), np.maximum(0.0, fc), float(np.mean(y_arr[-min(5, len(y_arr)):])))
            return fc
        except Exception:
            return np.full(steps, max(0.0, float(np.mean(y_arr[-min(5, len(y_arr)):]))), dtype=float)


def tune_ridge_and_rf(y, days, train, dev, dates=None, has_real_dates=False, naive_scale=None, fast_mode=False):
    """
    Single tuning pass on the training (y[:train]) and validation (y[train:dev]) split only.
    Never touches test data (y[dev:]). Never repeated at every origin or CV fold.
    - Ridge penalty over log-spaced grid of 10 values (RIDGE_LAMBDA_GRID).
    - Random Forest over 8 settings (RF_PARAM_GRID), random_state=42 fixed.
    In Fast mode, skips grid search and uses FIXED_RIDGE_LAMBDA and FIXED_RF_PARAMS.
    """
    X_tr, t_tr, _ = build_rich_design_matrix(y[:train], days[:train], dates_sub=dates[:train] if dates else None, has_real_dates=has_real_dates)
    val_actual = y[train:dev]
    tuning_rows = []

    # 1. Ridge tuning
    ridge_candidates = [FIXED_RIDGE_LAMBDA] if fast_mode else RIDGE_LAMBDA_GRID
    best_ridge_lam = ridge_candidates[0]
    best_ridge_mae = float('inf')
    ridge_scores = []
    for lam in ridge_candidates:
        r_mod = fit_standardized_ridge(X_tr, t_tr, lam=lam)
        preds = []
        for i in range(train, dev):
            dt_i = dates[:i] if dates else None
            fc_1 = recursive_rich_predict(
                y[:i], days[:i], [days[i]],
                dates_sub=dt_i, has_real_dates=has_real_dates,
                predictor_fn=lambda x_row, m=r_mod: predict_standardized_ridge_row(m, x_row),
            )[0]
            preds.append(fc_1)
        sc = score_with_scale(val_actual, preds, naive_scale=naive_scale)
        ridge_scores.append((lam, sc))
        if sc['MAE'] < best_ridge_mae:
            best_ridge_mae = sc['MAE']
            best_ridge_lam = float(lam)

    for lam, sc in ridge_scores:
        tuning_rows.append(dict(
            Model='Weekday_Ridge',
            Model_Label=LABELS['Weekday_Ridge'],
            Setting_Description=f'alpha/lambda = {lam:.6g}' + (' (Fast mode fixed)' if fast_mode else ''),
            ridge_lambda=float(lam),
            n_estimators=None,
            max_depth=None,
            min_samples_leaf=None,
            random_state=None,
            Validation_MAE=sc['MAE'],
            Validation_RMSE=sc['RMSE'],
            Validation_WAPE_pct=sc['WAPE_pct'],
            Validation_MASE=sc['MASE'],
            Validation_sMAPE_pct=sc['sMAPE_pct'],
            Selected=bool(abs(lam - best_ridge_lam) < 1e-12),
        ))

    # 2. Random Forest tuning
    best_rf_params = dict(FIXED_RF_PARAMS)
    if HAS_SKLEARN and RandomForestRegressor is not None:
        rf_candidates = [FIXED_RF_PARAMS] if fast_mode else RF_PARAM_GRID
        best_rf_mae = float('inf')
        rf_scores = []
        for params in rf_candidates:
            rf = RandomForestRegressor(**params)
            rf.fit(X_tr, t_tr)
            preds = []
            for i in range(train, dev):
                dt_i = dates[:i] if dates else None
                fc_1 = recursive_rich_predict(
                    y[:i], days[:i], [days[i]],
                    dates_sub=dt_i, has_real_dates=has_real_dates,
                    predictor_fn=lambda x_row, m=rf: float(m.predict(x_row[None, :])[0]),
                )[0]
                preds.append(fc_1)
            sc = score_with_scale(val_actual, preds, naive_scale=naive_scale)
            rf_scores.append((params, sc))
            if sc['MAE'] < best_rf_mae:
                best_rf_mae = sc['MAE']
                best_rf_params = dict(params)

        for params, sc in rf_scores:
            is_sel = (
                params['n_estimators'] == best_rf_params['n_estimators']
                and params['max_depth'] == best_rf_params['max_depth']
                and params['min_samples_leaf'] == best_rf_params['min_samples_leaf']
            )
            tuning_rows.append(dict(
                Model='Weekday_RF',
                Model_Label=LABELS['Weekday_RF'],
                Setting_Description=(
                    f"n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, "
                    f"min_samples_leaf={params['min_samples_leaf']}, random_state={params['random_state']}"
                    + (' (Fast mode fixed)' if fast_mode else '')
                ),
                ridge_lambda=None,
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_leaf=params['min_samples_leaf'],
                random_state=params['random_state'],
                Validation_MAE=sc['MAE'],
                Validation_RMSE=sc['RMSE'],
                Validation_WAPE_pct=sc['WAPE_pct'],
                Validation_MASE=sc['MASE'],
                Validation_sMAPE_pct=sc['sMAPE_pct'],
                Selected=bool(is_sel),
            ))

    return dict(
        best_ridge_lambda=best_ridge_lam,
        best_rf_params=best_rf_params,
        tuning_df=pd.DataFrame(tuning_rows),
    )


def compute_feature_importance(y, days, train, dev, dates=None, has_real_dates=False,
                               ridge_lambda=1.0, rf_params=None):
    """
    Compute:
    1) Ridge standardized coefficients on training data (y[:train]).
    2) Random Forest impurity importance (on training) and permutation importance on the validation set
       (y[train:dev], n_repeats = 5, fixed random_state = 42).
    """
    f_names = rich_feature_names(has_real_dates)
    X_tr, t_tr, _ = build_rich_design_matrix(
        y[:train], days[:train],
        dates_sub=dates[:train] if dates else None,
        has_real_dates=has_real_dates,
    )
    ridge_mod = fit_standardized_ridge(X_tr, t_tr, lam=ridge_lambda)
    beta_std = ridge_mod['beta_std']
    abs_beta = np.abs(beta_std)
    ridge_ranks = np.argsort(np.argsort(-abs_beta)) + 1

    # Validation rows for permutation importance (indices train..dev-1)
    X_val_rows = []
    y_val_rows = []
    for i in range(train, dev):
        dt_i = dates[i] if (has_real_dates and dates is not None and i < len(dates)) else None
        X_val_rows.append(build_rich_feature_row(y[:i], days[i], i, target_date=dt_i, has_real_dates=has_real_dates))
        y_val_rows.append(float(y[i]))
    X_val = np.vstack(X_val_rows) if X_val_rows else np.empty((0, len(f_names)), dtype=float)
    y_val = np.asarray(y_val_rows, dtype=float)

    rf_imp = np.full(len(f_names), np.nan)
    rf_perm_mean = np.full(len(f_names), np.nan)
    rf_perm_sd = np.full(len(f_names), np.nan)
    rf_perm_ranks = [None] * len(f_names)

    if HAS_SKLEARN and RandomForestRegressor is not None and len(X_tr) > 0 and len(X_val) > 0:
        rf = RandomForestRegressor(**(rf_params or FIXED_RF_PARAMS))
        rf.fit(X_tr, t_tr)
        if hasattr(rf, 'feature_importances_'):
            rf_imp = np.asarray(rf.feature_importances_, dtype=float)
        else:
            # Compute split-gain impurity importance if custom forest lacks feature_importances_
            gains = np.zeros(len(f_names), dtype=float)
            for j in range(len(f_names)):
                c = np.corrcoef(X_tr[:, j], t_tr)[0, 1] if np.std(X_tr[:, j]) > 1e-12 else 0.0
                gains[j] = abs(c) if math.isfinite(c) else 0.0
            tot_g = float(np.sum(gains))
            rf_imp = gains / tot_g if tot_g > 1e-12 else np.full(len(f_names), 1.0 / len(f_names))

        base_pred = np.maximum(0.0, np.asarray(rf.predict(X_val), dtype=float))
        base_mae = float(np.mean(np.abs(y_val - base_pred)))
        rng = np.random.default_rng(42)
        for j in range(len(f_names)):
            diffs = []
            for _ in range(RF_PERM_REPEATS):
                X_perm = X_val.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                perm_pred = np.maximum(0.0, np.asarray(rf.predict(X_perm), dtype=float))
                perm_mae = float(np.mean(np.abs(y_val - perm_pred)))
                diffs.append(perm_mae - base_mae)
            rf_perm_mean[j] = float(np.mean(diffs))
            rf_perm_sd[j] = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        order_p = np.argsort(np.argsort(-rf_perm_mean)) + 1
        rf_perm_ranks = [int(r) for r in order_p]

    rows = []
    for j, name in enumerate(f_names):
        rows.append(dict(
            Feature=name,
            Ridge_Std_Coef=float(beta_std[j]),
            Ridge_Abs_Std_Coef=float(abs_beta[j]),
            Ridge_Importance_Rank=int(ridge_ranks[j]),
            RF_Permutation_MAE_Increase=float(rf_perm_mean[j]) if math.isfinite(rf_perm_mean[j]) else None,
            RF_Permutation_SD=float(rf_perm_sd[j]) if math.isfinite(rf_perm_sd[j]) else None,
            RF_Impurity_Importance=float(rf_imp[j]) if math.isfinite(rf_imp[j]) else None,
            RF_Permutation_Rank=rf_perm_ranks[j],
        ))
    df = pd.DataFrame(rows)
    top_ridge_idx = int(np.argmax(abs_beta))
    top_rf_idx = int(np.nanargmax(rf_perm_mean)) if np.any(np.isfinite(rf_perm_mean)) else None
    return dict(
        table_df=df,
        top_ridge_feature=f_names[top_ridge_idx],
        top_ridge_coef=float(beta_std[top_ridge_idx]),
        top_rf_feature=f_names[top_rf_idx] if top_rf_idx is not None else None,
        top_rf_perm_mae=float(rf_perm_mean[top_rf_idx]) if top_rf_idx is not None else None,
        warning="Important caution: feature importance measures predictive association on this split, not causal effect. Because Lag_1–Lag_5 and rolling 5/10-day means are correlated, they share predictive importance—interpret individual ranks with care.",
    )


def make_prefix_accelerator(
    y_full, days_full, ridge_lambda=1.0, rf_params=None,
    dates_full=None, has_real_dates=False, use_rich_features=True,
    hw_config=('add', False), sarima_spec=None, train_end=None, dev_end=None,
):
    """Precompute exact prefix structures on (y_full, days_full) to accelerate rolling evaluation."""
    n = len(y_full)
    rf_params = dict(rf_params or FIXED_RF_PARAMS)
    t_end = int(train_end if train_end is not None else max(10, int(n * 0.4)))
    d_end = int(dev_end if dev_end is not None else max(t_end + 5, int(n * 2 / 3)))
    grid = np.linspace(0, 1, 101)
    ar = np.arange(n, dtype=float)
    W_grid = np.zeros((101, n))
    W_grid[0, :] = 1.0
    W_grid[100, 0] = 1.0
    for k in range(1, 100):
        b = 1.0 - grid[k]
        m = min(n, int(-700.0 / math.log(b)) + 1)
        W_grid[k, :m] = b ** ar[:m]
    y_pad = np.r_[0.0, y_full[:-1]]
    P_grid = np.vstack([lfilter([a], [1.0, -(1.0 - a)], y_pad) for a in grid])
    R_grid = y_full[None, :] - P_grid
    WW_cum = np.cumsum(W_grid * W_grid, axis=1)
    WR_cum = np.cumsum(W_grid * R_grid, axis=1)
    RR_cum = np.cumsum(R_grid * R_grid, axis=1)
    ses_cache = {}

    def ses_level_prefix(i):
        if i in ses_cache:
            return ses_cache[i]
        yi = y_full[:i]
        ww = WW_cum[:, i - 1]
        wr = WR_cum[:, i - 1]
        init_g = np.maximum(0.0, wr / ww)
        if i <= 1020:
            err_g = R_grid[:, :i] - W_grid[:, :i] * init_g[:, None]
            sse_g = np.einsum('ki,ki->k', err_g, err_g)
        else:
            sse_g = RR_cum[:, i - 1] - 2.0 * init_g * wr + (init_g ** 2) * ww
        idx = int(np.argmin(sse_g))
        lo, hi = grid[max(0, idx - 1)], grid[min(100, idx + 1)]
        yi_pad = y_pad[:i]

        def profile(alpha):
            if alpha <= 0.0:
                return float(sse_g[0]), float(init_g[0]), 1.0, 0.0
            if alpha >= 1.0:
                return float(sse_g[100]), float(init_g[100]), 0.0, float(yi_pad[-1])
            b = 1.0 - alpha
            m = min(i, int(-700.0 / math.log(b)) + 1)
            w_m = b ** ar[:m]
            part = lfilter([alpha], [1.0, -b], yi_pad)
            r = yi - part
            ww_a = float(w_m @ w_m)
            wr_a = float(w_m @ r[:m])
            initial = max(0.0, wr_a / ww_a)
            if i <= 1020:
                err = r.copy()
                err[:m] -= w_m * initial
                sse = float(err @ err)
            else:
                sse = float(r @ r) - 2.0 * initial * wr_a + (initial ** 2) * ww_a
            w_last = float(w_m[-1]) if m == i else 0.0
            return sse, initial, w_last, float(part[-1])

        if idx == 0 and profile(1e-7)[0] >= sse_g[0] and profile(1e-3)[0] >= sse_g[0]:
            sse, initial, w_last, p_last, alpha = float(sse_g[0]), float(init_g[0]), 1.0, 0.0, 0.0
        elif idx == 100 and profile(1.0 - 1e-7)[0] >= sse_g[100] and profile(1.0 - 1e-3)[0] >= sse_g[100]:
            sse, initial, w_last, p_last, alpha = float(sse_g[100]), float(init_g[100]), 0.0, float(yi_pad[-1]), 1.0
        else:
            opt = minimize_scalar(lambda a: profile(a)[0], bounds=(lo, hi),
                                  method='bounded', options={'xatol': 1e-10})
            if not opt.success:
                raise ValueError('Exponential smoothing could not converge. Check the demand values.')
            candidates = [
                ((float(sse_g[0]), float(init_g[0]), 1.0, 0.0), 0.0),
                ((float(sse_g[100]), float(init_g[100]), 0.0, float(yi_pad[-1])), 1.0),
                (profile(float(opt.x)), float(opt.x)),
            ]
            (sse, initial, w_last, p_last), alpha = min(candidates, key=lambda item: item[0][0])
        level = float(alpha * yi[-1] + (1.0 - alpha) * (w_last * initial + p_last))
        ses_cache[i] = level
        return level

    # Precompute Holt linear trend trajectories across (alpha, beta) grid on y_full
    l0_holt = float(np.mean(y_full[:min(5, n)]))
    b0_holt = float((np.mean(y_full[5:min(10, n)]) - l0_holt) / 5.0) if n >= 10 else 0.0
    holt_alphas = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
    holt_betas = [0.01, 0.03, 0.08, 0.15, 0.3]
    holt_pairs = [(a, b) for a in holt_alphas for b in holt_betas]
    M_holt = len(holt_pairs)
    holt_levels = np.empty((M_holt, n))
    holt_trends = np.empty((M_holt, n))
    holt_sse_cum = np.empty((M_holt, n))
    for idx_h, (alpha, beta) in enumerate(holt_pairs):
        a_poly = [1.0, -(2.0 - alpha - alpha * beta), (1.0 - alpha)]
        b_sub = [alpha * (1.0 + beta), -alpha]
        yhat0 = l0_holt + b0_holt
        e0 = y_full[0] - yhat0
        l_0 = yhat0 + alpha * e0
        b_0 = b0_holt + alpha * beta * e0
        yhat1 = l_0 + b_0
        zi = lfiltic(b_sub, a_poly, y=[yhat1, yhat0], x=[y_full[0]])
        yhat_rest, _ = lfilter(b_sub, a_poly, y_full[1:], zi=zi)
        yhat = np.r_[yhat0, yhat1, yhat_rest[:-1]]
        err = y_full - yhat
        holt_trends[idx_h] = b0_holt + (alpha * beta) * np.cumsum(err)
        holt_levels[idx_h] = yhat + alpha * err
        holt_sse_cum[idx_h] = np.cumsum(err * err)

    keep_all = np.flatnonzero(days_full[1:] == ((days_full[:-1] - 1) % WEEK_LENGTH + 2)) + 1
    d_keep = days_full[keep_all]
    X_all = np.column_stack([
        np.ones(len(keep_all)),
        *(d_keep == code for code in DAY_CODES[:-1]),
        y_full[keep_all - 1],
    ]).astype(float) if len(keep_all) else np.empty((0, WEEK_LENGTH + 1), dtype=float)
    target_all = y_full[keep_all] if len(keep_all) else np.empty(0, dtype=float)
    dr_cache = {}
    ridge_base_cache = {}
    ridge_rich_cache = {}
    rf_base_cache = {}
    rf_rich_cache = {}
    P_ridge = np.diag([0.0] + [1.0] * WEEK_LENGTH)

    # Precompute rich feature rows for all indices t in 1..n-1 on y_full (each row t uses only y_full[:t])
    p_rich = len(rich_feature_names(has_real_dates))
    X_rich_all = np.zeros((n, p_rich), dtype=float)
    for t in range(1, n):
        dt_t = dates_full[t] if (has_real_dates and dates_full is not None and t < len(dates_full)) else None
        X_rich_all[t] = build_rich_feature_row(y_full[:t], days_full[t], t, target_date=dt_t, has_real_dates=has_real_dates)

    # Block anchor fits for Holt-Winters and SARIMA (fit at train_end, dev_end, and full n; update state inside blocks)
    hw_seasonal_mode, hw_damped_flag = hw_config
    hw_anchor_cache = {}

    def get_hw_state_for_prefix(k, y_sub):
        is_pref = k <= n and np.shares_memory(y_sub, y_full)
        if not is_pref:
            return fit_holt_winters_state(y_sub, seasonal=hw_seasonal_mode, damped=hw_damped_flag, period=STL_SEASONAL_PERIOD)
        anchor = t_end if k < d_end else (d_end if k < n else n)
        anchor = min(anchor, k)
        if anchor not in hw_anchor_cache:
            hw_anchor_cache[anchor] = fit_holt_winters_state(
                y_full[:anchor], seasonal=hw_seasonal_mode, damped=hw_damped_flag, period=STL_SEASONAL_PERIOD
            )
        st = hw_anchor_cache[anchor]
        if k > anchor:
            st = hw_update_state_with_obs(st, y_full[anchor:k])
        return st

    sarima_anchor_cache = {}
    if sarima_spec is not None and sarima_spec.get('fitted_train_res') is not None:
        sarima_anchor_cache[t_end] = sarima_spec['fitted_train_res']

    def get_sarima_anchor_for_prefix(k, y_sub):
        if sarima_spec is None:
            return None
        s_ord, s_seas = sarima_spec['order'], sarima_spec['seasonal_order']
        is_pref = k <= n and np.shares_memory(y_sub, y_full)
        if not is_pref:
            return fit_sarima_block(y_sub, s_ord, s_seas)
        anchor = t_end if k < d_end else (d_end if k < n else n)
        anchor = min(anchor, k)
        if anchor not in sarima_anchor_cache:
            sarima_anchor_cache[anchor] = fit_sarima_block(y_full[:anchor], s_ord, s_seas)
        return sarima_anchor_cache[anchor]

    ensemble_members = []

    def set_ensemble_members(members):
        nonlocal ensemble_members
        ensemble_members = list(members)

    def fast_regression_fit(y_sub, days_sub):
        k = len(y_sub)
        if k <= n and np.shares_memory(y_sub, y_full):
            m = int(np.searchsorted(keep_all, k))
            X = X_all[:m]
            target = target_all[:m]
            keep = keep_all[:m]
            if k not in dr_cache:
                dr_cache[k] = np.linalg.lstsq(X, target, rcond=None)[0]
            return dr_cache[k], X, target, keep
        y_arr = np.asarray(y_sub, dtype=float)
        d_arr = np.asarray(days_sub, dtype=int)
        keep = np.flatnonzero(d_arr[1:] == ((d_arr[:-1] - 1) % WEEK_LENGTH + 2)) + 1
        dk = d_arr[keep]
        X = np.column_stack([np.ones(len(keep)), *(dk == code for code in DAY_CODES[:-1]), y_arr[keep - 1]]).astype(float)
        target = y_arr[keep]
        beta = np.linalg.lstsq(X, target, rcond=None)[0]
        return beta, X, target, keep

    def fast_ridge_fit(y_sub, days_sub, lam=ridge_lambda):
        k = len(y_sub)
        key = (k, float(lam))
        if k <= n and np.shares_memory(y_sub, y_full):
            if key not in ridge_base_cache:
                m = int(np.searchsorted(keep_all, k))
                X = X_all[:m]
                target = target_all[:m]
                ridge_base_cache[key] = np.linalg.solve(X.T @ X + lam * P_ridge, X.T @ target)
            return ridge_base_cache[key]
        _, X, target, _ = fast_regression_fit(y_sub, days_sub)
        return np.linalg.solve(X.T @ X + lam * P_ridge, X.T @ target)

    def fast_rich_ridge_fit(y_sub, days_sub, lam=ridge_lambda):
        k = len(y_sub)
        key = (k, float(lam))
        if k <= n and np.shares_memory(y_sub, y_full):
            if key not in ridge_rich_cache:
                start_t = 10 if k >= 16 else max(1, min(5, k - 5))
                valid_k = keep_all[(keep_all >= start_t) & (keep_all < k)]
                if len(valid_k) < 4:
                    valid_k = np.arange(start_t, k, dtype=int)
                X_sub = X_rich_all[valid_k]
                t_sub = y_full[valid_k]
                ridge_rich_cache[key] = fit_standardized_ridge(X_sub, t_sub, lam=lam)
            return ridge_rich_cache[key]
        dt_sub = dates_full[:k] if (dates_full is not None and k <= len(dates_full)) else None
        X_sub, t_sub, _ = build_rich_design_matrix(y_sub, days_sub, dates_sub=dt_sub, has_real_dates=has_real_dates)
        return fit_standardized_ridge(X_sub, t_sub, lam=lam)

    def fast_rich_rf_fit(y_sub, days_sub):
        k = len(y_sub)
        is_pref = k <= n and np.shares_memory(y_sub, y_full)
        # Anchor RF fit at train_end, dev_end, and full n (or every 5-day block for short series) for speed & consistency
        anchor = t_end if (is_pref and k < d_end) else (d_end if (is_pref and k < n) else k)
        anchor = min(anchor, k)
        if is_pref and anchor in rf_rich_cache:
            return rf_rich_cache[anchor]
        if is_pref:
            start_t = 10 if anchor >= 16 else max(1, min(5, anchor - 5))
            valid_k = keep_all[(keep_all >= start_t) & (keep_all < anchor)]
            if len(valid_k) < 4:
                valid_k = np.arange(start_t, anchor, dtype=int)
            X_sub = X_rich_all[valid_k]
            t_sub = y_full[valid_k]
        else:
            dt_sub = dates_full[:k] if (dates_full is not None and k <= len(dates_full)) else None
            X_sub, t_sub, _ = build_rich_design_matrix(y_sub, days_sub, dates_sub=dt_sub, has_real_dates=has_real_dates)
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_sub, t_sub)
        if is_pref:
            rf_rich_cache[anchor] = rf
        return rf

    def fast_predict(model, y_sub, days_sub, future_days, window=5):
        k = len(y_sub)
        is_prefix = k <= n and np.shares_memory(y_sub, y_full)
        if model == 'Ensemble':
            if not ensemble_members:
                raise ValueError('Ensemble members have not been initialized from validation MAE.')
            member_preds = [fast_predict(m, y_sub, days_sub, future_days, window) for m in ensemble_members]
            return np.mean(np.vstack(member_preds), axis=0)
        if model == 'SES':
            level = ses_level_prefix(k) if is_prefix else fit_ses(y_sub)['level']
            return np.full(len(future_days), level)
        if model == 'Holt':
            if is_prefix:
                best_idx = int(np.argmin(holt_sse_cum[:, k - 1]))
                lv = float(holt_levels[best_idx, k - 1])
                tr = float(holt_trends[best_idx, k - 1])
            else:
                h_fit = fit_holt(y_sub)
                lv, tr = h_fit['level'], h_fit['trend']
            steps = np.arange(1, len(future_days) + 1, dtype=float)
            return np.maximum(0.0, lv + steps * tr)
        if model == 'SMA':
            if len(future_days) == 1:
                return np.asarray([float(np.mean(y_sub[-window:]))])
            history = list(np.asarray(y_sub[-window:], dtype=float))
            output = []
            for _ in future_days:
                value = float(np.mean(history[-window:]))
                output.append(value)
                history.append(value)
            return np.asarray(output)
        if model == 'Weekday_Naive':
            weekday_history = {}
            for idx_back in range(k - 1, -1, -1):
                d = int(days_sub[idx_back])
                if d not in weekday_history:
                    weekday_history[d] = float(y_sub[idx_back])
                    if len(weekday_history) == WEEK_LENGTH:
                        break
            if len(weekday_history) < WEEK_LENGTH:
                d_arr = np.asarray(days_sub)
                for d in DAY_CODES:
                    if d not in weekday_history:
                        weekday_history[d] = float(y_sub[np.flatnonzero(d_arr == d)[-1]])
            return np.asarray([weekday_history[int(day)] for day in future_days], dtype=float)
        if model == 'Weekday_DR' or (model == 'Weekday_Ridge' and not use_rich_features):
            beta = fast_regression_fit(y_sub, days_sub)[0] if model == 'Weekday_DR' else fast_ridge_fit(y_sub, days_sub, lam=FIXED_RIDGE_LAMBDA)
            previous_day, previous_value = int(days_sub[-1]), float(y_sub[-1])
            output = []
            for day in future_days:
                day = int(day)
                count = (day - previous_day) % WEEK_LENGTH or WEEK_LENGTH
                for _ in range(count):
                    previous_day = (previous_day - 1) % WEEK_LENGTH + 2
                    feat = np.array([
                        1.0,
                        *(float(previous_day == code) for code in DAY_CODES[:-1]),
                        previous_value,
                    ])
                    previous_value = max(0.0, float(feat @ beta))
                output.append(previous_value)
            return np.asarray(output)
        if model == 'Weekday_Ridge' and use_rich_features:
            r_mod = fast_rich_ridge_fit(y_sub, days_sub, lam=ridge_lambda)
            dt_sub = dates_full[:k] if (dates_full is not None and k <= len(dates_full)) else None
            return recursive_rich_predict(
                y_sub, days_sub, future_days,
                dates_sub=dt_sub, has_real_dates=has_real_dates,
                predictor_fn=lambda x_row: predict_standardized_ridge_row(r_mod, x_row),
            )
        if model == 'Weekday_RF' and HAS_SKLEARN and RandomForestRegressor is not None:
            if not use_rich_features:
                anchor = t_end if (is_prefix and k < d_end) else (d_end if (is_prefix and k < n) else k)
                anchor = min(anchor, k)
                if is_prefix and anchor in rf_base_cache:
                    rf = rf_base_cache[anchor]
                else:
                    _, X, target, _ = fast_regression_fit(y_full[:anchor] if is_prefix else y_sub, days_full[:anchor] if is_prefix else days_sub)
                    rf = RandomForestRegressor(**FIXED_RF_PARAMS)
                    rf.fit(X[:, 1:], target)
                    if is_prefix:
                        rf_base_cache[anchor] = rf
                previous_day, previous_value = int(days_sub[-1]), float(y_sub[-1])
                output = []
                for day in future_days:
                    day = int(day)
                    count = (day - previous_day) % WEEK_LENGTH or WEEK_LENGTH
                    for _ in range(count):
                        previous_day = (previous_day - 1) % WEEK_LENGTH + 2
                        feat = np.array([[
                            *(float(previous_day == code) for code in DAY_CODES[:-1]),
                            previous_value,
                        ]])
                        previous_value = max(0.0, predict_forest_row(rf, feat[0]))
                    output.append(previous_value)
                return np.asarray(output)
            else:
                rf = fast_rich_rf_fit(y_sub, days_sub)
                dt_sub = dates_full[:k] if (dates_full is not None and k <= len(dates_full)) else None
                return recursive_rich_predict(
                    y_sub, days_sub, future_days,
                    dates_sub=dt_sub, has_real_dates=has_real_dates,
                    predictor_fn=lambda x_row: predict_forest_row(rf, x_row),
                )
        if model == 'Holt_Winters':
            st = get_hw_state_for_prefix(k, y_sub)
            return hw_forecast_from_state(st, len(future_days))
        if model == 'SARIMA':
            base_res = get_sarima_anchor_for_prefix(k, y_sub)
            return sarima_forecast_with_state_update(base_res, y_sub, len(future_days))
        raise ValueError(f'Unknown model: {model}')

    fast_predict.set_ensemble_members = set_ensemble_members
    fast_predict.get_ensemble_members = lambda: list(ensemble_members)
    return fast_predict, fast_regression_fit, fast_ridge_fit


def downsample_xy(x, y, max_points=MAX_PLOT_POINTS):
    """Downsample dense display series while preserving endpoints."""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    n = len(x_arr)
    if n <= max_points:
        return x_arr, y_arr
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return x_arr[idx], y_arr[idx]


def format_label_unit(data_label, unit_label):
    """Avoid repeating 'Orders (orders)' when data_label and unit_label match case-insensitively."""
    dl = str(data_label or '').strip()
    ul = str(unit_label or '').strip()
    if not ul or dl.lower() == ul.lower():
        return dl
    return f'{dl} ({ul})'


def bin_series_for_bars(x, y, max_bars=12, prefix=''):
    """Aggregate a sequence (x, y) into at most max_bars contiguous chronological blocks for readable bar charts."""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)
    if n == 0:
        return [], np.array([], dtype=float), []
    if n <= max_bars:
        labels = [f'{prefix}{int(v)}' if prefix else str(int(v)) for v in x_arr]
        bounds = [(int(v), int(v)) for v in x_arr]
        return labels, y_arr.copy(), bounds
    splits = [s for s in np.array_split(np.arange(n), max_bars) if len(s) > 0]
    labels, means, bounds = [], [], []
    for idx in splits:
        a, b = int(x_arr[idx[0]]), int(x_arr[idx[-1]])
        lbl = f'{prefix}{a}–{b}' if a != b else f'{prefix}{a}'
        labels.append(lbl)
        means.append(float(np.nanmean(y_arr[idx])))
        bounds.append((a, b))
    return labels, np.asarray(means, dtype=float), bounds


def compute_intervals_and_coverage(y, days, train, dev, models, predict, window, future, min_errors=DEFAULT_MIN_INTERVAL_ERRORS):
    """
    Build empirical-quantile prediction intervals (80% and 95%) without future data leakage:
    - Test-period 1-step intervals are calibrated strictly on validation-period 1-step errors (indices train..dev-1).
    - Future forecast h-step intervals are calibrated on validation + test rolling h-step errors (origins train..n-h),
      and displayed only for horizons h that have at least min_errors backtest errors (a labeled rule of thumb).
    """
    n = len(y)
    horizon = len(future)
    val_1step_errors = {}
    test_1step_bounds = {}

    for m in models:
        val_preds = np.array([predict(m, y[:i], days[:i], [days[i]], window)[0] for i in range(train, dev)], dtype=float)
        val_errs = y[train:dev] - val_preds
        val_1step_errors[m] = val_errs
        q10, q90 = float(np.quantile(val_errs, 0.10)), float(np.quantile(val_errs, 0.90))
        q025, q975 = float(np.quantile(val_errs, 0.025)), float(np.quantile(val_errs, 0.975))
        test_1step_bounds[m] = dict(q10=q10, q90=q90, q025=q025, q975=q975, val_n=len(val_errs))

    h_errors = {m: [[] for _ in range(horizon)] for m in models}
    single_models = [m for m in models if m != 'Ensemble']
    ens_members = getattr(predict, 'get_ensemble_members', lambda: [])()
    for origin in range(train, n):
        max_h = min(horizon, n - origin)
        if max_h <= 0:
            continue
        act = y[origin:origin + max_h]
        origin_fcs = {}
        for m in single_models:
            fc = predict(m, y[:origin], days[:origin], days[origin:origin + max_h], window)
            origin_fcs[m] = fc
            errs = act - fc
            for h_idx in range(max_h):
                h_errors[m][h_idx].append(float(errs[h_idx]))
        if 'Ensemble' in models:
            if ens_members and all(em in origin_fcs for em in ens_members):
                fc_ens = np.mean(np.vstack([origin_fcs[em] for em in ens_members]), axis=0)
            else:
                fc_ens = predict('Ensemble', y[:origin], days[:origin], days[origin:origin + max_h], window)
            errs_ens = act - fc_ens
            for h_idx in range(max_h):
                h_errors['Ensemble'][h_idx].append(float(errs_ens[h_idx]))

    error_counts = np.array([len(h_errors[models[0]][h_idx]) for h_idx in range(horizon)], dtype=int)
    supported_mask = error_counts >= int(min_errors)
    max_supported_h = int(np.sum(supported_mask))

    future_intervals = {}
    for m in models:
        f_vals = future[m].to_numpy(dtype=float)
        lo80 = np.full(horizon, np.nan)
        hi80 = np.full(horizon, np.nan)
        lo95 = np.full(horizon, np.nan)
        hi95 = np.full(horizon, np.nan)
        for h_idx in range(horizon):
            errs_h = h_errors[m][h_idx]
            if len(errs_h) >= int(min_errors):
                arr_e = np.asarray(errs_h, dtype=float)
                lo80[h_idx] = max(0.0, float(f_vals[h_idx] + np.quantile(arr_e, 0.10)))
                hi80[h_idx] = max(0.0, float(f_vals[h_idx] + np.quantile(arr_e, 0.90)))
                lo95[h_idx] = max(0.0, float(f_vals[h_idx] + np.quantile(arr_e, 0.025)))
                hi95[h_idx] = max(0.0, float(f_vals[h_idx] + np.quantile(arr_e, 0.975)))
        future_intervals[m] = dict(lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95)

    return dict(
        test_1step_bounds=test_1step_bounds,
        future_intervals=future_intervals,
        error_counts=error_counts,
        max_supported_h=max_supported_h,
        min_errors_rule=int(min_errors),
    )


def evaluate_test_coverage(test_df, models, interval_info):
    """Compute empirical test-period coverage of validation-calibrated 80% and 95% intervals for every model."""
    actual = test_df['Actual'].to_numpy(dtype=float)
    min_req = interval_info['min_errors_rule']
    rows = []
    for m in models:
        b = interval_info['test_1step_bounds'][m]
        fc = test_df[m].to_numpy(dtype=float)
        lo80 = np.maximum(0.0, fc + b['q10'])
        hi80 = np.maximum(0.0, fc + b['q90'])
        lo95 = np.maximum(0.0, fc + b['q025'])
        hi95 = np.maximum(0.0, fc + b['q975'])
        test_df[f'{m}_lower_80'] = lo80
        test_df[f'{m}_upper_80'] = hi80
        test_df[f'{m}_lower_95'] = lo95
        test_df[f'{m}_upper_95'] = hi95
        in80 = (actual >= lo80 - 1e-9) & (actual <= hi80 + 1e-9)
        in95 = (actual >= lo95 - 1e-9) & (actual <= hi95 + 1e-9)
        cov80 = float(100.0 * np.mean(in80))
        cov95 = float(100.0 * np.mean(in95))
        close80 = abs(cov80 - 80.0) <= COVERAGE_TOLERANCE_PCT
        close95 = abs(cov95 - 95.0) <= COVERAGE_TOLERANCE_PCT
        val_n = b['val_n']
        rule_note = (
            f"Calibrated on {val_n} validation errors (fewer than the {min_req}-error rule of thumb)."
            if val_n < min_req else
            f"Calibrated on {val_n} validation errors (meets the {min_req}-error rule of thumb)."
        )
        if close80 and close95:
            assessment = f"Both 80% ({cov80:.1f}%) and 95% ({cov95:.1f}%) test coverages are close to nominal (within ±{COVERAGE_TOLERANCE_PCT:g} percentage points). {rule_note}"
        elif cov95 < 95.0 - COVERAGE_TOLERANCE_PCT or cov80 < 80.0 - COVERAGE_TOLERANCE_PCT:
            assessment = f"Intervals are narrower than nominal on test data (80% coverage: {cov80:.1f}%, 95% coverage: {cov95:.1f}%). {rule_note}"
        else:
            assessment = f"Intervals are conservative/wider than nominal on test data (80% coverage: {cov80:.1f}%, 95% coverage: {cov95:.1f}%). {rule_note}"
        rows.append(dict(
            Model=m,
            Model_Label=LABELS[m],
            Validation_Calibration_Errors=val_n,
            Meets_30_Error_Rule=bool(val_n >= min_req),
            Nominal_80_pct=80.0,
            Empirical_80_Coverage_pct=cov80,
            Mean_80_Width=float(np.mean(hi80 - lo80)),
            Nominal_95_pct=95.0,
            Empirical_95_Coverage_pct=cov95,
            Mean_95_Width=float(np.mean(hi95 - lo95)),
            Close_To_Nominal=bool(close80 and close95),
            Assessment=assessment,
        ))
    return pd.DataFrame(rows)


def rolling_origin_cv(
    y, days, train, models, window, K=DEFAULT_CV_FOLDS,
    fast_regression_fit=None, fast_ridge_fit=None,
    dates=None, has_real_dates=False, use_rich_features=False,
    ridge_lambda=1.0, rf_params=None, hw_config=('add', False),
    sarima_spec=None, ensemble_members=None,
):
    """
    Expanding-window rolling-origin cross-validation across K consecutive blocks after the initial training portion.
    For each fold, model parameters are estimated strictly on all data before the fold (y[:fold_start])
    reusing the chosen tuning settings (never re-running hyperparameter/AIC grid searches per fold),
    and the block is forecast one step ahead using actual values as they arrive (zero future leakage).
    """
    n = len(y)
    rem_indices = np.arange(train, n)
    K = max(3, min(int(K), len(rem_indices)))
    folds = [arr for arr in np.array_split(rem_indices, K) if len(arr) > 0]
    K_actual = len(folds)
    fold_rows = []
    rf_params = dict(rf_params or FIXED_RF_PARAMS)

    if fast_regression_fit is None or fast_ridge_fit is None:
        _, fast_regression_fit, fast_ridge_fit = make_prefix_accelerator(y, days)

    for fold_idx, idx_arr in enumerate(folds, start=1):
        f_start = int(idx_arr[0])
        f_end = int(idx_arr[-1]) + 1
        y_hist = y[:f_start]
        d_hist = days[:f_start]
        dt_hist = dates[:f_start] if dates else None
        fold_naive_scale = float(np.mean(np.abs(np.diff(y_hist)))) if len(y_hist) > 1 else np.nan
        act_fold = y[f_start:f_end]

        # Fit fold parameters once on y[:f_start] (no data from >= f_start enters parameter estimation)
        ses_fit = fit_ses(y_hist)
        ses_alpha, ses_level_init = float(ses_fit['alpha']), float(ses_fit['level'])

        holt_fit = fit_holt(y_hist)
        h_alpha, h_beta = float(holt_fit['alpha']), float(holt_fit['beta'])
        h_level_init, h_trend_init = float(holt_fit['level']), float(holt_fit['trend'])

        dr_beta = fast_regression_fit(y_hist, d_hist)[0]

        if use_rich_features:
            X_r, t_r, _ = build_rich_design_matrix(y_hist, d_hist, dates_sub=dt_hist, has_real_dates=has_real_dates)
            ridge_rich_mod = fit_standardized_ridge(X_r, t_r, lam=ridge_lambda)
            ridge_beta = None
        else:
            ridge_beta = fast_ridge_fit(y_hist, d_hist, lam=ridge_lambda)
            ridge_rich_mod = None

        rf_model = None
        if 'Weekday_RF' in models and HAS_SKLEARN and RandomForestRegressor is not None:
            rf_model = RandomForestRegressor(**rf_params)
            if use_rich_features:
                rf_model.fit(X_r, t_r)
            else:
                _, X_rf, t_rf, _ = fast_regression_fit(y_hist, d_hist)
                rf_model.fit(X_rf[:, 1:], t_rf)

        hw_state = None
        if 'Holt_Winters' in models:
            hw_state = fit_holt_winters_state(
                y_hist, seasonal=hw_config[0], damped=hw_config[1], period=STL_SEASONAL_PERIOD
            )

        sarima_res = None
        if 'SARIMA' in models and sarima_spec is not None:
            sarima_res = fit_sarima_block(y_hist, sarima_spec['order'], sarima_spec['seasonal_order'])

        fold_preds_map = {}
        single_models = [m for m in models if m != 'Ensemble']
        for m in single_models:
            preds = []
            if m == 'SMA':
                for t in range(f_start, f_end):
                    preds.append(float(np.mean(y[t - window:t])))
            elif m == 'SES':
                lv = ses_level_init
                for t in range(f_start, f_end):
                    preds.append(max(0.0, float(lv)))
                    lv = ses_alpha * float(y[t]) + (1.0 - ses_alpha) * lv
            elif m == 'Holt':
                lv, tr = h_level_init, h_trend_init
                for t in range(f_start, f_end):
                    preds.append(max(0.0, float(lv + tr)))
                    new_lv = h_alpha * float(y[t]) + (1.0 - h_alpha) * (lv + tr)
                    tr = h_beta * (new_lv - lv) + (1.0 - h_beta) * tr
                    lv = new_lv
            elif m == 'Weekday_Naive':
                wk_map = {d: float(y_hist[np.flatnonzero(d_hist == d)[-1]]) for d in DAY_CODES}
                for t in range(f_start, f_end):
                    d_cur = int(days[t])
                    preds.append(wk_map[d_cur])
                    wk_map[d_cur] = float(y[t])
            elif m == 'Weekday_DR' or (m == 'Weekday_Ridge' and not use_rich_features):
                beta_vec = dr_beta if m == 'Weekday_DR' else ridge_beta
                for t in range(f_start, f_end):
                    prev_d, prev_v = int(days[t - 1]), float(y[t - 1])
                    target_d = int(days[t])
                    cnt = (target_d - prev_d) % WEEK_LENGTH or WEEK_LENGTH
                    for _ in range(cnt):
                        prev_d = (prev_d - 1) % WEEK_LENGTH + 2
                        feat = np.array([
                            1.0,
                            *(float(prev_d == code) for code in DAY_CODES[:-1]),
                            prev_v,
                        ])
                        prev_v = max(0.0, float(feat @ beta_vec))
                    preds.append(prev_v)
            elif m == 'Weekday_Ridge' and use_rich_features:
                for t in range(f_start, f_end):
                    dt_t = dates[:t] if dates else None
                    fc_t = recursive_rich_predict(
                        y[:t], days[:t], [days[t]],
                        dates_sub=dt_t, has_real_dates=has_real_dates,
                        predictor_fn=lambda x_row, rm=ridge_rich_mod: predict_standardized_ridge_row(rm, x_row),
                    )[0]
                    preds.append(float(fc_t))
            elif m == 'Weekday_RF' and rf_model is not None:
                if not use_rich_features:
                    for t in range(f_start, f_end):
                        prev_d, prev_v = int(days[t - 1]), float(y[t - 1])
                        target_d = int(days[t])
                        cnt = (target_d - prev_d) % WEEK_LENGTH or WEEK_LENGTH
                        for _ in range(cnt):
                            prev_d = (prev_d - 1) % WEEK_LENGTH + 2
                            feat = np.array([[
                                *(float(prev_d == code) for code in DAY_CODES[:-1]),
                                prev_v,
                            ]])
                            prev_v = max(0.0, float(rf_model.predict(feat)[0]))
                        preds.append(prev_v)
                else:
                    for t in range(f_start, f_end):
                        dt_t = dates[:t] if dates else None
                        fc_t = recursive_rich_predict(
                            y[:t], days[:t], [days[t]],
                            dates_sub=dt_t, has_real_dates=has_real_dates,
                            predictor_fn=lambda x_row, rfm=rf_model: float(rfm.predict(x_row[None, :])[0]),
                        )[0]
                        preds.append(float(fc_t))
            elif m == 'Holt_Winters' and hw_state is not None:
                cur_hw = dict(hw_state)
                for t in range(f_start, f_end):
                    preds.append(float(hw_forecast_from_state(cur_hw, 1)[0]))
                    cur_hw = hw_update_state_with_obs(cur_hw, [y[t]])
            elif m == 'SARIMA':
                for t in range(f_start, f_end):
                    preds.append(float(sarima_forecast_with_state_update(sarima_res, y[:t], 1)[0]))
            fold_preds_map[m] = np.asarray(preds, dtype=float)

        if 'Ensemble' in models:
            ens_m = ensemble_members if (ensemble_members and all(em in fold_preds_map for em in ensemble_members)) else single_models[:3]
            fold_preds_map['Ensemble'] = np.mean(np.vstack([fold_preds_map[em] for em in ens_m]), axis=0)

        fold_model_scores = {
            m: score_with_scale(act_fold, fold_preds_map[m], naive_scale=fold_naive_scale)
            for m in models
        }
        fold_winner = min(models, key=lambda m: fold_model_scores[m]['MAE'])
        for m in models:
            sc = fold_model_scores[m]
            fold_rows.append(dict(
                Fold=fold_idx,
                Train_End_Observation=f_start,
                Fold_Start_Observation=f_start + 1,
                Fold_End_Observation=f_end,
                Fold_Size=f_end - f_start,
                Model=m,
                Model_Label=LABELS[m],
                MAE=sc['MAE'],
                RMSE=sc['RMSE'],
                MASE=sc['MASE'],
                sMAPE_pct=sc['sMAPE_pct'],
                Won_Fold=bool(m == fold_winner),
            ))

    fold_df = pd.DataFrame(fold_rows)
    summary_rows = []
    for m in models:
        sub = fold_df[fold_df.Model == m]
        summary_rows.append(dict(
            Model=m,
            Model_Label=LABELS[m],
            Folds=K_actual,
            Folds_Won=int(sub.Won_Fold.sum()),
            Mean_MAE=float(sub.MAE.mean()),
            SD_MAE=float(sub.MAE.std(ddof=1)) if len(sub) > 1 else 0.0,
            Mean_RMSE=float(sub.RMSE.mean()),
            SD_RMSE=float(sub.RMSE.std(ddof=1)) if len(sub) > 1 else 0.0,
            Mean_MASE=float(sub.MASE.mean()),
            SD_MASE=float(sub.MASE.std(ddof=1)) if len(sub) > 1 else 0.0,
            Mean_sMAPE_pct=float(sub.sMAPE_pct.mean()),
        ))
    summary_df = pd.DataFrame(summary_rows).sort_values(['Mean_MAE', 'Model']).reset_index(drop=True)
    cv_winner = str(summary_df.iloc[0].Model)
    return fold_df, summary_df, cv_winner, K_actual


def validate(payload):
    raw_rows = payload.get('rows')
    if not isinstance(raw_rows, list) or not 40 <= len(raw_rows) <= MAX_OBSERVATIONS:
        raise ValueError(f'Use 40 to {MAX_OBSERVATIONS:,} daily observations. Keep them in chronological order.')
    # Never mutate the caller's original rows
    rows = [dict(r) if isinstance(r, dict) else r for r in raw_rows]
    settings = payload.get('settings', {}) or {}
    calendar_mode = str(settings.get('calendar_mode', 'weekdays') or 'weekdays').strip().lower()
    if calendar_mode not in ('weekdays', 'all_days'):
        raise ValueError('Operating calendar must be Monday–Friday or all seven days.')
    week_length = 7 if calendar_mode == 'all_days' else 5
    day_codes = tuple(range(2, 2 + week_length))

    def number(name, default, low, high, integer=False):
        try:
            value = float(settings.get(name, default))
        except (ValueError, TypeError):
            raise ValueError(f'Enter a valid number for {name.replace("_", " ")}.')
        if not math.isfinite(value) or not low <= value <= high or (integer and value != int(value)):
            raise ValueError(f'{name.replace("_", " ").capitalize()} must be between {low} and {high}' + (' and a whole number.' if integer else '.'))
        return int(value) if integer else value

    raw_dtype = str(settings.get('data_type', 'Orders') or 'Orders').strip()
    custom_dtype = str(settings.get('custom_data_type', '') or '').strip()
    if raw_dtype.lower() == 'custom':
        if not custom_dtype:
            raise ValueError('Enter a name for your custom data type.')
        data_label = custom_dtype
    else:
        data_label = raw_dtype
    unit_label = str(settings.get('unit_name', 'orders') or 'orders').strip()
    if not unit_label:
        raise ValueError('Enter a unit name for your data (for example: units, kWh, orders).')

    purchase_requested = bool(settings.get('purchase_plan_requested', False))
    plan_start_date = None
    current_stock = None
    if purchase_requested:
        for req_field, friendly in [
            ('plan_start_date', 'plan start date'),
            ('current_stock', 'current stock on hand'),
            ('lead_time', 'supplier lead time (days)'),
            ('order_cost', 'cost per order'),
            ('holding_cost_unit_year', 'holding cost per unit per year'),
        ]:
            val = settings.get(req_field)
            if val is None or str(val).strip() == '':
                raise ValueError(f'Please enter the {friendly} to calculate your purchase plan.')
        try:
            plan_start_date = date.fromisoformat(str(settings.get('plan_start_date')).strip())
        except ValueError:
            raise ValueError('Plan start date must be a valid date in YYYY-MM-DD format.')
        if calendar_mode == 'weekdays' and plan_start_date.weekday() > 4:
            raise ValueError('Plan start date falls on a weekend. Please choose a Monday–Friday operating date.')
        try:
            current_stock = float(settings.get('current_stock'))
        except (ValueError, TypeError):
            raise ValueError('Current stock on hand must be a valid number.')
        if not math.isfinite(current_stock) or current_stock < 0 or current_stock > 1e12:
            raise ValueError('Current stock on hand must be 0 or greater.')

    has_service_key = 'service_level' in settings
    raw_service = settings.get('service_level', None)
    is_blank_service = (
        raw_service is None
        or (isinstance(raw_service, str) and raw_service.strip().lower() in ('', 'none', 'null', 'optional', '0', '0%'))
    )
    if is_blank_service:
        if purchase_requested or has_service_key:
            svc_val = None
            service_level_provided = False
        else:
            svc_val = 0.95
            service_level_provided = False
    else:
        try:
            svc_val = float(raw_service)
        except (ValueError, TypeError):
            raise ValueError('Enter a valid number for service level, or leave it blank for no safety stock buffer.')
        if svc_val == 0.0:
            svc_val = None
            service_level_provided = False
        else:
            if svc_val > 1.0:
                svc_val = svc_val / 100.0
            if not math.isfinite(svc_val) or not .51 <= svc_val <= .999:
                raise ValueError('Service level must be between 51% and 99.9% (or left blank for 0 safety stock).')
            service_level_provided = True

    # Data preparation & Fast mode settings
    explicit_missing_opt = 'missing_treatment' in settings and settings.get('missing_treatment') is not None
    missing_treatment = str(settings.get('missing_treatment', 'interpolate') or 'interpolate').strip().lower()
    if missing_treatment not in ('interpolate', 'weekday_median', 'reject'):
        raise ValueError('Missing-value treatment must be interpolate, weekday_median, or reject.')
    max_missing_pct = number('max_missing_pct', DEFAULT_MAX_MISSING_SHARE_PCT, 0.0, 50.0, False)

    outlier_treatment = str(settings.get('outlier_treatment', 'keep') or 'keep').strip().lower()
    if outlier_treatment not in ('keep', 'cap', 'replace'):
        raise ValueError('Outlier treatment must be keep, cap, or replace.')
    outlier_threshold = number('outlier_threshold', DEFAULT_OUTLIER_Z_THRESHOLD, 1.5, 10.0, False)

    hw_seasonal = str(settings.get('hw_seasonal', 'auto') or 'auto').strip().lower()
    if hw_seasonal not in ('auto', 'add', 'mul'):
        hw_seasonal = 'auto'
    hw_damped_raw = str(settings.get('hw_damped', 'auto') or 'auto').strip().lower()
    if hw_damped_raw not in ('auto', 'true', 'false'):
        hw_damped_raw = 'auto'

    fast_mode_threshold = number('fast_mode_row_threshold', DEFAULT_FAST_MODE_ROW_THRESHOLD, 40, 10000, True)
    n = len(rows)
    if 'fast_mode' in settings and settings.get('fast_mode') is not None and str(settings.get('fast_mode')).strip() != '':
        fm_val = settings.get('fast_mode')
        if isinstance(fm_val, str):
            fast_mode = fm_val.strip().lower() in ('1', 'true', 'yes', 'on')
        else:
            fast_mode = bool(fm_val)
    else:
        fast_mode = bool(n > fast_mode_threshold)

    options = dict(
        horizon=number('horizon', 60, 5, 120, True),
        lead_time=number('lead_time', 5, 1, 20, True),
        operating_days_year=number('operating_days_year', 365 if week_length == 7 else 260, 1, 366, True),
        order_cost=number('order_cost', 50, .01, 1e6),
        holding_cost_unit_year=number('holding_cost_unit_year', 2, .001, 1e6),
        service_level=svc_val,
        service_level_provided=service_level_provided,
        start_weekday=number('start_weekday', 2, 2, 1 + week_length, True),
        cv_folds=number('cv_folds', DEFAULT_CV_FOLDS, 3, 10, True),
        min_interval_errors=number('min_interval_errors', DEFAULT_MIN_INTERVAL_ERRORS, 5, 500, True),
        missing_treatment=missing_treatment,
        max_missing_pct=max_missing_pct,
        outlier_treatment=outlier_treatment,
        outlier_threshold=outlier_threshold,
        hw_seasonal=hw_seasonal,
        hw_damped=hw_damped_raw,
        fast_mode=fast_mode,
        fast_mode_row_threshold=fast_mode_threshold,
        data_type=data_label,
        unit_name=unit_label,
        purchase_plan_requested=purchase_requested,
        plan_start_date=str(plan_start_date) if plan_start_date else None,
        current_stock=current_stock,
        calendar_mode=calendar_mode,
        week_length=week_length,
    )
    options.update(train_end=int(n * .4), validation_end=int(n * 2 / 3))
    if options['horizon'] < options['lead_time']:
        raise ValueError('Forecast horizon must be at least as long as supplier lead time.')
    if options['validation_end'] - options['train_end'] - options['lead_time'] + 1 < 3:
        raise ValueError('Not enough development data for this lead time. Use more rows or a shorter lead time.')

    raw_values = []
    blank_indices = []
    days = []
    dates = []
    derived = 0
    names = {'mon': 2, 'tue': 3, 'wed': 4, 'thu': 5, 'fri': 6, 'sat': 7, 'sun': 8}
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'Invalid input at row {idx+1}.')
        raw_dem = row.get('demand', '')
        is_blank_str = (raw_dem is None) or (isinstance(raw_dem, str) and raw_dem.strip() == '')
        is_nan_val = isinstance(raw_dem, (float, np.floating)) and math.isnan(float(raw_dem))
        if is_blank_str or (is_nan_val and explicit_missing_opt):
            raw_values.append(np.nan)
            blank_indices.append(idx)
        else:
            try:
                value = float(raw_dem)
            except (TypeError, ValueError):
                raise ValueError(f'Row {idx+1}: value must be a number without units or thousands separators.')
            if not math.isfinite(value) or value < 0 or value > 1e9:
                raise ValueError(f'Row {idx+1}: value must be finite and between 0 and 1,000,000,000.')
            raw_values.append(value)

        raw_day = str(row.get('weekday', '')).strip().lower()
        raw_date = str(row.get('date', '')).strip()
        if raw_date:
            try:
                d = date.fromisoformat(raw_date)
            except ValueError:
                raise ValueError(f'Row {idx+1}: use YYYY-MM-DD dates.')
            if week_length == 5 and d.weekday() > 4:
                raise ValueError(f'Row {idx+1}: this project models Monday-Friday operating days. Remove weekend records.')
            expected_gap = 1 if week_length == 7 else (3 if dates and dates[-1].weekday() == 4 else 1)
            if dates and (d - dates[-1]).days != expected_gap:
                calendar_text = 'calendar days' if week_length == 7 else 'Monday-Friday operating days'
                raise ValueError(f'Date rows must be consecutive {calendar_text} in ascending order. Fill missing records, or map a weekday column to use the explicit minimum-gap assumption.')
            dates.append(d)
            day = d.weekday() + 2
        elif raw_day:
            if raw_day[:3] in names:
                day = names[raw_day[:3]]
            else:
                try:
                    f = float(raw_day); day = int(f)
                    if day != f or day not in day_codes:
                        raise ValueError()
                except (ValueError, OverflowError):
                    allowed = 'Mon-Sun or codes 2-8' if week_length == 7 else 'Mon-Fri or UCI codes 2-6'
                    raise ValueError(f'Row {idx+1}: weekday must be {allowed}.')
        else:
            derived += 1
            day = (options['start_weekday'] - 2 + idx) % week_length + 2
        days.append(day)

    if derived not in (0, n):
        raise ValueError('Some weekdays are blank. Fill every weekday, or choose generated weekdays for the entire column.')
    has_real_dates = bool(len(dates) == n)
    options['has_real_dates'] = has_real_dates
    days = np.array(days, int)

    # Handle blank demand values without future leakage
    blank_positions_1based = [int(i + 1) for i in blank_indices]
    blank_share_pct = 100.0 * len(blank_indices) / n
    if len(blank_indices) > 0:
        if missing_treatment == 'reject':
            pos_str = ', '.join(str(p) for p in blank_positions_1based[:15])
            raise ValueError(
                f"Rejected {len(blank_indices)} blank demand value(s) at observation(s) {pos_str} "
                "(missing-value option is set to 'reject'). Choose linear interpolation or weekday median to impute blanks."
            )
        if blank_share_pct > max_missing_pct + 1e-9:
            pos_str = ', '.join(str(p) for p in blank_positions_1based[:15])
            raise ValueError(
                f"Dataset has {len(blank_indices)} blank demand values ({blank_share_pct:.1f}% of {n} rows, at observations {pos_str}), "
                f"which exceeds the rule-of-thumb maximum missing share of {max_missing_pct:g}%."
            )

    y_raw_arr = np.array(raw_values, dtype=float)
    train_end = options['train_end']
    blank_records = []
    y_imputed = y_raw_arr.copy()
    if len(blank_indices) > 0:
        valid_mask = np.isfinite(y_raw_arr)
        if not np.any(valid_mask[:train_end]):
            raise ValueError('The training period contains no valid numeric demand observations.')
        for b_idx in blank_indices:
            w_day = int(days[b_idx])
            # Causal window: inside training block, use only training observations; for >= train_end, use only prior observations <= b_idx
            causal_limit = train_end if b_idx < train_end else (b_idx + 1)
            if missing_treatment == 'weekday_median':
                wk_candidates = y_imputed[:causal_limit][np.isfinite(y_imputed[:causal_limit]) & (days[:causal_limit] == w_day)]
                if len(wk_candidates) == 0:
                    wk_candidates = y_imputed[:train_end][np.isfinite(y_imputed[:train_end])]
                fill_val = max(0.0, float(np.median(wk_candidates)))
            else:
                # Linear interpolation within training block, and causal linear/carry-forward for >= train_end
                prev_valid = np.flatnonzero(np.isfinite(y_imputed[:b_idx]))
                next_valid = np.flatnonzero(np.isfinite(y_imputed[b_idx + 1:causal_limit])) + (b_idx + 1)
                if len(prev_valid) > 0 and len(next_valid) > 0:
                    i0, i1 = int(prev_valid[-1]), int(next_valid[0])
                    v0, v1 = float(y_imputed[i0]), float(y_imputed[i1])
                    fill_val = max(0.0, v0 + (v1 - v0) * ((b_idx - i0) / float(i1 - i0)))
                elif len(prev_valid) > 0:
                    fill_val = max(0.0, float(y_imputed[int(prev_valid[-1])]))
                elif len(next_valid) > 0:
                    fill_val = max(0.0, float(y_imputed[int(next_valid[0])]))
                else:
                    fill_val = max(0.0, float(np.nanmedian(y_imputed[:train_end])))
            y_imputed[b_idx] = fill_val
            blank_records.append(dict(
                Observation=b_idx + 1,
                Weekday=WEEKDAY_NAMES.get(w_day, str(w_day)),
                Missing_Treatment=missing_treatment,
                Filled_Demand=fill_val,
            ))

    y = y_imputed
    if y.sum() <= 0:
        raise ValueError(f'Include some positive {data_label.lower()}. An all-zero series cannot support the calculations.')
    if set(days[:options['train_end']]) != set(day_codes):
        raise ValueError(f'The first training block must contain all {week_length} weekdays for this model comparison.')
    notes = []
    if derived:
        calendar_text = 'Monday-Sunday calendar days' if week_length == 7 else 'Monday-Friday operating days'
        notes.append(f'Weekdays were generated as consecutive {calendar_text} from your selected first weekday. Confirm that this matches the data.')
    if len(blank_indices) > 0:
        notes.append(
            f"Data preparation: filled {len(blank_indices)} blank demand value(s) ({blank_share_pct:.1f}% of rows, at observations "
            f"{', '.join(str(p) for p in blank_positions_1based)}) using {missing_treatment.replace('_', ' ')} "
            f"(rule-of-thumb max missing share: {max_missing_pct:g}%)."
        )
    gaps = []
    for i in range(1, n):
        expected = (int(days[i - 1]) - 1) % week_length + 2
        if days[i] != expected:
            gaps.append(dict(Previous_Observation=i, Current_Observation=i + 1,
                             Minimum_Skipped_Weekdays=((int(days[i]) - int(days[i - 1])) % week_length or week_length) - 1))
    if gaps:
        notes.append(f'{len(gaps)} nonconsecutive weekday transitions found. Unknown-lag fitting rows are excluded; forecasts bridge minimum weekday-consistent gaps. Full-week gaps cannot be inferred from weekday codes.')
    if len(gaps) > n * .15:
        raise ValueError('Too many weekday gaps for this comparison. Check the weekday mapping and chronological order.')

    options['_dates_list'] = dates if has_real_dates else None
    options['_raw_values_with_nan'] = [None if math.isnan(v) else float(v) for v in y_raw_arr]
    options['_blank_positions'] = blank_positions_1based
    options['_blank_share_pct'] = blank_share_pct
    options['_blank_records'] = blank_records
    return y, days, options, notes, gaps


def diagnostics(core, y, days):
    beta, X, target, _ = core['regression_fit'](y, days)
    residual = target - X @ beta; n, p = X.shape; df = n - p
    inverse = np.linalg.pinv(X.T @ X)
    leverage = np.sum((X @ inverse) * X, axis=1)
    adjusted = residual / np.maximum(1 - leverage, 1e-10)
    covariance = inverse @ (X.T @ ((adjusted ** 2)[:, None] * X)) @ inverse
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    sse = float(residual @ residual); sst = float(np.sum((target - target.mean()) ** 2))
    epsilon = np.finfo(float).eps * max(float(target @ target), 1.) * 100
    identifiable = np.linalg.matrix_rank(X) == p and sse > epsilon
    stat = np.divide(beta, se, out=np.full(p, np.nan), where=se > 0) if identifiable else np.full(p, np.nan)
    coef_terms = ['Intercept', *(WEEKDAY_NAMES[d] for d in DAY_CODES[:-1]), 'Lag1']
    coef = pd.DataFrame(dict(Term=coef_terms, Estimate=beta,
                             HC3_SE=se if identifiable else np.full(p, np.nan), t_stat=stat, p_value=2 * student_t.sf(np.abs(stat), df)))
    r2 = 1 - sse / sst if sst > epsilon else np.nan
    center = residual - residual.mean(); denom = float(center @ center)
    acf = np.array([float(center[h:] @ center[:-h] / denom) for h in range(1, 6)]) if denom > epsilon else np.full(5, np.nan)
    lb = float(n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, 6))))
    sq = residual ** 2; fit_sq = X @ np.linalg.lstsq(X, sq, rcond=None)[0]
    sqsst = float(np.sum((sq - sq.mean()) ** 2))
    bp = float(n * (1 - np.sum((sq - fit_sq) ** 2) / sqsst)) if sqsst > epsilon else np.nan
    return coef, dict(N=n, DF=df, R2=r2, Adjusted_R2=1 - (1 - r2) * (n - 1) / df,
                      Residual_Mean=float(residual.mean()), Residual_SD=float(np.sqrt(sse / df)),
                      LB5=lb, LB5_p=float(chi2.sf(lb, 5)), BP=bp, BP_p=float(chi2.sf(bp, p - 1)), ACF=acf.tolist()), residual, X @ beta


def build_purchase_plan(y_future, winner_label, final_inv, options, data_label, unit_label, sim_fn):
    """Build the Part 2 Purchase Plan schedule and plain-language summary when requested by the user."""
    if not options.get('purchase_plan_requested'):
        return None
    start_dt = date.fromisoformat(options['plan_start_date'])
    cur_stock = float(options['current_stock'])
    lead = int(options['lead_time'])
    horizon = len(y_future)
    q = float(final_inv['EOQ'])
    rop = float(final_inv['reorder_point'])
    safety = float(final_inv['safety_stock'])
    sim_df, sim_summary, sim_events = sim_fn(y_future, q, rop, lead, initial=cur_stock)
    b_dates = generate_business_dates(start_dt, horizon + lead + 5)

    order_now = cur_stock <= rop + 1e-8
    schedule = []
    expected_dates = []
    if len(sim_events):
        # The continuous-review simulator may place several EOQ batches at the
        # same instant when the opening inventory position is far below the ROP.
        # They are one operational purchasing decision, not duplicate orders.
        event_rows = sim_events.copy()
        event_rows['Day_Index'] = np.floor(event_rows['Time'].astype(float) + 1e-9).astype(int)
        order_rows = event_rows[event_rows.Event == 'Order']
        grouped_orders = order_rows.groupby('Day_Index', sort=True)
        for idx_o, (raw_day_idx, group) in enumerate(grouped_orders, start=1):
            day_idx = min(max(0, int(raw_day_idx)), horizon - 1)
            t_val = float(group['Time'].min())
            ord_date = b_dates[day_idx]
            rec_date = advance_business_days(ord_date, lead)
            proj_stock = float(group.iloc[0].get('Net_Inventory', cur_stock if day_idx == 0 and t_val < 1e-6 else sim_df.iloc[day_idx]['Net_Inventory']))
            backlog = max(0.0, -proj_stock)
            batch_count = int(len(group))
            total_qty = float(group['Quantity'].sum())
            outstanding = float(group.iloc[-1].get('Outstanding_Receipts', total_qty))
            inventory_position = float(group.iloc[0].get('Inventory_Position', cur_stock if t_val < 1e-6 else rop))
            lead_slice = y_future[day_idx:min(horizon, day_idx + lead)]
            schedule.append(dict(
                Order_Number=idx_o,
                Operating_Day=day_idx + 1,
                Order_Date=str(ord_date),
                Order_Weekday=ord_date.strftime('%A'),
                Projected_Stock_At_Trigger=round(proj_stock, 2),
                Backlog_At_Trigger=round(backlog, 2),
                Inventory_Position_At_Trigger=round(inventory_position, 2),
                Outstanding_Receipts_After_Order=round(outstanding, 2),
                EOQ_Batch_Count=batch_count,
                Order_Quantity_EOQ=round(q, 2),
                Total_Order_Quantity=round(total_qty, 2),
                Expected_Receipt_Date=str(rec_date),
                Receipt_Weekday=rec_date.strftime('%A'),
                Forecast_Lead_Demand=round(float(np.sum(lead_slice)), 2),
            ))
            expected_dates.append(f"{ord_date} ({ord_date.strftime('%a')})")

    if order_now:
        first_receipt = advance_business_days(start_dt, lead)
        initial_event = schedule[0] if schedule and schedule[0]['Order_Date'] == str(start_dt) else None
        initial_qty = float(initial_event['Total_Order_Quantity']) if initial_event else q
        initial_batches = int(initial_event['EOQ_Batch_Count']) if initial_event else 1
        batch_text = f" ({initial_batches} EOQ batches consolidated into one order)" if initial_batches > 1 else ""
        action_Now = (
            f"Yes — order {initial_qty:,.2f} {unit_label}{batch_text} immediately on {start_dt} ({start_dt.strftime('%A')}). "
            f"Your current stock on hand ({cur_stock:,.2f} {unit_label}) is at or below the reorder point "
            f"({rop:,.2f} {unit_label}). With a {lead}-day supplier lead time, this shipment is expected on "
            f"{first_receipt} ({first_receipt.strftime('%A')})."
        )
    elif schedule:
        first_ord = schedule[0]
        action_Now = (
            f"No immediate order needed on {start_dt}. Your current stock ({cur_stock:,.2f} {unit_label}) is "
            f"{cur_stock - rop:,.2f} {unit_label} above the reorder point ({rop:,.2f} {unit_label}). "
            f"Based on the {winner_label.lower()} forecast, your first order of {q:,.2f} {unit_label} is projected on "
            f"{first_ord['Order_Date']} ({first_ord['Order_Weekday']}, operating day {first_ord['Operating_Day']})."
        )
    else:
        action_Now = (
            f"No order is projected during the {horizon}-day forecast horizon. Your current stock "
            f"({cur_stock:,.2f} {unit_label}) stays above the reorder point ({rop:,.2f} {unit_label}) across all {horizon} forecasted operating days."
        )

    sched_df = pd.DataFrame(schedule) if schedule else pd.DataFrame(columns=[
        'Order_Number', 'Operating_Day', 'Order_Date', 'Order_Weekday',
        'Projected_Stock_At_Trigger', 'Backlog_At_Trigger', 'Inventory_Position_At_Trigger',
        'Outstanding_Receipts_After_Order', 'EOQ_Batch_Count', 'Order_Quantity_EOQ',
        'Total_Order_Quantity', 'Expected_Receipt_Date', 'Receipt_Weekday', 'Forecast_Lead_Demand'
    ])
    svc_opt = options.get('service_level')
    svc_provided = bool(svc_opt is not None and float(svc_opt) > 0.0)
    svc_pct = round(float(svc_opt) * 100.0, 2) if svc_provided else None
    svc_disclaimer = (
        f"and {float(svc_opt)*100:g}% nominal service target"
        if svc_provided else
        "with no service level specified (safety stock = 0; deterministic EOQ and lead-time reorder point used)"
    )
    return clean_json(dict(
        enabled=True,
        plan_start_date=str(start_dt),
        current_stock=cur_stock,
        lead_time=lead,
        order_cost=options['order_cost'],
        holding_cost_unit_year=options['holding_cost_unit_year'],
        service_level_provided=svc_provided,
        service_level_pct=svc_pct,
        unit_name=unit_label,
        data_type=data_label,
        selected_model_label=winner_label,
        EOQ=q,
        safety_stock=safety,
        reorder_point=rop,
        annual_ordering_cost=final_inv['annual_ordering_cost'],
        annual_cycle_holding=final_inv['annual_cycle_holding'],
        annual_buffer_holding=final_inv['annual_buffer_holding'],
        annual_relevant_cost=final_inv['annual_relevant_cost'],
        order_now=order_now,
        order_now_explanation=action_Now,
        expected_order_dates=expected_dates,
        total_planned_orders=len(schedule),
        total_eoq_batches=int(sum(r['EOQ_Batch_Count'] for r in schedule)),
        order_schedule=schedule,
        order_schedule_csv=sched_df.to_csv(index=False),
        forward_simulation_summary=sim_summary,
        forward_simulation_rows=sim_df.to_dict(orient='records'),
        disclaimer=(
            f"This purchase plan depends directly on the selected {winner_label.lower()} forecast for {data_label.lower()} "
            f"and on the exact parameters you entered (start date {start_dt}, current stock {cur_stock:,.2f} {unit_label}, "
            f"lead time {lead} {'calendar days' if options.get('calendar_mode') == 'all_days' else 'operating days'}, order cost {options['order_cost']:,.2f}, annual holding cost "
            f"{options['holding_cost_unit_year']:,.2f} per {unit_label}, {svc_disclaimer})."
        ),
    ))


def analyze(payload, progress=None):
    global WEEK_LENGTH, DAY_CODES, STL_SEASONAL_PERIOD, SARIMA_CANDIDATES
    progress = progress or (lambda message: None)
    progress('Step 1/8: Validating and preparing your data')
    y_imputed, days, options, notes, gaps = validate(payload)
    dates_list = options.pop('_dates_list', None)
    raw_values_with_nan = options.pop('_raw_values_with_nan', None)
    blank_positions = options.pop('_blank_positions', [])
    blank_share_pct = options.pop('_blank_share_pct', 0.0)
    blank_records = options.pop('_blank_records', [])
    has_real_dates = bool(options.get('has_real_dates', False))
    fast_mode = bool(options.get('fast_mode', False))
    WEEK_LENGTH = int(options.get('week_length', 5))
    DAY_CODES = tuple(range(2, 2 + WEEK_LENGTH))
    STL_SEASONAL_PERIOD = WEEK_LENGTH
    SARIMA_CANDIDATES = [
        ((p, d, q), (P, 0, Q, STL_SEASONAL_PERIOD))
        for p in (0, 1) for d in (0, 1) for q in (0, 1) for (P, Q) in ((0, 0), (1, 1))
    ]

    data_label = options['data_type']
    data_lower = data_label.lower()
    unit_label = options['unit_name']
    n = len(y_imputed)
    train = options['train_end']
    dev = options['validation_end']
    lead = options['lead_time']

    # Step 2: Data exploration & causal one-step outlier preparation
    progress('Step 2/8: Exploring STL decomposition, stationarity, distribution, and ACF/PACF')
    outlier_prep = apply_causal_outlier_treatment(
        y_imputed, train_end=train,
        threshold=options['outlier_threshold'],
        treatment=options['outlier_treatment'],
        excluded_indices=[position - 1 for position in blank_positions],
    )
    y = outlier_prep['y_treated']
    if outlier_prep['outlier_count'] > 0:
        pos_preview = ', '.join(str(p) for p in outlier_prep['outlier_positions'][:12])
        notes.append(
            f"Data preparation: flagged {outlier_prep['outlier_count']} causal one-step residual outlier(s) "
            f"(|robust z-score| > {options['outlier_threshold']:g}, rule of thumb; first {OUTLIER_WARMUP} observations are warm-up) at observation(s) {pos_preview}; "
            f"treatment applied: '{options['outlier_treatment']}'."
        )

    stl_info = compute_stl_decomposition(y_imputed, period=STL_SEASONAL_PERIOD)
    stl_df = pd.DataFrame(dict(
        Observation=np.arange(1, n + 1),
        Day_of_week=days,
        Raw_Imputed_Demand=y_imputed,
        Treated_Demand=y,
        STL_Trend=stl_info['trend'],
        STL_Seasonal=stl_info['seasonal'],
        STL_Residual=stl_info['resid'],
        Causal_Modified_Z_Score=outlier_prep['causal_mod_z'],
        Outlier_Flag=outlier_prep['flagged_mask'],
        Outlier_Treatment=options['outlier_treatment'],
    ))
    stationarity_df, stationarity_verdicts, stationarity_summary = compute_stationarity_tests(y, data_label=data_label)
    dist_df, dist_stats = compute_distribution_summary(y, data_label=data_label, unit_label=unit_label)
    demand_acf_pacf_df, demand_acf_ci = compute_demand_acf_pacf(y, max_lag=DEMAND_ACF_MAX_LAG)

    naive_scale = float(np.mean(np.abs(np.diff(y[:train])))) if train > 1 else np.nan
    ns = runpy.run_path(str(Path(__file__).with_name('analysis_core.py')))
    ns['WEEK_LENGTH'] = WEEK_LENGTH
    ns['DAY_CODES'] = DAY_CODES
    ns['WEEKDAY_NAMES'] = WEEKDAY_NAMES
    for value in ns.values():
        if callable(value) and hasattr(value, '__globals__'):
            value.__globals__['WEEK_LENGTH'] = WEEK_LENGTH
            value.__globals__['DAY_CODES'] = DAY_CODES
            value.__globals__['WEEKDAY_NAMES'] = WEEKDAY_NAMES
    ns['PARAMS'].update(options)

    rf_enabled = bool(HAS_SKLEARN and RandomForestRegressor is not None)
    sm_enabled = bool(HAS_STATSMODELS and SMExponentialSmoothing is not None and SMSARIMAX is not None)
    unavailable_models = []
    if not rf_enabled:
        err_detail = f" ({SKLEARN_IMPORT_ERROR})" if SKLEARN_IMPORT_ERROR else ""
        reason_msg = (
            f"Random Forest regression was not run because scikit-learn failed to load in this Python environment{err_detail}."
        )
        unavailable_models.append(dict(
            model='Weekday_RF',
            label=LABELS['Weekday_RF'],
            reason=reason_msg,
        ))
        notes.insert(0, f"Warning: {reason_msg}")

    if not sm_enabled:
        err_detail = f" ({STATSMODELS_IMPORT_ERROR})" if STATSMODELS_IMPORT_ERROR else ""
        sm_reason = (
            f"Holt-Winters seasonal and SARIMA were not run because statsmodels failed to load in this Python environment{err_detail}. "
            "The analysis ran with the remaining models and fallback exploration routines."
        )
        for sm_key in ('Holt_Winters', 'SARIMA'):
            unavailable_models.append(dict(
                model=sm_key,
                label=LABELS[sm_key],
                reason=sm_reason,
            ))
        notes.insert(0, f"Warning: {sm_reason}")

    # Step 3: Single tuning pass for Ridge and Random Forest
    progress('Step 3/8: Tuning moving average, Ridge regression, and Random Forest on training/validation split')
    tuning_info = tune_ridge_and_rf(
        y, days, train, dev,
        dates=dates_list, has_real_dates=has_real_dates,
        naive_scale=naive_scale, fast_mode=fast_mode,
    )
    best_ridge_lambda = tuning_info['best_ridge_lambda']
    best_rf_params = tuning_info['best_rf_params']

    # Step 4: Holt-Winters config & one-time SARIMA AIC search on training split
    progress('Step 4/8: Configuring Holt-Winters and running one-time SARIMA AIC search on training data')
    hw_seasonal_cfg, hw_damped_cfg, hw_cfg_rows = select_holt_winters_config(
        y[:train], y[train:dev],
        hw_seasonal_opt=options['hw_seasonal'],
        hw_damped_opt=options['hw_damped'],
        fast_mode=fast_mode,
    )
    sarima_info = None
    if sm_enabled and not fast_mode:
        sarima_info = search_sarima_order_by_aic(y[:train])
        if sarima_info['skipped_count'] > 0:
            notes.append(
                f"SARIMA AIC search on training data tried {sarima_info['candidates_tried']} candidates, "
                f"skipped {sarima_info['skipped_count']} non-convergent candidate(s), and selected "
                f"SARIMA{sarima_info['order']}x{sarima_info['seasonal_order']} (AIC = {sarima_info['best_aic']:.2f})."
            )

    single_active_models = (
        ['SES', 'SMA', 'Weekday_DR', 'Weekday_Naive', 'Holt', 'Weekday_Ridge']
        + (['Weekday_RF'] if rf_enabled else [])
        + (['Holt_Winters'] if sm_enabled else [])
        + (['SARIMA'] if (sm_enabled and not fast_mode) else [])
    )
    active_models = single_active_models + ['Ensemble']
    method_order = (
        BASE_METHOD_ORDER
        + (['Weekday_RF'] if rf_enabled else [])
        + (['Holt_Winters'] if sm_enabled else [])
        + (['SARIMA'] if (sm_enabled and not fast_mode) else [])
        + ['Ensemble']
    )

    fast_predict, fast_regression_fit, fast_ridge_fit = make_prefix_accelerator(
        y, days,
        ridge_lambda=best_ridge_lambda,
        rf_params=best_rf_params,
        dates_full=dates_list,
        has_real_dates=has_real_dates,
        use_rich_features=True,
        hw_config=(hw_seasonal_cfg, hw_damped_cfg),
        sarima_spec=sarima_info,
        train_end=train,
        dev_end=dev,
    )

    def score_fn(actual, forecast):
        return score_with_scale(actual, forecast, naive_scale=naive_scale)

    ns['fit_ses'] = fit_ses
    ns['predict'] = fast_predict
    ns['regression_fit'] = fast_regression_fit
    ns['score'] = score_fn
    for fn_name in ('choose_window', 'choose_model', 'lead_calibration', 'diagnostics'):
        if fn_name in ns:
            ns[fn_name].__globals__['predict'] = fast_predict
            ns[fn_name].__globals__['regression_fit'] = fast_regression_fit
            ns[fn_name].__globals__['fit_ses'] = fit_ses
            ns[fn_name].__globals__['score'] = score_fn

    # Step 5: Window selection, single-model validation, Top-3 Ensemble construction, and overall validation selection
    progress('Step 5/8: Comparing validation forecasts and constructing the top-3 validation ensemble')
    window, tuning = ns['choose_window'](y[:train], days[:train])
    ns[' choose_model_single'] = single_active_models
    ns['choose_model'].__globals__['MODELS'] = single_active_models
    _, val_single_df = ns['choose_model'](y, days, window)
    ensemble_members = [str(m) for m in val_single_df.iloc[:ENSEMBLE_TOP_K]['Model'].tolist()]
    fast_predict.set_ensemble_members(ensemble_members)

    ns['MODELS'] = active_models
    ns['choose_model'].__globals__['MODELS'] = active_models
    winner, validation = ns['choose_model'](y, days, window)

    ens_val_row = validation[validation.Model == 'Ensemble'].iloc[0]
    ensemble_weights_df = pd.DataFrame([
        dict(
            Rank=idx_e + 1,
            Member_Model=m_key,
            Member_Label=LABELS[m_key],
            Weight=1.0 / len(ensemble_members),
            Member_Validation_MAE=float(val_single_df[val_single_df.Model == m_key].iloc[0].MAE),
            Ensemble_Validation_MAE=float(ens_val_row.MAE),
        )
        for idx_e, m_key in enumerate(ensemble_members)
    ])

    try:
        calibration = ns['lead_calibration'](y[:dev], days[:dev], winner, window, lead)
    except ValueError:
        raise ValueError('Fewer than three valid cumulative-error windows remain. Use more consecutive weekday records or reduce the lead time.')

    # Step 6: Final test evaluation, intervals, DM tests, and CV
    models = active_models
    predict = fast_predict
    score = score_fn
    progress(f'Step 6/8: Evaluating all {len(models)} forecasting methods, intervals, DM tests, and CV folds')
    test = pd.DataFrame(dict(Observation=np.arange(dev + 1, n + 1), Actual=y[dev:]))
    metrics = []
    origin_rows = []
    test_errors = {}
    for model in models:
        rolling = []
        for i in range(dev, n):
            rolling.append(predict(model, y[:i], days[:i], [days[i]], window)[0])
        fixed = predict(model, y[:dev], days[:dev], days[dev:], window)
        test[model] = rolling
        test[model + '_fixed'] = fixed
        test_errors[model] = y[dev:] - np.asarray(rolling, dtype=float)
        metrics.extend([
            dict(Model=model, Protocol='Rolling one step', **score(y[dev:], rolling)),
            dict(Model=model, Protocol=f'Fixed {n - dev} step', **score(y[dev:], fixed)),
        ])
        multi = []
        for origin in range(dev, n, lead):
            end = min(origin + lead, n)
            forecast = predict(model, y[:origin], days[:origin], days[origin:end], window)
            multi.extend(forecast)
            origin_rows.append(dict(Model=model, Origin=origin, Horizon=end - origin, **score(y[origin:end], forecast)))
        metrics.append(dict(Model=model, Protocol=f'Blocks of {lead} steps', **score(y[dev:], multi)))
    metrics = pd.DataFrame(metrics)

    dm_rows = []
    for m in method_order:
        for loss_type in ('absolute', 'squared'):
            dm_rows.append(diebold_mariano_pair(test_errors[m], test_errors['Weekday_Naive'], m, 'Weekday_Naive', loss=loss_type))
            if winner != 'Weekday_Naive':
                dm_rows.append(diebold_mariano_pair(test_errors[m], test_errors[winner], m, winner, loss=loss_type))
    dm_df = pd.DataFrame(dm_rows)

    future_days = ((int(days[-1]) - 2 + np.arange(1, options['horizon'] + 1)) % WEEK_LENGTH) + 2
    future_weekday_names = [WEEKDAY_NAMES.get(int(d), str(int(d))) for d in future_days]
    future_cols = dict(
        Forecast_Step=np.arange(1, len(future_days) + 1),
        Weekday_Code=future_days,
        Weekday_Name=future_weekday_names,
    )
    if has_real_dates and dates_list:
        future_cols['Forecast_Date'] = [str(advance_business_days(dates_list[-1], int(s))) for s in range(1, len(future_days) + 1)]
    elif options.get('plan_start_date'):
        b_fut_dates = generate_business_dates(date.fromisoformat(options['plan_start_date']), len(future_days))
        future_cols['Forecast_Date'] = [str(d) for d in b_fut_dates]
    future = pd.DataFrame(future_cols)
    for model in models:
        future[model] = predict(model, y, days, future_days, window)

    interval_info = compute_intervals_and_coverage(
        y, days, train, dev, models, predict, window, future,
        min_errors=options['min_interval_errors']
    )
    coverage_df = evaluate_test_coverage(test, models, interval_info)
    future['Backtest_Error_Count'] = interval_info['error_counts']
    for m in models:
        fi = interval_info['future_intervals'][m]
        future[f'{m}_lower_80'] = fi['lo80']
        future[f'{m}_upper_80'] = fi['hi80']
        future[f'{m}_lower_95'] = fi['lo95']
        future[f'{m}_upper_95'] = fi['hi95']

    cv_fold_df, cv_summary_df, cv_winner, cv_k = rolling_origin_cv(
        y, days, train, models, window, K=options['cv_folds'],
        fast_regression_fit=fast_regression_fit, fast_ridge_fit=fast_ridge_fit,
        dates=dates_list, has_real_dates=has_real_dates, use_rich_features=True,
        ridge_lambda=best_ridge_lambda, rf_params=best_rf_params,
        hw_config=(hw_seasonal_cfg, hw_damped_cfg),
        sarima_spec=sarima_info, ensemble_members=ensemble_members,
    )
    cv_warning = None
    if cv_winner != winner:
        cv_win_row = cv_summary_df[cv_summary_df.Model == cv_winner].iloc[0]
        cv_warning = (
            f"Cross-validation prefers {LABELS[cv_winner]} (wins {int(cv_win_row.Folds_Won)} of {cv_k} folds, "
            f"mean CV MAE {float(cv_win_row.Mean_MAE):,.2f} {unit_label}) while validation selected {LABELS[winner]} "
            f"(validation MAE {float(validation.iloc[0].MAE):,.2f} {unit_label}). "
            "The model selection rule remains validation MAE; see the Diebold–Mariano table to check whether the difference is statistically significant."
        )
        notes.insert(0, f"Cross-validation note: {cv_warning}")

    # Step 7: Old-vs-New feature comparison, Feature importance, Raw-vs-Treated comparison, and Inventory
    progress('Step 7/8: Computing feature importance, feature/treatment comparisons, and inventory metrics')
    importance_info = compute_feature_importance(
        y, days, train, dev,
        dates=dates_list, has_real_dates=has_real_dates,
        ridge_lambda=best_ridge_lambda, rf_params=best_rf_params,
    )

    feature_comp_df = None
    feature_comp_summary = None
    if not fast_mode:
        base_pred_fn, _, _ = make_prefix_accelerator(
            y, days, ridge_lambda=FIXED_RIDGE_LAMBDA, rf_params=FIXED_RF_PARAMS,
            dates_full=dates_list, has_real_dates=has_real_dates, use_rich_features=False,
            train_end=train, dev_end=dev,
        )
        fc_rows = []
        verdict_parts = []
        for m_key in (['Weekday_Ridge'] + (['Weekday_RF'] if rf_enabled else [])):
            # Baseline features
            b_val_p = [base_pred_fn(m_key, y[:i], days[:i], [days[i]], window)[0] for i in range(train, dev)]
            b_test_p = [base_pred_fn(m_key, y[:i], days[:i], [days[i]], window)[0] for i in range(dev, n)]
            b_val_sc = score_fn(y[train:dev], b_val_p)
            b_test_sc = score_fn(y[dev:], b_test_p)
            # Rich features (already computed in validation and test)
            r_val_row = validation[validation.Model == m_key].iloc[0]
            r_test_row = metrics[(metrics.Model == m_key) & (metrics.Protocol == 'Rolling one step')].iloc[0]
            val_diff = float(r_val_row.MAE) - b_val_sc['MAE']
            test_diff = float(r_test_row.MAE) - b_test_sc['MAE']
            helped_val = val_diff < -1e-9
            helped_test = test_diff < -1e-9
            if helped_val and helped_test:
                honest_v = f"New features helped {LABELS[m_key]} on both validation ({val_diff:+.2f} {unit_label}) and test ({test_diff:+.2f} {unit_label})."
            elif helped_val and not helped_test:
                honest_v = f"New features lowered validation MAE ({val_diff:+.2f} {unit_label}) but did not improve final test MAE ({test_diff:+.2f} {unit_label})."
            elif not helped_val and helped_test:
                honest_v = f"New features did not help on validation ({val_diff:+.2f} {unit_label}) despite lower test MAE ({test_diff:+.2f} {unit_label})."
            else:
                honest_v = f"Honest report: new features did not help {LABELS[m_key]} on validation ({val_diff:+.2f} {unit_label}) or test ({test_diff:+.2f} {unit_label})."
            verdict_parts.append(honest_v)
            fc_rows.append(dict(
                Model=m_key,
                Model_Label=LABELS[m_key],
                Feature_Set='Baseline features (Lag_1 + Weekday)',
                Feature_Count=5,
                Validation_MAE=b_val_sc['MAE'],
                Validation_RMSE=b_val_sc['RMSE'],
                Validation_MASE=b_val_sc['MASE'],
                Validation_sMAPE_pct=b_val_sc['sMAPE_pct'],
                Test_MAE=b_test_sc['MAE'],
                Test_RMSE=b_test_sc['RMSE'],
                Test_MASE=b_test_sc['MASE'],
                Test_sMAPE_pct=b_test_sc['sMAPE_pct'],
                Val_MAE_Change_vs_Baseline=0.0,
                Test_MAE_Change_vs_Baseline=0.0,
                Verdict='Baseline reference',
            ))
            fc_rows.append(dict(
                Model=m_key,
                Model_Label=LABELS[m_key],
                Feature_Set=f"New richer features ({len(rich_feature_names(has_real_dates))} features, tuned)",
                Feature_Count=len(rich_feature_names(has_real_dates)),
                Validation_MAE=float(r_val_row.MAE),
                Validation_RMSE=float(r_val_row.RMSE),
                Validation_MASE=float(r_val_row.MASE),
                Validation_sMAPE_pct=float(r_val_row.sMAPE_pct),
                Test_MAE=float(r_test_row.MAE),
                Test_RMSE=float(r_test_row.RMSE),
                Test_MASE=float(r_test_row.MASE),
                Test_sMAPE_pct=float(r_test_row.sMAPE_pct),
                Val_MAE_Change_vs_Baseline=val_diff,
                Test_MAE_Change_vs_Baseline=test_diff,
                Verdict=honest_v,
            ))
        feature_comp_df = pd.DataFrame(fc_rows)
        feature_comp_summary = ' '.join(verdict_parts)

    # Raw vs Treated comparison when a treatment was applied
    treatment_applied = bool(len(blank_positions) > 0 or (options['outlier_treatment'] != 'keep' and outlier_prep['outlier_count'] > 0))
    treatment_comp_df = None
    if treatment_applied and not fast_mode:
        # Compare selected model on raw (untreated outlier series y_imputed) vs treated series (y)
        raw_pred_fn, _, _ = make_prefix_accelerator(
            y_imputed, days,
            ridge_lambda=best_ridge_lambda, rf_params=best_rf_params,
            dates_full=dates_list, has_real_dates=has_real_dates, use_rich_features=True,
            hw_config=(hw_seasonal_cfg, hw_damped_cfg), sarima_spec=sarima_info,
            train_end=train, dev_end=dev,
        )
        raw_pred_fn.set_ensemble_members(ensemble_members)
        raw_naive_scale = float(np.mean(np.abs(np.diff(y_imputed[:train])))) if train > 1 else np.nan
        raw_val_preds = [raw_pred_fn(winner, y_imputed[:i], days[:i], [days[i]], window)[0] for i in range(train, dev)]
        raw_test_preds = [raw_pred_fn(winner, y_imputed[:i], days[:i], [days[i]], window)[0] for i in range(dev, n)]
        raw_val_sc = score_with_scale(y_imputed[train:dev], raw_val_preds, naive_scale=raw_naive_scale)
        raw_test_sc = score_with_scale(y_imputed[dev:], raw_test_preds, naive_scale=raw_naive_scale)

        tr_val_row = validation[validation.Model == winner].iloc[0]
        tr_test_row = metrics[(metrics.Model == winner) & (metrics.Protocol == 'Rolling one step')].iloc[0]
        # Also evaluate treated-history forecasts against untouched raw test actuals y_imputed[dev:]
        tr_vs_raw_test_sc = score_with_scale(y_imputed[dev:], test[winner].to_numpy(dtype=float), naive_scale=raw_naive_scale)

        treatment_comp_df = pd.DataFrame([
            dict(
                Series_Variant='Raw (before outlier treatment)',
                Selected_Model=winner,
                Selected_Model_Label=LABELS[winner],
                Validation_MAE=raw_val_sc['MAE'],
                Validation_RMSE=raw_val_sc['RMSE'],
                Validation_WAPE_pct=raw_val_sc['WAPE_pct'],
                Validation_MASE=raw_val_sc['MASE'],
                Validation_sMAPE_pct=raw_val_sc['sMAPE_pct'],
                Test_MAE=raw_test_sc['MAE'],
                Test_RMSE=raw_test_sc['RMSE'],
                Test_WAPE_pct=raw_test_sc['WAPE_pct'],
                Test_MASE=raw_test_sc['MASE'],
                Test_sMAPE_pct=raw_test_sc['sMAPE_pct'],
            ),
            dict(
                Series_Variant=f"Treated (missing={options['missing_treatment']}, outlier={options['outlier_treatment']})",
                Selected_Model=winner,
                Selected_Model_Label=LABELS[winner],
                Validation_MAE=float(tr_val_row.MAE),
                Validation_RMSE=float(tr_val_row.RMSE),
                Validation_WAPE_pct=float(tr_val_row.WAPE_pct),
                Validation_MASE=float(tr_val_row.MASE),
                Validation_sMAPE_pct=float(tr_val_row.sMAPE_pct),
                Test_MAE=float(tr_test_row.MAE),
                Test_RMSE=float(tr_test_row.RMSE),
                Test_WAPE_pct=float(tr_test_row.WAPE_pct),
                Test_MASE=float(tr_test_row.MASE),
                Test_sMAPE_pct=float(tr_test_row.sMAPE_pct),
            ),
            dict(
                Series_Variant='Treated model evaluated against untouched raw actuals',
                Selected_Model=winner,
                Selected_Model_Label=LABELS[winner],
                Validation_MAE=float(tr_val_row.MAE),
                Validation_RMSE=float(tr_val_row.RMSE),
                Validation_WAPE_pct=float(tr_val_row.WAPE_pct),
                Validation_MASE=float(tr_val_row.MASE),
                Validation_sMAPE_pct=float(tr_val_row.sMAPE_pct),
                Test_MAE=tr_vs_raw_test_sc['MAE'],
                Test_RMSE=tr_vs_raw_test_sc['RMSE'],
                Test_WAPE_pct=tr_vs_raw_test_sc['WAPE_pct'],
                Test_MASE=tr_vs_raw_test_sc['MASE'],
                Test_sMAPE_pct=tr_vs_raw_test_sc['sMAPE_pct'],
            ),
        ])

    max_sup_h = interval_info['max_supported_h']
    min_err_rule = interval_info['min_errors_rule']
    if max_sup_h < options['horizon']:
        notes.append(
            f"Prediction interval rule of thumb: 80% and 95% empirical prediction intervals are shown up to forecast step {max_sup_h} "
            f"because steps {max_sup_h + 1}–{options['horizon']} have fewer than {min_err_rule} rolling backtest errors (available backtest errors start at {int(interval_info['error_counts'][0])} for step 1 and decrease by 1 per step)."
        )

    dev_days = ((int(days[dev - 1]) - 2 + np.arange(1, options['horizon'] + 1)) % WEEK_LENGTH) + 2
    dev_forecast = predict(winner, y[:dev], days[:dev], dev_days, window)
    if min(dev_forecast.mean(), future[winner].mean()) <= 1e-10:
        raise ValueError(f'The selected forecast is zero over the planning horizon. Planning needs positive expected {data_lower}; check your data or choose a longer horizon.')
    frozen = ns['inventory'](dev_forecast.mean(), dev_forecast[:lead].sum(), calibration.Error.to_numpy(), options['service_level'])
    final = ns['inventory'](future[winner].mean(), future[winner].iloc[:lead].sum(), calibration.Error.to_numpy(), options['service_level'])
    if np.sum(y[dev:]) * 1.3 / frozen['EOQ'] > 20000:
        raise ValueError(f'These cost and {data_lower} settings would create more than 20,000 simulated orders. Increase the ordering cost, reduce the holding cost, or rescale {data_lower}.')
    if np.sum(y[dev:]) <= 0:
        raise ValueError(f'The final test block has zero total {data_lower}. Use a test period containing positive {data_lower} for evaluation.')
    policy, policy_summary, events = ns['simulate_policy'](y[dev:], frozen['EOQ'], frozen['reorder_point'], lead)
    _, stress, _ = ns['simulate_policy'](y[dev:] * 1.3, frozen['EOQ'], frozen['reorder_point'], lead)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        coef, diag, residual, fitted = diagnostics(ns, y[:dev], days[:dev])
    beta, X, _, _ = ns['regression_fit'](y[:dev], days[:dev])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        notes.append('The development regression is rank-deficient. Coefficients are not separately identifiable; interpret forecasts and diagnostics cautiously.')
    if n > MAX_PLOT_POINTS:
        notes.append(f'Time-series charts retain the observations; the calendar view focuses on recent weeks for readability. All forecast models, metrics and CSV tables use all {n:,} observations.')
    if options.get('service_level') is not None:
        notes.append(f'The uncertainty buffer uses {len(calibration)} overlapping, post-selection development windows. It does not validate a {options["service_level"]*100:g}% real service level.')
    else:
        notes.append('No service level was specified for inventory planning, so safety stock is set to 0 and the reorder point equals expected lead-time demand.')
    notes.append(f'The simulation contains {policy_summary["Completed_Cycles"]} completed receipt cycles. Initial seeded stock and terminal censoring limit cycle-service interpretation.')
    notes.append(f'Inventory calculations express quantities in {unit_label} using fixed lead time, uniform within-day consumption and user-specified costs. The future forecast is a model scenario, not observed future {data_lower}.')

    fast_mode_skipped = dict(
        models=['SARIMA'] if fast_mode else [],
        tuning_grids=['Ridge penalty grid (10 values)', 'Random Forest hyperparameter grid (8 settings)'] if fast_mode else [],
        tables=['feature_comparison', 'treatment_comparison'] if fast_mode else [],
    )
    fast_mode_message = (
        "Fast mode skipped SARIMA, the tuning grids (using fixed Ridge and Random Forest settings), "
        "and two comparison tables (old-vs-new feature comparison and raw-vs-treated comparison). "
        "Turn Fast mode off in Step 2 to run them."
        if fast_mode else None
    )
    if fast_mode_message:
        notes.insert(0, fast_mode_message)

    purchase_plan = build_purchase_plan(
        future[winner].to_numpy(float),
        LABELS[winner],
        final,
        options,
        data_label,
        unit_label,
        ns['simulate_policy'],
    )

    prep_log_rows = []
    for br in blank_records:
        prep_log_rows.append(dict(
            Observation=br['Observation'],
            Type='Blank demand imputed',
            Raw_Value='',
            Treated_Value=br['Filled_Demand'],
            Method=br['Missing_Treatment'],
            Detail=f"Weekday {br['Weekday']}",
        ))
    for orow in outlier_prep['outlier_records']:
        prep_log_rows.append(dict(
            Observation=orow['Observation'],
            Type='Causal one-step residual outlier flagged',
            Raw_Value=orow['Raw_Demand'],
            Treated_Value=orow['Treated_Demand'],
            Method=orow['Treatment_Action'],
            Detail=f"Robust z={orow['Modified_Z_Score']:+.2f} (threshold ±{orow['Threshold']:g})",
        ))
    prep_log_df = pd.DataFrame(prep_log_rows) if prep_log_rows else pd.DataFrame(columns=[
        'Observation', 'Type', 'Raw_Value', 'Treated_Value', 'Method', 'Detail'
    ])

    raw_input_df = pd.DataFrame(dict(
        Observation=np.arange(1, n + 1),
        Day_of_week=days,
        Weekday_Name=[WEEKDAY_NAMES.get(int(d), str(int(d))) for d in days],
        Raw_Demand=raw_values_with_nan,
    ))
    input_df = pd.DataFrame(dict(
        Observation=np.arange(1, n + 1),
        Day_of_week=days,
        Weekday_Name=[WEEKDAY_NAMES.get(int(d), str(int(d))) for d in days],
        Total_Demand=y,
    ))

    # Build weekly forecast roll-up using the selected operating calendar.
    weekly_rows = []
    hz_total = len(future)
    n_weeks = int(math.ceil(hz_total / float(WEEK_LENGTH)))
    for w_idx in range(n_weeks):
        s_i = w_idx * WEEK_LENGTH
        e_i = min(hz_total, (w_idx + 1) * WEEK_LENGTH)
        sub_f = future.iloc[s_i:e_i]
        win_vals = sub_f[winner].to_numpy(dtype=float)
        lo80_w = sub_f[f'{winner}_lower_80'].to_numpy(dtype=float)
        hi80_w = sub_f[f'{winner}_upper_80'].to_numpy(dtype=float)
        lo95_w = sub_f[f'{winner}_lower_95'].to_numpy(dtype=float)
        hi95_w = sub_f[f'{winner}_upper_95'].to_numpy(dtype=float)
        w_row = dict(
            Operating_Week=w_idx + 1,
            Step_Range=f'Days {s_i + 1}–{e_i}',
            Operating_Days=int(e_i - s_i),
            Selected_Model=LABELS[winner],
            Weekly_Total=float(np.sum(win_vals)),
            Daily_Mean=float(np.mean(win_vals)),
            Min_Day=float(np.min(win_vals)),
            Max_Day=float(np.max(win_vals)),
            Lower_80_Total=float(np.sum(lo80_w)) if np.all(np.isfinite(lo80_w)) else None,
            Upper_80_Total=float(np.sum(hi80_w)) if np.all(np.isfinite(hi80_w)) else None,
            Lower_95_Total=float(np.sum(lo95_w)) if np.all(np.isfinite(lo95_w)) else None,
            Upper_95_Total=float(np.sum(hi95_w)) if np.all(np.isfinite(hi95_w)) else None,
        )
        if 'Forecast_Date' in sub_f.columns:
            w_row['Date_Range'] = f"{sub_f['Forecast_Date'].iloc[0]} to {sub_f['Forecast_Date'].iloc[-1]}"
        for m_k in models:
            w_row[f'{m_k}_Weekly_Total'] = float(sub_f[m_k].sum())
        weekly_rows.append(w_row)
    weekly_forecast_df = pd.DataFrame(weekly_rows)

    val_lookup = {row.Model: row for _, row in validation.iterrows()}
    test_roll_lookup = {row.Model: row for _, row in metrics[metrics.Protocol == 'Rolling one step'].iterrows()}
    cov_lookup = {row.Model: row for _, row in coverage_df.iterrows()}
    cv_sum_lookup_all = {row.Model: row for _, row in cv_summary_df.iterrows()}

    # Build 10-Model Leaderboard Table sorted by Validation MAE
    leaderboard_rows = []
    for rank_i, (_, v_row) in enumerate(validation.iterrows(), start=1):
        m_k = str(v_row.Model)
        t_row = test_roll_lookup[m_k]
        cv_r = cv_sum_lookup_all.get(m_k)
        role_str = 'Selected Winner' if m_k == winner else ('Top-3 Ensemble Member' if m_k in ensemble_members else ('Top-3 Ensemble' if m_k == 'Ensemble' else 'Candidate'))
        leaderboard_rows.append(dict(
            Validation_Rank=rank_i,
            Model=m_k,
            Model_Label=LABELS[m_k],
            Role=role_str,
            Validation_MAE=float(v_row.MAE),
            Validation_WAPE_pct=float(v_row.WAPE_pct),
            Test_MAE=float(t_row.MAE),
            Test_RMSE=float(t_row.RMSE),
            Test_WAPE_pct=float(t_row.WAPE_pct),
            Test_MASE=float(t_row.MASE),
            Test_sMAPE_pct=float(t_row.sMAPE_pct),
            Test_Bias=float(t_row.Bias),
            Tracking_Signal=float(getattr(t_row, 'Tracking_Signal', 0.0)),
            CV_Mean_MAE=float(cv_r.Mean_MAE) if cv_r is not None else None,
        ))
    leaderboard_df = pd.DataFrame(leaderboard_rows)

    frames = {
        'model_leaderboard': leaderboard_df,
        'weekly_forecast_summary': weekly_forecast_df,
        'input_data': input_df,
        'raw_input_data': raw_input_df,
        'data_preparation_log': prep_log_df,
        'descriptive_statistics': pd.DataFrame({data_label: pd.Series(y).describe()}),
        'stl_decomposition': stl_df,
        'stationarity_tests': stationarity_df,
        'distribution_summary': dist_df,
        'demand_acf_pacf': demand_acf_pacf_df,
        'sma_tuning': tuning,
        'model_tuning': tuning_info['tuning_df'],
        'feature_importance': importance_info['table_df'],
        'ensemble_weights': ensemble_weights_df,
        'validation_metrics': validation,
        'model_metrics': metrics,
        'test_predictions': test,
        'forecast': future,
        'forecast_5_steps': future.head(5),
        'interval_coverage': coverage_df,
        'cv_fold_metrics': cv_fold_df,
        'cv_summary': cv_summary_df,
        'diebold_mariano': dm_df,
        'lead_time_errors': calibration,
        'weekday_gaps': pd.DataFrame(gaps),
        'inventory_metrics': pd.DataFrame([dict(Basis='Frozen before final test', **frozen), dict(Basis='Full history planning', **final)]),
        'policy_simulation': policy,
        'policy_events': events,
        'policy_summary': pd.DataFrame([dict(Scenario=f'Recorded test {data_lower}', **policy_summary), dict(Scenario=f'{data_label} multiplied by 1.30', **stress)]),
        'regression_coefficients': coef,
        'multi_origin_metrics': pd.DataFrame(origin_rows),
    }
    if feature_comp_df is not None:
        frames['feature_comparison'] = feature_comp_df
    if treatment_comp_df is not None:
        frames['treatment_comparison'] = treatment_comp_df
    if sarima_info is not None and sarima_info.get('table_df') is not None:
        frames['sarima_aic_search'] = sarima_info['table_df']

    sensitivity = []
    for level in [None, .90, .95, .99]:
        inv_l = ns['inventory'](future[winner].mean(), future[winner].iloc[:lead].sum(), calibration.Error.to_numpy(), level)
        sensitivity.append(dict(
            Service_Target=0.0 if level is None else level,
            Service_Label='No buffer (0%)' if level is None else f'{int(level * 100)}%',
            **inv_l,
        ))
    frames['service_sensitivity'] = pd.DataFrame(sensitivity)
    winner_test = metrics[(metrics.Model == winner) & (metrics.Protocol == 'Rolling one step')].iloc[0].to_dict()

    new_models_added = [m for m in ['Holt', 'Weekday_Ridge', 'Weekday_RF', 'Holt_Winters', 'SARIMA', 'Ensemble'] if m in models]
    best_new_model = min(new_models_added, key=lambda m: val_lookup[m].MAE)
    new_beat_baseline = val_lookup[best_new_model].MAE < min(val_lookup[m].MAE for m in ['SES', 'SMA', 'Weekday_DR', 'Weekday_Naive'])

    means_by_wk = pd.DataFrame(dict(day=days, y=y)).groupby('day').y.mean().reindex(DAY_CODES)
    max_wk_d = int(means_by_wk.idxmax())
    min_wk_d = int(means_by_wk.idxmin())
    wk_spread_pct = ((float(means_by_wk[max_wk_d]) - float(means_by_wk[min_wk_d])) / float(y.mean()) * 100.0) if float(y.mean()) > 0 else 0.0

    executive_summary = dict(
        winner_model=winner,
        winner_label=LABELS[winner],
        validation_mae=float(validation.iloc[0].MAE),
        test_mae=float(winner_test['MAE']),
        test_wape_pct=float(winner_test['WAPE_pct']),
        test_mase=float(winner_test['MASE']),
        test_bias=float(winner_test['Bias']),
        tracking_signal=float(winner_test.get('Tracking_Signal', 0.0)),
        next_5_days_total=float(future[winner].iloc[:min(5, len(future))].sum()),
        next_5_days_mean=float(future[winner].iloc[:min(5, len(future))].mean()),
        horizon_days=int(options['horizon']),
        horizon_total=float(future[winner].sum()),
        horizon_mean=float(future[winner].mean()),
        peak_weekday=WEEKDAY_NAMES[max_wk_d],
        peak_weekday_mean=float(means_by_wk[max_wk_d]),
        low_weekday=WEEKDAY_NAMES[min_wk_d],
        low_weekday_mean=float(means_by_wk[min_wk_d]),
        weekday_spread_pct=wk_spread_pct,
        eoq=float(final['EOQ']),
    )

    raw_sha256 = hashlib.sha256(raw_input_df[['Observation', 'Day_of_week', 'Raw_Demand']].to_csv(index=False).encode()).hexdigest()
    input_sha256_str = hashlib.sha256(input_df[['Observation', 'Day_of_week', 'Total_Demand']].to_csv(index=False).encode()).hexdigest()
    summary = clean_json(dict(
        n=n,
        split=dict(train=train, validation=dev - train, test=n - dev, development_end=dev),
        data_type=data_label,
        unit_name=unit_label,
        display_label=format_label_unit(data_label, unit_label),
        has_real_dates=has_real_dates,
        fast_mode=fast_mode,
        fast_mode_row_threshold=options['fast_mode_row_threshold'],
        fast_mode_skipped=fast_mode_skipped,
        fast_mode_message=fast_mode_message,
        naive_scale_train=naive_scale,
        models_evaluated=models,
        model_labels={m: LABELS[m] for m in models},
        unavailable_models=unavailable_models,
        new_models_beat_baseline=bool(new_beat_baseline),
        selected_model=winner,
        selected_label=LABELS[winner],
        sma_window=window,
        selected_validation_MAE=float(validation.iloc[0].MAE),
        test_metrics=winner_test,
        executive_summary=executive_summary,
        leaderboard=leaderboard_rows,
        ensemble_members=ensemble_members,
        ensemble_member_labels=[LABELS[m] for m in ensemble_members],
        ensemble_weights=[1.0 / len(ensemble_members)] * len(ensemble_members),
        tuning=dict(
            fast_mode_skipped=fast_mode,
            best_ridge_lambda=best_ridge_lambda,
            best_rf_params=best_rf_params,
            ridge_candidates_tested=1 if fast_mode else len(RIDGE_LAMBDA_GRID),
            rf_candidates_tested=(1 if fast_mode else len(RF_PARAM_GRID)) if rf_enabled else 0,
            feature_names=rich_feature_names(has_real_dates),
            feature_comparison_summary=feature_comp_summary,
        ),
        feature_importance=dict(
            top_ridge_feature=importance_info['top_ridge_feature'],
            top_ridge_coef=importance_info['top_ridge_coef'],
            top_rf_feature=importance_info['top_rf_feature'],
            top_rf_perm_mae=importance_info['top_rf_perm_mae'],
            warning=importance_info['warning'],
        ),
        exploration=dict(
            stationarity_verdicts=stationarity_verdicts,
            stationarity_summary=stationarity_summary,
            distribution=dist_stats,
            acf_ci_95=demand_acf_ci,
        ),
        data_preparation=dict(
            missing_treatment=options['missing_treatment'],
            max_missing_pct_rule=options['max_missing_pct'],
            blank_count=len(blank_positions),
            blank_positions=blank_positions,
            blank_share_pct=blank_share_pct,
            outlier_treatment=options['outlier_treatment'],
            outlier_threshold_rule=options['outlier_threshold'],
            outlier_warmup_rule=OUTLIER_WARMUP,
            outlier_count=outlier_prep['outlier_count'],
            outlier_positions=outlier_prep['outlier_positions'],
            treatment_applied=treatment_applied,
        ),
        holt_winters=dict(
            enabled=sm_enabled,
            seasonal=hw_seasonal_cfg,
            damped_trend=hw_damped_cfg,
            seasonal_periods=STL_SEASONAL_PERIOD,
        ),
        sarima=dict(
            enabled=bool(sm_enabled and not fast_mode),
            skipped_by_fast_mode=fast_mode,
            order=list(sarima_info['order']) if sarima_info else None,
            seasonal_order=list(sarima_info['seasonal_order']) if sarima_info else None,
            best_aic=sarima_info['best_aic'] if sarima_info else None,
            candidates_tried=sarima_info['candidates_tried'] if sarima_info else 0,
            skipped_count=sarima_info['skipped_count'] if sarima_info else 0,
            converged_count=sarima_info['converged_count'] if sarima_info else 0,
        ),
        cv_winner=cv_winner,
        cv_winner_label=LABELS[cv_winner],
        cv_folds=cv_k,
        cv_warning=cv_warning,
        interval_max_supported_horizon=max_sup_h,
        interval_min_errors_rule=min_err_rule,
        final_inventory=final,
        frozen_inventory=frozen,
        policy=policy_summary,
        stress_policy=stress,
        lead_calibration_N=len(calibration),
        diagnostics=diag,
        parameters=options,
        warnings=notes,
        raw_input_sha256=raw_sha256,
        input_sha256=input_sha256_str,
    ))

    progress('Step 8/8: Drawing charts and generating reading guides')
    files = {}
    figures = []
    figures.append(dict(key='01_demand', group='Data', part=1, path='figures/01_demand.png', accent_color=GRAPH_ACCENTS['01_demand']))
    figures.append(dict(key='02_evaluation', group='Evaluation', part=1, path='figures/02_evaluation.png', accent_color=GRAPH_ACCENTS['02_evaluation']))
    figures.append(dict(key='03_future', group='Forecasts', part=1, path='figures/03_future.png', accent_color=GRAPH_ACCENTS['03_future']))
    figures.append(dict(key='04_SES', group='Forecasts', part=1, path='figures/04_SES.png', accent_color=GRAPH_ACCENTS['04_SES']))
    figures.append(dict(key='04_SMA', group='Forecasts', part=1, path='figures/04_SMA.png', accent_color=GRAPH_ACCENTS['04_SMA']))
    figures.append(dict(key='04_Weekday_DR', group='Forecasts', part=1, path='figures/04_Weekday_DR.png', accent_color=GRAPH_ACCENTS['04_Weekday_DR']))
    figures.append(dict(key='04_Weekday_Naive', group='Forecasts', part=1, path='figures/04_Weekday_Naive.png', accent_color=GRAPH_ACCENTS['04_Weekday_Naive']))
    figures.append(dict(key='05_weekday', group='Data', part=1, path='figures/05_weekday.png', accent_color=GRAPH_ACCENTS['05_weekday']))
    figures.append(dict(key='06_errors', group='Evaluation', part=1, path='figures/06_errors.png', accent_color=GRAPH_ACCENTS['06_errors']))
    figures.append(dict(key='07_residuals', group='Diagnostics', part=1, path='figures/07_residuals.png', accent_color=GRAPH_ACCENTS['07_residuals']))
    figures.append(dict(key='08_acf', group='Diagnostics', part=1, path='figures/08_acf.png', accent_color=GRAPH_ACCENTS['08_acf']))
    figures.append(dict(key='09_lead_errors', group='Diagnostics', part=1, path='figures/09_lead_errors.png', accent_color=GRAPH_ACCENTS['09_lead_errors']))
    figures.append(dict(key='10_policy', group='Inventory', part=2, path='figures/10_policy.png', accent_color=GRAPH_ACCENTS['10_policy']))
    figures.append(dict(key='11_cost', group='Inventory', part=2, path='figures/11_cost.png', accent_color=GRAPH_ACCENTS['11_cost']))
    figures.append(dict(key='12_service', group='Inventory', part=2, path='figures/12_service.png', accent_color=GRAPH_ACCENTS['12_service']))
    figures.append(dict(key='13_coverage', group='Evaluation', part=1, path='figures/13_coverage.png', accent_color=GRAPH_ACCENTS['13_coverage']))
    figures.append(dict(key='14_cv_folds', group='Evaluation', part=1, path='figures/14_cv_folds.png', accent_color=GRAPH_ACCENTS['14_cv_folds']))
    figures.append(dict(key='15_dm_test', group='Evaluation', part=1, path='figures/15_dm_test.png', accent_color=GRAPH_ACCENTS['15_dm_test']))
    figures.append(dict(key='16_feature_importance', group='Diagnostics', part=1, path='figures/16_feature_importance.png', accent_color=GRAPH_ACCENTS['16_feature_importance']))
    figures.append(dict(key='17_stl_decomposition', group='Exploration', part=1, path='figures/17_stl_decomposition.png', accent_color=GRAPH_ACCENTS['17_stl_decomposition']))
    figures.append(dict(key='18_stationarity', group='Exploration', part=1, path='figures/18_stationarity.png', accent_color=GRAPH_ACCENTS['18_stationarity']))
    figures.append(dict(key='19_distribution', group='Exploration', part=1, path='figures/19_distribution.png', accent_color=GRAPH_ACCENTS['19_distribution']))
    figures.append(dict(key='20_demand_acf_pacf', group='Exploration', part=1, path='figures/20_demand_acf_pacf.png', accent_color=GRAPH_ACCENTS['20_demand_acf_pacf']))
    lead_plot_values = {}
    lead_means = {}
    for m in method_order:
        cal_m = calibration if m == winner else ns['lead_calibration'](y[:dev], days[:dev], m, window, lead)
        lead_plot_values[m] = cal_m
        lead_means[m] = float(np.mean(np.abs(cal_m.Error)))
    from visualization import redraw
    redraw({**globals(), **locals()}, files, figures)

    calendar_day_label = 'calendar days' if WEEK_LENGTH == 7 else 'operating days'
    TABLE_LABELS = {
        'model_leaderboard': f'Model leaderboard — Validation & Final Test ranking ({unit_label})',
        'weekly_forecast_summary': f'Weekly aggregated forecast roll-up ({WEEK_LENGTH}-day weeks, {unit_label})',
        'input_data': f'Input {data_lower} series ({unit_label})',
        'raw_input_data': f'Original untouched input {data_lower} series ({unit_label})',
        'data_preparation_log': 'Data preparation log (blank imputations & causal outlier treatments)',
        'descriptive_statistics': f'Descriptive statistics ({unit_label})',
        'stl_decomposition': f'STL decomposition (period={WEEK_LENGTH}) & modified z-scores ({unit_label})',
        'stationarity_tests': f'Stationarity tests — ADF & KPSS on level and Δ{data_lower}',
        'distribution_summary': f'Distribution histogram bins, mean, median & skewness ({unit_label})',
        'demand_acf_pacf': f'Demand ACF & PACF up to lag {DEMAND_ACF_MAX_LAG}',
        'sma_tuning': f'Moving average window tuning ({unit_label})',
        'model_tuning': f'Ridge & Random Forest hyperparameter tuning grid ({unit_label})',
        'feature_comparison': f'Old (baseline) vs. new (rich) features comparison ({unit_label})',
        'feature_importance': 'Feature importance — Ridge standardized coefs & RF validation permutation',
        'treatment_comparison': f'Raw vs. treated data accuracy comparison for selected model ({unit_label})',
        'sarima_aic_search': 'SARIMA training AIC candidate search & convergence log',
        'ensemble_weights': 'Top-3 validation ensemble members & weights',
        'validation_metrics': f'Validation model comparison — MAE, RMSE, WAPE, MASE, sMAPE ({unit_label})',
        'model_metrics': f'Final test model performance — MAE, RMSE, WAPE, MASE, sMAPE ({unit_label})',
        'test_predictions': f'Final test predictions & 80%/95% intervals ({unit_label})',
        'forecast': f'Future {options["horizon"]}-step forecasts & 80%/95% intervals ({unit_label})',
        'forecast_5_steps': f'Next 5 {calendar_day_label} forecast & intervals ({unit_label})',
        'interval_coverage': 'Test prediction interval coverage check (80% & 95%)',
        'cv_fold_metrics': f'Rolling-origin cross-validation per-fold metrics ({cv_k} folds, {unit_label})',
        'cv_summary': f'Rolling-origin cross-validation summary ({cv_k} folds, {unit_label})',
        'diebold_mariano': 'Diebold–Mariano HLN significance tests (vs. Naive & Winner)',
        'lead_time_errors': f'Development {lead}-day cumulative errors ({unit_label})',
        'weekday_gaps': 'Nonconsecutive weekday gaps',
        'inventory_metrics': f'EOQ, safety stock and reorder point ({unit_label})',
        'policy_simulation': f'Daily replenishment simulation ({unit_label})',
        'policy_events': f'Simulated order and receipt events ({unit_label})',
        'policy_summary': 'Replenishment simulation summary',
        'regression_coefficients': 'Weekday regression coefficients (HC3)',
        'multi_origin_metrics': f'Multi-origin {lead}-step block errors ({unit_label})',
        'service_sensitivity': f'Service level sensitivity (90% / 95% / 99%, {unit_label})',
    }

    tables = []
    TABLE_LABELS.update({'21_cv_heatmap':'Cross-validation error by model and fold','22_demand_calendar_heatmap':'Demand by operating week and weekday','23_actual_vs_forecast':'Actual versus forecast scatter data','24_error_direction_matrix':'Final-test error direction by weekday'})
    for key, frame in frames.items():
        content = frame.to_csv(index=False)
        files[f'tables/{key}.csv'] = content.encode()
        tables.append(dict(
            key=key,
            label=TABLE_LABELS.get(key, key.replace('_', ' ').capitalize()),
            csv=content,
            columns=list(frame.columns),
            rows=clean_json(frame.to_dict(orient='records')),
        ))
    if purchase_plan and purchase_plan.get('order_schedule_csv'):
        files['tables/purchase_order_schedule.csv'] = purchase_plan['order_schedule_csv'].encode()
        files['purchase_plan.json'] = json.dumps(purchase_plan, indent=2, allow_nan=False).encode()
    files['summary.json'] = json.dumps(summary, indent=2, allow_nan=False).encode()
    files['regression_diagnostics.json'] = json.dumps(clean_json(diag), indent=2, allow_nan=False).encode()
    files['README.txt'] = (
        f'Demand Forecast Lab — {data_label} ({unit_label})\n\n'
        + '\n'.join(notes)
        + '\n\nCharts use suitable statistical forms. Supplementary charts are descriptive and do not affect forecasts.\nDisplay: Light, Dark and Report themes are remembered locally; Guided mode filters existing views. Reading guides use four section buttons, with the finding selected first and Show all four available. Print report provides an A4-friendly Part 1 report. Clear data is available in setup and Data. Theme and display choices do not affect these results.\nThe raw_input_data.csv file preserves the original uploaded values, and input_data.csv contains the prepared series used for modeling.\n'
    ).encode()
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    result = dict(
        summary=summary,
        figures=figures,
        tables=tables,
        purchase_plan=purchase_plan,
        zip=base64.b64encode(bundle.getvalue()).decode(),
    )
    progress('Analysis complete')
    return result, files


def analyze_json(text, progress=None):
    return json.dumps(analyze(json.loads(text), progress)[0], allow_nan=False)
