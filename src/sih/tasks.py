import math
from pathlib import Path
from datetime import date, datetime, timedelta

import joblib
import pandas as pd

from celery import shared_task
from django.db import transaction

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

from sih.models import (
    HospitalAdmission,
    HospitalOccupancyPrediction,
    HospitalOccupancyPredictionAudit,
    HospitalOccupancyPredictionRun,
)


def _parse_date(value):
    if not value:
        return None
    if isinstance(value, (date, datetime)):
        return value.date() if isinstance(value, datetime) else value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _make_future_week_starts(start_d: date, horizon_days: int) -> list[date]:
    end_d = start_d + timedelta(days=horizon_days)
    cur = _week_start(start_d)
    last = _week_start(end_d)
    out = []

    while cur <= last:
        out.append(cur)
        cur += timedelta(days=7)

    return out


def _safe_mean(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def _safe_std(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= 1:
        return 0.0
    value = float(s.std())
    return 0.0 if math.isnan(value) else value


def _safe_quantile(series: pd.Series, q: float):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.quantile(q))


def _safe_min(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.min())


def _safe_max(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.max())


def _safe_last(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _safe_div_series(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    den = den.replace(0, pd.NA)
    return num.div(den)


def _safe_div_value(num, den):
    if num is None or den is None:
        return None
    try:
        num = float(num)
        den = float(den)
    except (TypeError, ValueError):
        return None
    if den == 0:
        return None
    return float(num / den)


def _fallback_stat(*values, default=None):
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        return float(value)
    return default


def _robust_recent_mean(series: pd.Series, preferred_windows=(4, 8, 12), fallback=None):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return fallback

    for window in preferred_windows:
        if len(s) >= window:
            return float(s.tail(window).mean())

    return float(s.mean())


def _robust_recent_std(series: pd.Series, preferred_windows=(4, 8, 12), fallback=0.0):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return fallback

    for window in preferred_windows:
        if len(s) >= window:
            tail = s.tail(window)
            if len(tail) <= 1:
                return 0.0
            value = float(tail.std())
            return 0.0 if math.isnan(value) else value

    if len(s) <= 1:
        return 0.0

    value = float(s.std())
    return 0.0 if math.isnan(value) else value


def _robust_lag(series: pd.Series, n: int, fallback=None):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) >= n:
        return float(s.iloc[-n])

    if fallback is not None:
        return fallback

    if s.empty:
        return None

    return float(s.iloc[0])


def _robust_roll_mean(series: pd.Series, window: int, fallback=None):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return fallback

    if len(s) >= window:
        return float(s.tail(window).mean())

    if fallback is not None:
        return fallback

    return float(s.mean())


def _robust_roll_std(series: pd.Series, window: int, fallback=0.0):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return fallback

    if len(s) >= window:
        tail = s.tail(window)
        if len(tail) <= 1:
            return 0.0
        value = float(tail.std())
        return 0.0 if math.isnan(value) else value

    if len(s) <= 1:
        return 0.0

    value = float(s.std())
    return 0.0 if math.isnan(value) else value


def _safe_trend_delta(short_value, long_value, fallback=0.0):
    if short_value is None or long_value is None:
        return fallback
    return float(short_value - long_value)


def _safe_trend_ratio(short_value, long_value, fallback=1.0):
    if short_value is None or long_value in (None, 0):
        return fallback
    return float(short_value / long_value)


def _seasonal_window_mask(week_series: pd.Series, target_week: int, radius: int = 1) -> pd.Series:
    valid_weeks = set()

    for offset in range(-radius, radius + 1):
        wk = target_week + offset

        if wk < 1:
            wk += 52
        elif wk > 52:
            wk -= 52

        valid_weeks.add(wk)

    return week_series.isin(valid_weeks)


def _build_seasonal_baseline(
    hospital: str,
    future_week_start: date,
    history_df: pd.DataFrame,
) -> dict:
    hist = history_df[
        history_df["health_facility_registry_code"] == hospital
    ].sort_values("week_start").copy()

    if hist.empty:
        return {
            "seasonal_adm_strict": 0.0,
            "seasonal_adm_window": 0.0,
            "seasonal_adm_last_year": 0.0,
            "seasonal_los_strict": 0.0,
            "seasonal_los_window": 0.0,
            "seasonal_los_last_year": 0.0,
        }

    future_iso = future_week_start.isocalendar()
    target_week = int(future_iso.week)
    target_year = int(future_iso.year)

    hist["week"] = pd.to_numeric(hist["week"], errors="coerce")
    hist["year"] = pd.to_numeric(hist["year"], errors="coerce")

    strict_same_week = hist[hist["week"] == target_week]
    window_same_period = hist[_seasonal_window_mask(hist["week"], target_week, radius=1)]
    last_year_same_week = hist[
        (hist["year"] == target_year - 1) & (hist["week"] == target_week)
    ]

    seasonal_adm_strict = _safe_mean(strict_same_week["admissions_count"])
    seasonal_adm_window = _safe_mean(window_same_period["admissions_count"])
    seasonal_adm_last_year = _safe_mean(last_year_same_week["admissions_count"])

    seasonal_los_strict = _safe_mean(strict_same_week["avg_length_of_stay_days"])
    seasonal_los_window = _safe_mean(window_same_period["avg_length_of_stay_days"])
    seasonal_los_last_year = _safe_mean(last_year_same_week["avg_length_of_stay_days"])

    return {
        "seasonal_adm_strict": _fallback_stat(
            seasonal_adm_strict,
            seasonal_adm_window,
            seasonal_adm_last_year,
            default=0.0,
        ),
        "seasonal_adm_window": _fallback_stat(
            seasonal_adm_window,
            seasonal_adm_strict,
            seasonal_adm_last_year,
            default=0.0,
        ),
        "seasonal_adm_last_year": _fallback_stat(
            seasonal_adm_last_year,
            seasonal_adm_strict,
            seasonal_adm_window,
            default=0.0,
        ),
        "seasonal_los_strict": _fallback_stat(
            seasonal_los_strict,
            seasonal_los_window,
            seasonal_los_last_year,
            default=0.0,
        ),
        "seasonal_los_window": _fallback_stat(
            seasonal_los_window,
            seasonal_los_strict,
            seasonal_los_last_year,
            default=0.0,
        ),
        "seasonal_los_last_year": _fallback_stat(
            seasonal_los_last_year,
            seasonal_los_strict,
            seasonal_los_window,
            default=0.0,
        ),
    }


def _build_future_feature_row(
    hospital: str,
    future_week_start: date,
    history_df: pd.DataFrame,
) -> dict:
    hist = history_df[
        history_df["health_facility_registry_code"] == hospital
    ].sort_values("week_start").copy()

    admissions_series = pd.to_numeric(hist["admissions_count"], errors="coerce").dropna()
    deaths_series = pd.to_numeric(hist["deaths_count"], errors="coerce").dropna()
    icu_series = pd.to_numeric(hist["icu_days_sum"], errors="coerce").dropna()
    paid_series = pd.to_numeric(hist["total_amount_paid_sum_brl"], errors="coerce").dropna()
    los_series = pd.to_numeric(hist["avg_length_of_stay_days"], errors="coerce").dropna()

    hist_adm_mean = _safe_mean(admissions_series)
    hist_los_mean = _safe_mean(los_series)
    hist_deaths_mean = _safe_mean(deaths_series)
    hist_icu_mean = _safe_mean(icu_series)
    hist_paid_mean = _safe_mean(paid_series)

    last_adm = _safe_last(admissions_series)
    last_los = _safe_last(los_series)

    admissions_lag_1 = _robust_lag(admissions_series, 1, fallback=hist_adm_mean)
    admissions_lag_2 = _robust_lag(admissions_series, 2, fallback=admissions_lag_1)
    admissions_lag_3 = _robust_lag(admissions_series, 3, fallback=admissions_lag_2)
    admissions_lag_4 = _robust_lag(admissions_series, 4, fallback=admissions_lag_3)
    admissions_lag_8 = _robust_lag(admissions_series, 8, fallback=hist_adm_mean)
    admissions_lag_12 = _robust_lag(admissions_series, 12, fallback=hist_adm_mean)
    admissions_lag_26 = _robust_lag(admissions_series, 26, fallback=hist_adm_mean)
    admissions_lag_52 = _robust_lag(admissions_series, 52, fallback=hist_adm_mean)

    deaths_lag_1 = _robust_lag(deaths_series, 1, fallback=hist_deaths_mean)
    deaths_lag_2 = _robust_lag(deaths_series, 2, fallback=deaths_lag_1)
    deaths_lag_4 = _robust_lag(deaths_series, 4, fallback=hist_deaths_mean)

    icu_lag_1 = _robust_lag(icu_series, 1, fallback=hist_icu_mean)
    icu_lag_2 = _robust_lag(icu_series, 2, fallback=icu_lag_1)
    icu_lag_4 = _robust_lag(icu_series, 4, fallback=hist_icu_mean)

    paid_lag_1 = _robust_lag(paid_series, 1, fallback=hist_paid_mean)
    paid_lag_2 = _robust_lag(paid_series, 2, fallback=paid_lag_1)

    los_lag_1 = _robust_lag(los_series, 1, fallback=hist_los_mean)
    los_lag_2 = _robust_lag(los_series, 2, fallback=los_lag_1)
    los_lag_3 = _robust_lag(los_series, 3, fallback=los_lag_2)
    los_lag_4 = _robust_lag(los_series, 4, fallback=los_lag_3)
    los_lag_8 = _robust_lag(los_series, 8, fallback=hist_los_mean)
    los_lag_12 = _robust_lag(los_series, 12, fallback=hist_los_mean)

    roll_mean_2 = _robust_roll_mean(admissions_series, 2, fallback=hist_adm_mean)
    roll_mean_4 = _robust_roll_mean(admissions_series, 4, fallback=hist_adm_mean)
    roll_mean_8 = _robust_roll_mean(admissions_series, 8, fallback=hist_adm_mean)
    roll_mean_12 = _robust_roll_mean(admissions_series, 12, fallback=hist_adm_mean)

    roll_std_2 = _robust_roll_std(admissions_series, 2, fallback=0.0)
    roll_std_4 = _robust_roll_std(admissions_series, 4, fallback=0.0)
    roll_std_8 = _robust_roll_std(admissions_series, 8, fallback=0.0)
    roll_std_12 = _robust_roll_std(admissions_series, 12, fallback=0.0)

    los_roll_mean_2 = _robust_roll_mean(los_series, 2, fallback=hist_los_mean)
    los_roll_mean_4 = _robust_roll_mean(los_series, 4, fallback=hist_los_mean)
    los_roll_mean_8 = _robust_roll_mean(los_series, 8, fallback=hist_los_mean)
    los_roll_mean_12 = _robust_roll_mean(los_series, 12, fallback=hist_los_mean)

    los_roll_std_2 = _robust_roll_std(los_series, 2, fallback=0.0)
    los_roll_std_4 = _robust_roll_std(los_series, 4, fallback=0.0)
    los_roll_std_8 = _robust_roll_std(los_series, 8, fallback=0.0)
    los_roll_std_12 = _robust_roll_std(los_series, 12, fallback=0.0)

    recent_mean_3 = _robust_roll_mean(admissions_series, 3, fallback=hist_adm_mean)
    recent_mean_4 = _robust_roll_mean(admissions_series, 4, fallback=hist_adm_mean)
    recent_mean_6 = _robust_roll_mean(admissions_series, 6, fallback=hist_adm_mean)
    recent_mean_8 = _robust_roll_mean(admissions_series, 8, fallback=hist_adm_mean)
    recent_mean_12 = _robust_roll_mean(admissions_series, 12, fallback=hist_adm_mean)

    recent_los_mean_3 = _robust_roll_mean(los_series, 3, fallback=hist_los_mean)
    recent_los_mean_4 = _robust_roll_mean(los_series, 4, fallback=hist_los_mean)
    recent_los_mean_8 = _robust_roll_mean(los_series, 8, fallback=hist_los_mean)
    recent_los_mean_12 = _robust_roll_mean(los_series, 12, fallback=hist_los_mean)

    recent_max_4 = _fallback_stat(
        _safe_max(admissions_series.tail(4)),
        recent_mean_4,
        hist_adm_mean,
        default=0.0,
    )
    recent_max_8 = _fallback_stat(
        _safe_max(admissions_series.tail(8)),
        recent_mean_8,
        hist_adm_mean,
        default=0.0,
    )

    iso = future_week_start.isocalendar()
    week_num = int(iso.week)
    same_week_hist = hist[hist["week"] == week_num]

    seasonal_info = _build_seasonal_baseline(
        hospital=hospital,
        future_week_start=future_week_start,
        history_df=history_df,
    )

    seasonal_week_mean_admissions = _fallback_stat(
        seasonal_info["seasonal_adm_last_year"],
        seasonal_info["seasonal_adm_strict"],
        seasonal_info["seasonal_adm_window"],
        recent_mean_8,
        hist_adm_mean,
        last_adm,
        default=0.0,
    )
    seasonal_week_median_admissions = _fallback_stat(
        _safe_quantile(same_week_hist["admissions_count"], 0.50),
        seasonal_info["seasonal_adm_last_year"],
        seasonal_info["seasonal_adm_window"],
        seasonal_week_mean_admissions,
        default=0.0,
    )

    seasonal_week_mean_los = _fallback_stat(
        seasonal_info["seasonal_los_last_year"],
        seasonal_info["seasonal_los_strict"],
        seasonal_info["seasonal_los_window"],
        recent_los_mean_8,
        hist_los_mean,
        last_los,
        default=0.0,
    )
    seasonal_week_median_los = _fallback_stat(
        _safe_quantile(same_week_hist["avg_length_of_stay_days"], 0.50),
        seasonal_info["seasonal_los_last_year"],
        seasonal_info["seasonal_los_window"],
        seasonal_week_mean_los,
        default=0.0,
    )

    hospital_mean_admissions = _fallback_stat(hist_adm_mean, recent_mean_8, last_adm, default=0.0)
    hospital_median_admissions = _fallback_stat(
        _safe_quantile(hist["admissions_count"], 0.50),
        hospital_mean_admissions,
        default=0.0,
    )
    hospital_min_admissions = _fallback_stat(_safe_min(admissions_series), hospital_mean_admissions, default=0.0)
    hospital_max_admissions = _fallback_stat(_safe_max(admissions_series), hospital_mean_admissions, default=0.0)
    hospital_q05_admissions = _fallback_stat(
        _safe_quantile(hist["admissions_count"], 0.05),
        hospital_min_admissions,
        default=0.0,
    )
    hospital_q10_admissions = _fallback_stat(
        _safe_quantile(hist["admissions_count"], 0.10),
        hospital_mean_admissions,
        default=0.0,
    )
    hospital_q90_admissions = _fallback_stat(
        _safe_quantile(hist["admissions_count"], 0.90),
        hospital_mean_admissions,
        default=0.0,
    )
    hospital_q95_admissions = _fallback_stat(
        _safe_quantile(hist["admissions_count"], 0.95),
        hospital_q90_admissions,
        hospital_mean_admissions,
        default=0.0,
    )

    hospital_mean_los = _fallback_stat(hist_los_mean, recent_los_mean_8, last_los, default=0.0)
    hospital_median_los = _fallback_stat(
        _safe_quantile(hist["avg_length_of_stay_days"], 0.50),
        hospital_mean_los,
        default=0.0,
    )
    hospital_min_los = _fallback_stat(_safe_min(los_series), hospital_mean_los, default=0.0)
    hospital_max_los = _fallback_stat(_safe_max(los_series), hospital_mean_los, default=0.0)

    return {
        "health_facility_registry_code": hospital,
        "week": week_num,
        "month": future_week_start.month,
        "quarter": ((future_week_start.month - 1) // 3) + 1,
        "week_sin": math.sin(2 * math.pi * week_num / 52),
        "week_cos": math.cos(2 * math.pi * week_num / 52),
        "admissions_lag_1": admissions_lag_1,
        "admissions_lag_2": admissions_lag_2,
        "admissions_lag_3": admissions_lag_3,
        "admissions_lag_4": admissions_lag_4,
        "admissions_lag_8": admissions_lag_8,
        "admissions_lag_12": admissions_lag_12,
        "admissions_lag_26": admissions_lag_26,
        "admissions_lag_52": admissions_lag_52,
        "deaths_lag_1": deaths_lag_1,
        "deaths_lag_2": deaths_lag_2,
        "deaths_lag_4": deaths_lag_4,
        "icu_lag_1": icu_lag_1,
        "icu_lag_2": icu_lag_2,
        "icu_lag_4": icu_lag_4,
        "paid_lag_1": paid_lag_1,
        "paid_lag_2": paid_lag_2,
        "los_lag_1": los_lag_1,
        "los_lag_2": los_lag_2,
        "los_lag_3": los_lag_3,
        "los_lag_4": los_lag_4,
        "los_lag_8": los_lag_8,
        "los_lag_12": los_lag_12,
        "roll_mean_2": roll_mean_2,
        "roll_mean_4": roll_mean_4,
        "roll_mean_8": roll_mean_8,
        "roll_mean_12": roll_mean_12,
        "roll_std_2": roll_std_2,
        "roll_std_4": roll_std_4,
        "roll_std_8": roll_std_8,
        "roll_std_12": roll_std_12,
        "los_roll_mean_2": los_roll_mean_2,
        "los_roll_mean_4": los_roll_mean_4,
        "los_roll_mean_8": los_roll_mean_8,
        "los_roll_mean_12": los_roll_mean_12,
        "los_roll_std_2": los_roll_std_2,
        "los_roll_std_4": los_roll_std_4,
        "los_roll_std_8": los_roll_std_8,
        "los_roll_std_12": los_roll_std_12,
        "recent_mean_3": recent_mean_3,
        "recent_mean_4": recent_mean_4,
        "recent_mean_6": recent_mean_6,
        "recent_mean_8": recent_mean_8,
        "recent_mean_12": recent_mean_12,
        "recent_los_mean_3": recent_los_mean_3,
        "recent_los_mean_4": recent_los_mean_4,
        "recent_los_mean_8": recent_los_mean_8,
        "recent_los_mean_12": recent_los_mean_12,
        "recent_max_4": recent_max_4,
        "recent_max_8": recent_max_8,
        "trend_adm_3_8": _safe_trend_delta(recent_mean_3, roll_mean_8, fallback=0.0),
        "trend_adm_4_8": _safe_trend_delta(recent_mean_4, roll_mean_8, fallback=0.0),
        "trend_adm_3_12": _safe_trend_delta(recent_mean_3, roll_mean_12, fallback=0.0),
        "trend_adm_6_12": _safe_trend_delta(recent_mean_6, roll_mean_12, fallback=0.0),
        "trend_los_3_8": _safe_trend_delta(recent_los_mean_3, los_roll_mean_8, fallback=0.0),
        "trend_los_4_8": _safe_trend_delta(recent_los_mean_4, los_roll_mean_8, fallback=0.0),
        "trend_los_3_12": _safe_trend_delta(recent_los_mean_3, los_roll_mean_12, fallback=0.0),
        "trend_ratio_adm_3_8": _safe_trend_ratio(recent_mean_3, roll_mean_8, fallback=1.0),
        "trend_ratio_adm_4_8": _safe_trend_ratio(recent_mean_4, roll_mean_8, fallback=1.0),
        "trend_ratio_adm_3_12": _safe_trend_ratio(recent_mean_3, roll_mean_12, fallback=1.0),
        "trend_ratio_los_3_8": _safe_trend_ratio(recent_los_mean_3, los_roll_mean_8, fallback=1.0),
        "seasonal_week_mean_admissions": seasonal_week_mean_admissions,
        "seasonal_week_median_admissions": seasonal_week_median_admissions,
        "seasonal_week_mean_los": seasonal_week_mean_los,
        "seasonal_week_median_los": seasonal_week_median_los,
        "hospital_mean_admissions": hospital_mean_admissions,
        "hospital_median_admissions": hospital_median_admissions,
        "hospital_min_admissions": hospital_min_admissions,
        "hospital_max_admissions": hospital_max_admissions,
        "hospital_q05_admissions": hospital_q05_admissions,
        "hospital_q10_admissions": hospital_q10_admissions,
        "hospital_q90_admissions": hospital_q90_admissions,
        "hospital_q95_admissions": hospital_q95_admissions,
        "hospital_mean_los": hospital_mean_los,
        "hospital_median_los": hospital_median_los,
        "hospital_min_los": hospital_min_los,
        "hospital_max_los": hospital_max_los,
        "admissions_ratio_1": _safe_div_value(admissions_lag_1, admissions_lag_2) or 1.0,
        "admissions_ratio_4": _safe_div_value(admissions_lag_1, roll_mean_4) or 1.0,
        "los_ratio_1": _safe_div_value(los_lag_1, los_lag_2) or 1.0,
    }


def _build_weekly_dataset(date_init=None, date_end=None):
    qs = (
        HospitalAdmission.objects
        .exclude(admission_date__isnull=True)
        .exclude(health_facility_registry_code__isnull=True)
        .exclude(health_facility_registry_code__exact="")
    )

    if date_init:
        qs = qs.filter(admission_date__gte=date_init)

    if date_end:
        qs = qs.filter(admission_date__lte=date_end)

    data = list(
        qs.values(
            "record_identifier",
            "health_facility_registry_code",
            "admission_date",
            "discharge_date",
            "death_during_admission",
            "intensive_care_total_days",
            "total_amount_paid_brl",
        )
    )

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    df["health_facility_registry_code"] = (
        df["health_facility_registry_code"].astype(str).str.strip().str.upper()
    )

    df = df[df["health_facility_registry_code"].astype(str).str.strip() != ""].copy()

    df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce")
    df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")

    df = df[df["admission_date"].notna()].copy()

    df["week_start"] = df["admission_date"] - pd.to_timedelta(
        df["admission_date"].dt.weekday,
        unit="D",
    )
    df["week_start"] = df["week_start"].dt.normalize()

    iso = df["week_start"].dt.isocalendar()
    df["year"] = iso.year.astype(int)
    df["week"] = iso.week.astype(int)
    df["month"] = df["week_start"].dt.month.astype(int)
    df["quarter"] = df["week_start"].dt.quarter.astype(int)

    df["deaths_count_flag"] = df["death_during_admission"].fillna(False).astype(int)
    df["icu_days_sum_value"] = pd.to_numeric(
        df["intensive_care_total_days"],
        errors="coerce",
    ).fillna(0)
    df["total_amount_paid_sum_brl_value"] = pd.to_numeric(
        df["total_amount_paid_brl"],
        errors="coerce",
    ).fillna(0)

    valid_los_mask = (
        df["discharge_date"].notna()
        & (df["discharge_date"] >= df["admission_date"])
    )

    df["length_of_stay_days"] = pd.NA
    df.loc[valid_los_mask, "length_of_stay_days"] = (
        df.loc[valid_los_mask, "discharge_date"] - df.loc[valid_los_mask, "admission_date"]
    ).dt.days.astype(float)

    grouped = (
        df.groupby(
            [
                "health_facility_registry_code",
                "week_start",
                "year",
                "week",
                "month",
                "quarter",
            ],
            dropna=False,
        )
        .agg(
            admissions_count=("record_identifier", "count"),
            deaths_count=("deaths_count_flag", "sum"),
            icu_days_sum=("icu_days_sum_value", "sum"),
            total_amount_paid_sum_brl=("total_amount_paid_sum_brl_value", "sum"),
            avg_length_of_stay_days=("length_of_stay_days", "mean"),
        )
        .reset_index()
        .sort_values(["health_facility_registry_code", "week_start"])
    )

    grouped["week_start"] = pd.to_datetime(grouped["week_start"]).dt.date
    return grouped


def _add_hospital_stats(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df.groupby("health_facility_registry_code")
        .agg(
            hospital_mean_admissions=("admissions_count", "mean"),
            hospital_median_admissions=("admissions_count", "median"),
            hospital_min_admissions=("admissions_count", "min"),
            hospital_max_admissions=("admissions_count", "max"),
            hospital_q05_admissions=("admissions_count", lambda s: float(pd.Series(s).quantile(0.05))),
            hospital_q10_admissions=("admissions_count", lambda s: float(pd.Series(s).quantile(0.10))),
            hospital_q90_admissions=("admissions_count", lambda s: float(pd.Series(s).quantile(0.90))),
            hospital_q95_admissions=("admissions_count", lambda s: float(pd.Series(s).quantile(0.95))),
            hospital_mean_los=("avg_length_of_stay_days", "mean"),
            hospital_median_los=("avg_length_of_stay_days", "median"),
            hospital_min_los=("avg_length_of_stay_days", "min"),
            hospital_max_los=("avg_length_of_stay_days", "max"),
        )
        .reset_index()
    )

    seasonal = (
        df.groupby(["health_facility_registry_code", "week"])
        .agg(
            seasonal_week_mean_admissions=("admissions_count", "mean"),
            seasonal_week_median_admissions=("admissions_count", "median"),
            seasonal_week_mean_los=("avg_length_of_stay_days", "mean"),
            seasonal_week_median_los=("avg_length_of_stay_days", "median"),
        )
        .reset_index()
    )

    out = df.merge(base, on="health_facility_registry_code", how="left")
    out = out.merge(seasonal, on=["health_facility_registry_code", "week"], how="left")
    return out


def _add_features(df: pd.DataFrame, min_weeks: int) -> pd.DataFrame:
    group = "health_facility_registry_code"

    for lag in [1, 2, 3, 4, 8, 12, 26, 52]:
        df[f"admissions_lag_{lag}"] = df.groupby(group)["admissions_count"].shift(lag)
        df[f"deaths_lag_{lag}"] = df.groupby(group)["deaths_count"].shift(lag)
        df[f"icu_lag_{lag}"] = df.groupby(group)["icu_days_sum"].shift(lag)
        df[f"paid_lag_{lag}"] = df.groupby(group)["total_amount_paid_sum_brl"].shift(lag)
        df[f"los_lag_{lag}"] = df.groupby(group)["avg_length_of_stay_days"].shift(lag)

    for window in [2, 4, 8, 12]:
        admissions_base = df.groupby(group)["admissions_count"].shift(1)
        los_base = df.groupby(group)["avg_length_of_stay_days"].shift(1)

        df[f"roll_mean_{window}"] = (
            admissions_base.groupby(df[group]).rolling(window).mean().reset_index(level=0, drop=True)
        )
        df[f"roll_std_{window}"] = (
            admissions_base.groupby(df[group]).rolling(window).std().reset_index(level=0, drop=True)
        )

        df[f"los_roll_mean_{window}"] = (
            los_base.groupby(df[group]).rolling(window).mean().reset_index(level=0, drop=True)
        )
        df[f"los_roll_std_{window}"] = (
            los_base.groupby(df[group]).rolling(window).std().reset_index(level=0, drop=True)
        )

    df["recent_mean_3"] = (
        df.groupby(group)["admissions_count"]
        .shift(1)
        .groupby(df[group])
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["recent_mean_4"] = (
        df.groupby(group)["admissions_count"]
        .shift(1)
        .groupby(df[group])
        .rolling(4)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["recent_mean_6"] = (
        df.groupby(group)["admissions_count"]
        .shift(1)
        .groupby(df[group])
        .rolling(6)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["recent_mean_8"] = (
        df.groupby(group)["admissions_count"]
        .shift(1)
        .groupby(df[group])
        .rolling(8)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["recent_mean_12"] = (
        df.groupby(group)["admissions_count"]
        .shift(1)
        .groupby(df[group])
        .rolling(12)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["recent_los_mean_3"] = (
        df.groupby(group)["avg_length_of_stay_days"]
        .shift(1)
        .groupby(df[group])
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["recent_los_mean_4"] = (
        df.groupby(group)["avg_length_of_stay_days"]
        .shift(1)
        .groupby(df[group])
        .rolling(4)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["recent_los_mean_8"] = (
        df.groupby(group)["avg_length_of_stay_days"]
        .shift(1)
        .groupby(df[group])
        .rolling(8)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["recent_los_mean_12"] = (
        df.groupby(group)["avg_length_of_stay_days"]
        .shift(1)
        .groupby(df[group])
        .rolling(12)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["recent_max_4"] = (
        df.groupby(group)["admissions_count"]
        .shift(1)
        .groupby(df[group])
        .rolling(4)
        .max()
        .reset_index(level=0, drop=True)
    )

    df["recent_max_8"] = (
        df.groupby(group)["admissions_count"]
        .shift(1)
        .groupby(df[group])
        .rolling(8)
        .max()
        .reset_index(level=0, drop=True)
    )

    df["admissions_ratio_1"] = _safe_div_series(df["admissions_count"], df["admissions_lag_1"])
    df["admissions_ratio_4"] = _safe_div_series(df["admissions_count"], df["roll_mean_4"])
    df["los_ratio_1"] = _safe_div_series(df["avg_length_of_stay_days"], df["los_lag_1"])

    df["trend_adm_3_8"] = df["recent_mean_3"] - df["roll_mean_8"]
    df["trend_adm_4_8"] = df["recent_mean_4"] - df["roll_mean_8"]
    df["trend_adm_3_12"] = df["recent_mean_3"] - df["roll_mean_12"]
    df["trend_adm_6_12"] = df["recent_mean_6"] - df["roll_mean_12"]

    df["trend_los_3_8"] = df["recent_los_mean_3"] - df["los_roll_mean_8"]
    df["trend_los_4_8"] = df["recent_los_mean_4"] - df["los_roll_mean_8"]
    df["trend_los_3_12"] = df["recent_los_mean_3"] - df["los_roll_mean_12"]

    df["trend_ratio_adm_3_8"] = _safe_div_series(df["recent_mean_3"], df["roll_mean_8"])
    df["trend_ratio_adm_4_8"] = _safe_div_series(df["recent_mean_4"], df["roll_mean_8"])
    df["trend_ratio_adm_3_12"] = _safe_div_series(df["recent_mean_3"], df["roll_mean_12"])
    df["trend_ratio_los_3_8"] = _safe_div_series(df["recent_los_mean_3"], df["los_roll_mean_8"])

    df["week_sin"] = df["week"].apply(lambda x: math.sin(2 * math.pi * x / 52))
    df["week_cos"] = df["week"].apply(lambda x: math.cos(2 * math.pi * x / 52))

    df = _add_hospital_stats(df)

    counts = df.groupby(group)["week_start"].nunique()
    keep = counts[counts >= min_weeks].index
    df = df[df[group].isin(keep)].copy()

    return df


def _training_feature_cols():
    return [
        "week",
        "month",
        "quarter",
        "week_sin",
        "week_cos",
        "admissions_lag_1",
        "admissions_lag_2",
        "admissions_lag_3",
        "admissions_lag_4",
        "admissions_lag_8",
        "admissions_lag_12",
        "admissions_lag_26",
        "admissions_lag_52",
        "deaths_lag_1",
        "deaths_lag_2",
        "deaths_lag_4",
        "icu_lag_1",
        "icu_lag_2",
        "icu_lag_4",
        "paid_lag_1",
        "paid_lag_2",
        "los_lag_1",
        "los_lag_2",
        "los_lag_3",
        "los_lag_4",
        "los_lag_8",
        "los_lag_12",
        "roll_mean_2",
        "roll_mean_4",
        "roll_mean_8",
        "roll_mean_12",
        "roll_std_2",
        "roll_std_4",
        "roll_std_8",
        "roll_std_12",
        "los_roll_mean_2",
        "los_roll_mean_4",
        "los_roll_mean_8",
        "los_roll_mean_12",
        "los_roll_std_2",
        "los_roll_std_4",
        "los_roll_std_8",
        "los_roll_std_12",
        "recent_mean_3",
        "recent_mean_4",
        "recent_mean_6",
        "recent_mean_8",
        "recent_mean_12",
        "recent_los_mean_3",
        "recent_los_mean_4",
        "recent_los_mean_8",
        "recent_los_mean_12",
        "recent_max_4",
        "recent_max_8",
        "trend_adm_3_8",
        "trend_adm_4_8",
        "trend_adm_3_12",
        "trend_adm_6_12",
        "trend_los_3_8",
        "trend_los_4_8",
        "trend_los_3_12",
        "trend_ratio_adm_3_8",
        "trend_ratio_adm_4_8",
        "trend_ratio_adm_3_12",
        "trend_ratio_los_3_8",
        "seasonal_week_mean_admissions",
        "seasonal_week_median_admissions",
        "seasonal_week_mean_los",
        "seasonal_week_median_los",
        "hospital_mean_admissions",
        "hospital_median_admissions",
        "hospital_min_admissions",
        "hospital_max_admissions",
        "hospital_q05_admissions",
        "hospital_q10_admissions",
        "hospital_q90_admissions",
        "hospital_q95_admissions",
        "hospital_mean_los",
        "hospital_median_los",
        "hospital_min_los",
        "hospital_max_los",
        "admissions_ratio_1",
        "admissions_ratio_4",
        "los_ratio_1",
    ]


def _prepare_model_frame(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def _build_hgbr(
    loss: str,
    quantile=None,
    max_iter: int = 300,
    learning_rate: float = 0.03,
    max_leaf_nodes: int = 21,
    min_samples_leaf: int = 4,
    l2_regularization: float = 0.1,
):
    kwargs = {
        "loss": loss,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
        "l2_regularization": l2_regularization,
        "random_state": 42,
        "early_stopping": False,
    }

    if loss == "quantile":
        kwargs["quantile"] = quantile

    return HistGradientBoostingRegressor(**kwargs)


def _classify_hospital_size(hospital_df: pd.DataFrame) -> str:
    mean_adm = float(pd.to_numeric(hospital_df["admissions_count"], errors="coerce").mean())

    if mean_adm >= 250:
        return "very_large"
    if mean_adm >= 120:
        return "large"
    if mean_adm >= 30:
        return "medium"
    return "small"


def _fit_large_hospital_models(hospital_df: pd.DataFrame, feature_cols: list[str], *, very_large: bool):
    X = _prepare_model_frame(hospital_df, feature_cols)
    y = hospital_df["admissions_count"].astype(float)

    main_model = _build_hgbr(
        loss="poisson",
        max_iter=460 if very_large else 420,
        learning_rate=0.03,
        max_leaf_nodes=31 if very_large else 25,
        min_samples_leaf=6,
        l2_regularization=0.12 if very_large else 0.15,
    )
    main_model.fit(X, y)

    high_model = _build_hgbr(
        loss="quantile",
        quantile=0.85,
        max_iter=380 if very_large else 340,
        learning_rate=0.03,
        max_leaf_nodes=31 if very_large else 25,
        min_samples_leaf=6,
        l2_regularization=0.08 if very_large else 0.10,
    )
    high_model.fit(X, y)

    return {
        "type": "very_large" if very_large else "large",
        "main_model": main_model,
        "high_model": high_model,
    }


def _fit_medium_hospital_models(hospital_df: pd.DataFrame, feature_cols: list[str]):
    X = _prepare_model_frame(hospital_df, feature_cols)
    y = hospital_df["admissions_count"].astype(float)

    main_model = _build_hgbr(
        loss="poisson",
        max_iter=280,
        learning_rate=0.035,
        max_leaf_nodes=15,
        min_samples_leaf=5,
        l2_regularization=0.22,
    )
    main_model.fit(X, y)

    return {
        "type": "medium",
        "main_model": main_model,
    }


def _fit_hospital_models(df: pd.DataFrame):
    feature_cols = _training_feature_cols()
    hospitals = sorted(df["health_facility_registry_code"].unique().tolist())

    models = {}

    for hospital in hospitals:
        hospital_df = df[df["health_facility_registry_code"] == hospital].copy()

        if len(hospital_df) < 12:
            continue

        hospital_size = _classify_hospital_size(hospital_df)

        if hospital_size == "very_large":
            models[hospital] = _fit_large_hospital_models(hospital_df, feature_cols, very_large=True)
        elif hospital_size == "large":
            models[hospital] = _fit_large_hospital_models(hospital_df, feature_cols, very_large=False)
        elif hospital_size == "medium":
            models[hospital] = _fit_medium_hospital_models(hospital_df, feature_cols)
        else:
            models[hospital] = {"type": "small"}

    return {
        "feature_cols": feature_cols,
        "hospital_models": models,
    }


def _peak_signal(feature_row: dict):
    last_1 = feature_row.get("admissions_lag_1")
    recent_8 = feature_row.get("recent_mean_8")
    recent_max_4 = feature_row.get("recent_max_4")
    seasonal = feature_row.get("seasonal_week_mean_admissions")
    hospital_mean = feature_row.get("hospital_mean_admissions")

    signal = 0.0
    signal += 0.40 * max((_safe_div_value(last_1, recent_8) or 1.0) - 1.0, 0.0)
    signal += 0.35 * max((_safe_div_value(recent_max_4, recent_8) or 1.0) - 1.0, 0.0)
    signal += 0.25 * max((_safe_div_value(seasonal, hospital_mean) or 1.0) - 1.0, 0.0)
    return float(min(max(signal, 0.0), 1.0))


def _predict_small_hospital(feature_row: dict, hospital_hist: pd.DataFrame):
    admissions = pd.to_numeric(hospital_hist["admissions_count"], errors="coerce").dropna()

    if admissions.empty:
        return 0.0

    recent_4 = _robust_recent_mean(admissions, preferred_windows=(4, 3, 2), fallback=0.0) or 0.0
    recent_8 = _robust_recent_mean(admissions, preferred_windows=(8, 6, 4), fallback=recent_4) or recent_4
    seasonal = feature_row.get("seasonal_week_mean_admissions") or recent_8
    seasonal_median = feature_row.get("seasonal_week_median_admissions") or seasonal
    recent_std = _robust_recent_std(admissions, preferred_windows=(8, 6, 4), fallback=0.0)
    q10 = feature_row.get("hospital_q10_admissions")
    q90 = feature_row.get("hospital_q90_admissions")
    recent_max_4 = feature_row.get("recent_max_4") or recent_4
    trend_short = feature_row.get("trend_adm_4_8") or 0.0

    pred = (
        0.25 * recent_4
        + 0.15 * recent_8
        + 0.45 * seasonal
        + 0.15 * seasonal_median
    )

    if trend_short > 0:
        pred += 0.12 * trend_short

    pred += 0.05 * recent_std

    lower = max(0.0, (q10 * 0.90) if q10 is not None else 0.0)
    upper_candidates = [
        recent_max_4 * 1.25,
        recent_8 * 1.35,
        seasonal * 1.25,
    ]
    if q90 is not None:
        upper_candidates.append(q90 * 1.20)

    upper = max(upper_candidates)
    pred = min(max(pred, lower), upper)

    return max(float(pred), 0.0)


def _predict_medium_hospital(
    feature_row: dict,
    model_info: dict,
    feature_cols: list[str],
):
    X = _prepare_model_frame(pd.DataFrame([feature_row]), feature_cols)
    pred_main = float(model_info["main_model"].predict(X)[0])

    last_1 = feature_row.get("admissions_lag_1") or pred_main
    recent_4 = feature_row.get("recent_mean_4") or feature_row.get("recent_mean_6") or pred_main
    recent_8 = feature_row.get("recent_mean_8") or recent_4
    seasonal = feature_row.get("seasonal_week_mean_admissions") or recent_8
    seasonal_median = feature_row.get("seasonal_week_median_admissions") or seasonal
    hospital_mean = feature_row.get("hospital_mean_admissions") or recent_8
    recent_max_4 = feature_row.get("recent_max_4") or recent_4
    recent_max_8 = feature_row.get("recent_max_8") or recent_8
    q90 = feature_row.get("hospital_q90_admissions")
    hist_max = feature_row.get("hospital_max_admissions") or pred_main

    pred = (
        0.35 * pred_main
        + 0.15 * recent_4
        + 0.10 * recent_8
        + 0.30 * seasonal
        + 0.10 * seasonal_median
    )

    floor = max(
        last_1 * 0.82,
        recent_4 * 0.84,
        recent_8 * 0.80,
        hospital_mean * 0.72,
        seasonal * 0.75,
    )

    ceiling = max(
        (q90 * 1.18) if q90 is not None else 0.0,
        recent_max_4 * 1.15,
        recent_max_8 * 1.12,
        hist_max * 1.08,
        seasonal * 1.22,
    )

    pred = min(max(pred, floor), ceiling)
    return max(float(pred), 0.0)


def _predict_large_hospital(
    feature_row: dict,
    model_info: dict,
    feature_cols: list[str],
):
    X = _prepare_model_frame(pd.DataFrame([feature_row]), feature_cols)
    pred_main = float(model_info["main_model"].predict(X)[0])
    pred_high = float(model_info["high_model"].predict(X)[0])

    last_1 = feature_row.get("admissions_lag_1") or pred_main
    recent_4 = feature_row.get("recent_mean_4") or feature_row.get("recent_mean_6") or pred_main
    recent_8 = feature_row.get("recent_mean_8") or recent_4
    seasonal = feature_row.get("seasonal_week_mean_admissions") or recent_8
    seasonal_median = feature_row.get("seasonal_week_median_admissions") or seasonal
    hospital_mean = feature_row.get("hospital_mean_admissions") or recent_8
    recent_max_4 = feature_row.get("recent_max_4") or recent_4
    recent_max_8 = feature_row.get("recent_max_8") or recent_8
    q95 = feature_row.get("hospital_q95_admissions")
    hist_max = feature_row.get("hospital_max_admissions") or pred_main
    peak_signal = _peak_signal(feature_row)

    pred = (
        0.38 * pred_main
        + 0.12 * recent_4
        + 0.08 * recent_8
        + 0.30 * seasonal
        + 0.12 * seasonal_median
    )

    pred += peak_signal * 0.28 * max(pred_high - pred, 0.0)

    floor = max(
        last_1 * 0.80,
        recent_4 * 0.80,
        recent_8 * 0.78,
        hospital_mean * 0.70,
        seasonal * 0.75,
    )

    ceiling = max(
        (q95 * 1.18) if q95 is not None else 0.0,
        recent_max_4 * 1.16,
        recent_max_8 * 1.14,
        hist_max * 1.08,
        seasonal * 1.22,
    )

    pred = min(max(pred, floor), ceiling)
    return max(float(pred), 0.0)


def _predict_very_large_hospital(
    feature_row: dict,
    model_info: dict,
    feature_cols: list[str],
):
    X = _prepare_model_frame(pd.DataFrame([feature_row]), feature_cols)
    pred_main = float(model_info["main_model"].predict(X)[0])
    pred_high = float(model_info["high_model"].predict(X)[0])

    last_1 = feature_row.get("admissions_lag_1") or pred_main
    recent_4 = feature_row.get("recent_mean_4") or feature_row.get("recent_mean_6") or pred_main
    recent_8 = feature_row.get("recent_mean_8") or recent_4
    seasonal = feature_row.get("seasonal_week_mean_admissions") or recent_8
    seasonal_median = feature_row.get("seasonal_week_median_admissions") or seasonal
    hospital_mean = feature_row.get("hospital_mean_admissions") or recent_8
    recent_max_4 = feature_row.get("recent_max_4") or recent_4
    recent_max_8 = feature_row.get("recent_max_8") or recent_8
    q95 = feature_row.get("hospital_q95_admissions")
    hist_max = feature_row.get("hospital_max_admissions") or pred_main
    peak_signal = _peak_signal(feature_row)

    pred = (
        0.34 * pred_main
        + 0.14 * recent_4
        + 0.08 * recent_8
        + 0.32 * seasonal
        + 0.12 * seasonal_median
    )

    pred += peak_signal * 0.32 * max(pred_high - pred, 0.0)

    floor = max(
        last_1 * 0.82,
        recent_4 * 0.82,
        recent_8 * 0.80,
        hospital_mean * 0.72,
        seasonal * 0.76,
    )

    ceiling = max(
        (q95 * 1.20) if q95 is not None else 0.0,
        recent_max_4 * 1.18,
        recent_max_8 * 1.15,
        hist_max * 1.10,
        seasonal * 1.24,
    )

    pred = min(max(pred, floor), ceiling)
    return max(float(pred), 0.0)


def _predict_admissions(
    hospital: str,
    feature_row: dict,
    hospital_hist: pd.DataFrame,
    fitted_models: dict,
):
    feature_cols = fitted_models["feature_cols"]
    hospital_models = fitted_models["hospital_models"]

    model_info = hospital_models.get(hospital)

    if model_info is None:
        return _predict_small_hospital(feature_row, hospital_hist)

    model_type = model_info["type"]

    if model_type == "small":
        return _predict_small_hospital(feature_row, hospital_hist)
    if model_type == "medium":
        return _predict_medium_hospital(feature_row, model_info, feature_cols)
    if model_type == "large":
        return _predict_large_hospital(feature_row, model_info, feature_cols)
    return _predict_very_large_hospital(feature_row, model_info, feature_cols)


def _predict_los_simple(feature_row: dict, hospital_hist: pd.DataFrame):
    los = pd.to_numeric(hospital_hist["avg_length_of_stay_days"], errors="coerce").dropna()

    if los.empty:
        return 0.0

    recent_4 = _robust_recent_mean(los, preferred_windows=(4, 3, 2), fallback=0.0) or 0.0
    recent_8 = _robust_recent_mean(los, preferred_windows=(8, 6, 4), fallback=recent_4) or recent_4
    seasonal = feature_row.get("seasonal_week_mean_los") or recent_8
    seasonal_median = feature_row.get("seasonal_week_median_los") or seasonal
    hospital_mean = feature_row.get("hospital_mean_los") or recent_8
    last_value = feature_row.get("los_lag_1") or recent_4

    pred = (
        0.30 * recent_4
        + 0.15 * recent_8
        + 0.30 * seasonal
        + 0.15 * seasonal_median
        + 0.10 * hospital_mean
    )

    pred = 0.88 * pred + 0.12 * last_value

    hist_min = feature_row.get("hospital_min_los") or pred
    hist_max = feature_row.get("hospital_max_los") or pred

    lower = max(0.0, hist_min * 0.80, hospital_mean * 0.70, seasonal * 0.75)
    upper = max(hist_max * 1.18, recent_8 * 1.18, hospital_mean * 1.25, seasonal * 1.20)

    pred = min(max(pred, lower), upper)
    return max(float(pred), 0.0)


def _append_simulated_week(
    history_df: pd.DataFrame,
    hospital: str,
    week_start_value: date,
    predicted_total: float,
    predicted_los: float,
    feature_row: dict,
) -> pd.DataFrame:
    iso = week_start_value.isocalendar()

    new_row = {
        "health_facility_registry_code": hospital,
        "week_start": week_start_value,
        "year": int(iso.year),
        "week": int(iso.week),
        "month": int(week_start_value.month),
        "quarter": int(((week_start_value.month - 1) // 3) + 1),
        "admissions_count": float(predicted_total),
        "deaths_count": float(feature_row.get("deaths_lag_1") or 0.0),
        "icu_days_sum": float(feature_row.get("icu_lag_1") or 0.0),
        "total_amount_paid_sum_brl": float(feature_row.get("paid_lag_1") or 0.0),
        "avg_length_of_stay_days": float(predicted_los),
    }

    return pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)


def _forecast_with_gap_fill(
    fitted_models,
    base_df: pd.DataFrame,
    start_date: date,
    horizon_days: int,
):
    history_df = base_df.copy()
    history_df["week_start"] = pd.to_datetime(history_df["week_start"]).dt.date
    history_df = history_df.sort_values(
        ["health_facility_registry_code", "week_start"]
    ).reset_index(drop=True)

    requested_weeks = set(_make_future_week_starts(start_date, horizon_days))
    hospitals = sorted(history_df["health_facility_registry_code"].unique().tolist())

    if not hospitals:
        return pd.DataFrame(columns=[
            "hospital",
            "week_start",
            "estimated_total",
            "estimated_avg_length_of_stay",
        ])

    global_last_real_week = max(pd.to_datetime(history_df["week_start"]).dt.date.tolist())
    start_simulation_week = global_last_real_week + timedelta(days=7)
    end_requested_week = max(requested_weeks)

    if start_simulation_week > end_requested_week:
        all_weeks_to_simulate = sorted(requested_weeks)
    else:
        all_weeks_to_simulate = []
        cur = start_simulation_week
        while cur <= end_requested_week:
            all_weeks_to_simulate.append(cur)
            cur += timedelta(days=7)

    output_rows = []

    for future_ws in all_weeks_to_simulate:
        for hospital in hospitals:
            hospital_hist = history_df[
                history_df["health_facility_registry_code"] == hospital
            ].sort_values("week_start").copy()

            if hospital_hist.empty:
                continue

            feature_row = _build_future_feature_row(
                hospital=hospital,
                future_week_start=future_ws,
                history_df=history_df,
            )

            predicted_total = _predict_admissions(
                hospital=hospital,
                feature_row=feature_row,
                hospital_hist=hospital_hist,
                fitted_models=fitted_models,
            )

            predicted_los = _predict_los_simple(
                feature_row=feature_row,
                hospital_hist=hospital_hist,
            )

            history_df = _append_simulated_week(
                history_df=history_df,
                hospital=hospital,
                week_start_value=future_ws,
                predicted_total=predicted_total,
                predicted_los=predicted_los,
                feature_row=feature_row,
            )

            if future_ws in requested_weeks:
                output_rows.append(
                    {
                        "hospital": hospital,
                        "week_start": future_ws,
                        "estimated_total": round(float(predicted_total), 2),
                        "estimated_avg_length_of_stay": round(float(predicted_los), 2),
                    }
                )

    out = pd.DataFrame(output_rows).sort_values(["hospital", "week_start"])
    return out


def _rolling_backtest(
    full_df: pd.DataFrame,
    test_year: int,
):
    train_df = full_df[full_df["year"] < test_year].copy()
    test_df = full_df[full_df["year"] == test_year].copy()

    if train_df.empty:
        return {"ok": False, "error": f"No training data before test year {test_year}"}

    if test_df.empty:
        return {"ok": False, "error": f"No rows for test year {test_year}"}

    fitted_models = _fit_hospital_models(train_df)

    history_df = train_df[
        [
            "health_facility_registry_code",
            "week_start",
            "year",
            "week",
            "month",
            "quarter",
            "admissions_count",
            "deaths_count",
            "icu_days_sum",
            "total_amount_paid_sum_brl",
            "avg_length_of_stay_days",
        ]
    ].copy()

    test_base_df = test_df[
        [
            "health_facility_registry_code",
            "week_start",
            "year",
            "week",
            "month",
            "quarter",
            "admissions_count",
            "deaths_count",
            "icu_days_sum",
            "total_amount_paid_sum_brl",
            "avg_length_of_stay_days",
        ]
    ].copy()

    predictions = []

    ordered_weeks = sorted(pd.to_datetime(test_base_df["week_start"]).dt.date.unique().tolist())
    hospitals = sorted(test_base_df["health_facility_registry_code"].unique().tolist())

    for week_start_value in ordered_weeks:
        for hospital in hospitals:
            actual_row_df = test_base_df[
                (test_base_df["health_facility_registry_code"] == hospital)
                & (pd.to_datetime(test_base_df["week_start"]).dt.date == week_start_value)
            ].copy()

            if actual_row_df.empty:
                continue

            hospital_hist = history_df[
                history_df["health_facility_registry_code"] == hospital
            ].sort_values("week_start")

            feature_row = _build_future_feature_row(
                hospital=hospital,
                future_week_start=week_start_value,
                history_df=history_df,
            )

            pred_total = _predict_admissions(
                hospital=hospital,
                feature_row=feature_row,
                hospital_hist=hospital_hist,
                fitted_models=fitted_models,
            )

            pred_los = _predict_los_simple(
                feature_row=feature_row,
                hospital_hist=hospital_hist,
            )

            actual_row = actual_row_df.iloc[0]

            predictions.append(
                {
                    "hospital": hospital,
                    "week_start": week_start_value,
                    "real_total": float(actual_row["admissions_count"]),
                    "estimated_total": round(float(pred_total), 2),
                    "real_avg_length_of_stay": actual_row["avg_length_of_stay_days"],
                    "estimated_avg_length_of_stay": round(float(pred_los), 2),
                }
            )

            history_df = pd.concat(
                [history_df, actual_row_df],
                ignore_index=True,
            )

    if not predictions:
        return {"ok": False, "error": "No rolling predictions were generated"}

    out_df = pd.DataFrame(predictions).sort_values(["hospital", "week_start"])

    y_true = out_df["real_total"].astype(float)
    y_pred = out_df["estimated_total"].astype(float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    los_eval_df = out_df[out_df["real_avg_length_of_stay"].notna()].copy()

    los_mae = None
    los_rmse = None
    los_r2 = None

    if not los_eval_df.empty:
        los_true = los_eval_df["real_avg_length_of_stay"].astype(float)
        los_pred = los_eval_df["estimated_avg_length_of_stay"].astype(float)

        los_mae = float(mean_absolute_error(los_true, los_pred))
        los_rmse = float(root_mean_squared_error(los_true, los_pred))
        los_r2 = float(r2_score(los_true, los_pred))

    return {
        "ok": True,
        "out_df": out_df,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "los_mae": los_mae,
        "los_rmse": los_rmse,
        "los_r2": los_r2,
        "fitted_models": fitted_models,
    }


def _persist_predictions(
    forecast_df: pd.DataFrame,
    model_out: str,
    csv_out: str,
):
    with transaction.atomic():
        run = HospitalOccupancyPredictionRun.objects.create(
            model_path=model_out,
            csv_path=csv_out,
            rows_count=int(len(forecast_df)),
        )

        audit_objects = []
        current_objects = []

        for row in forecast_df.itertuples(index=False):
            hospital = str(row.hospital).strip().upper()
            week_start = row.week_start
            estimated_total = float(row.estimated_total)
            estimated_avg_length_of_stay = float(row.estimated_avg_length_of_stay)

            audit_objects.append(
                HospitalOccupancyPredictionAudit(
                    run=run,
                    hospital=hospital,
                    week_start=week_start,
                    estimated_total=estimated_total,
                    estimated_avg_length_of_stay=estimated_avg_length_of_stay,
                )
            )

            current_objects.append(
                HospitalOccupancyPrediction(
                    hospital=hospital,
                    week_start=week_start,
                    estimated_total=estimated_total,
                    estimated_avg_length_of_stay=estimated_avg_length_of_stay,
                )
            )

        HospitalOccupancyPredictionAudit.objects.bulk_create(
            audit_objects,
            batch_size=1000,
        )

        HospitalOccupancyPrediction.objects.all().delete()

        HospitalOccupancyPrediction.objects.bulk_create(
            current_objects,
            batch_size=1000,
        )

    return run


@shared_task
def run_hospital_occupancy_forecast(
    start_date=None,
    horizon_days=30,
    date_init=None,
    date_end=None,
    date_de_teste=None,
    min_weeks=20,
    csv_out="/app/models/hospital_forecast_next_30d.csv",
    model_out="/app/models/hospital_occupancy.joblib",
):
    date_init = _parse_date(date_init)
    date_end = _parse_date(date_end)
    date_de_teste = _parse_date(date_de_teste)
    start_date = _parse_date(start_date) or date.today()
    start_date = _week_start(start_date)

    df = _build_weekly_dataset(date_init=date_init, date_end=date_end)

    if df.empty:
        return {"ok": False, "error": "No data found"}

    df = _add_features(df, min_weeks=min_weeks)

    if df.empty:
        return {"ok": False, "error": "No hospitals with enough weeks to train"}

    df["week_start"] = pd.to_datetime(df["week_start"]).dt.date
    available_years = sorted(df["year"].dropna().astype(int).unique().tolist())

    if date_de_teste:
        test_year = int(date_de_teste.year)

        if test_year not in available_years:
            return {
                "ok": False,
                "error": f"Test year {test_year} not found in dataset",
                "available_years": available_years,
            }

        backtest = _rolling_backtest(
            full_df=df,
            test_year=test_year,
        )

        if not backtest["ok"]:
            return backtest

        out_df = backtest["out_df"]

        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "fitted_models": backtest["fitted_models"],
                "feature_cols": _training_feature_cols(),
            },
            model_out,
        )

        Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(csv_out, index=False)

        return {
            "ok": True,
            "mode": "test",
            "test_year": test_year,
            "available_years": available_years,
            "model_out": model_out,
            "csv_out": csv_out,
            "mae": backtest["mae"],
            "rmse": backtest["rmse"],
            "r2": backtest["r2"],
            "los_mae": backtest["los_mae"],
            "los_rmse": backtest["los_rmse"],
            "los_r2": backtest["los_r2"],
            "rows": int(len(out_df)),
            "trained_hospitals": int(len(backtest["fitted_models"]["hospital_models"])),
        }

    fitted_models = _fit_hospital_models(df)

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "fitted_models": fitted_models,
            "feature_cols": _training_feature_cols(),
        },
        model_out,
    )

    base_df = df[
        [
            "health_facility_registry_code",
            "week_start",
            "year",
            "week",
            "month",
            "quarter",
            "admissions_count",
            "deaths_count",
            "icu_days_sum",
            "total_amount_paid_sum_brl",
            "avg_length_of_stay_days",
        ]
    ].copy()

    forecast_df = _forecast_with_gap_fill(
        fitted_models=fitted_models,
        base_df=base_df,
        start_date=start_date,
        horizon_days=int(horizon_days),
    )

    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(csv_out, index=False)

    run = _persist_predictions(
        forecast_df=forecast_df,
        model_out=model_out,
        csv_out=csv_out,
    )

    return {
        "ok": True,
        "mode": "forecast",
        "run_id": run.id,
        "start_date": start_date.isoformat(),
        "horizon_days": int(horizon_days),
        "available_years": available_years,
        "model_out": model_out,
        "csv_out": csv_out,
        "rows": int(len(forecast_df)),
        "trained_hospitals": int(len(fitted_models["hospital_models"])),
    }