import math
from pathlib import Path
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd

from celery import shared_task
from django.db import transaction

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

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


def _safe_div_series(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    den = den.replace(0, np.nan)
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
            hospital_q10_admissions=("admissions_count", lambda s: float(pd.Series(s).quantile(0.10))),
            hospital_q90_admissions=("admissions_count", lambda s: float(pd.Series(s).quantile(0.90))),
            hospital_mean_los=("avg_length_of_stay_days", "mean"),
            hospital_median_los=("avg_length_of_stay_days", "median"),
            hospital_min_los=("avg_length_of_stay_days", "min"),
            hospital_max_los=("avg_length_of_stay_days", "max"),
            hospital_q10_los=("avg_length_of_stay_days", lambda s: float(pd.Series(s).dropna().quantile(0.10)) if pd.Series(s).dropna().shape[0] else np.nan),
            hospital_q90_los=("avg_length_of_stay_days", lambda s: float(pd.Series(s).dropna().quantile(0.90)) if pd.Series(s).dropna().shape[0] else np.nan),
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

    df["recent_mean_6"] = (
        df.groupby(group)["admissions_count"]
        .shift(1)
        .groupby(df[group])
        .rolling(6)
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

    df["recent_los_mean_6"] = (
        df.groupby(group)["avg_length_of_stay_days"]
        .shift(1)
        .groupby(df[group])
        .rolling(6)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["admissions_ratio_1"] = _safe_div_series(
        df["admissions_count"],
        df["admissions_lag_1"],
    )
    df["admissions_ratio_4"] = _safe_div_series(
        df["admissions_count"],
        df["roll_mean_4"],
    )
    df["los_ratio_1"] = _safe_div_series(
        df["avg_length_of_stay_days"],
        df["los_lag_1"],
    )

    df["week_sin"] = df["week"].apply(lambda x: math.sin(2 * math.pi * x / 52))
    df["week_cos"] = df["week"].apply(lambda x: math.cos(2 * math.pi * x / 52))

    df = _add_hospital_stats(df)

    counts = df.groupby(group)["week_start"].nunique()
    keep = counts[counts >= min_weeks].index
    df = df[df[group].isin(keep)].copy()

    return df


def _build_pipeline(feature_cols: list[str], n_estimators: int = 1400):
    categorical = ["health_facility_registry_code"]
    numeric = [c for c in feature_cols if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("hospital", OneHotEncoder(handle_unknown="ignore"), categorical),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric,
            ),
        ]
    )

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    return Pipeline(
        steps=[
            ("pre", preprocessor),
            ("model", model),
        ]
    )


def _training_feature_cols():
    return [
        "health_facility_registry_code",
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
        "los_lag_26",
        "los_lag_52",
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
        "recent_mean_6",
        "recent_los_mean_3",
        "recent_los_mean_6",
        "seasonal_week_mean_admissions",
        "seasonal_week_median_admissions",
        "seasonal_week_mean_los",
        "seasonal_week_median_los",
        "hospital_mean_admissions",
        "hospital_median_admissions",
        "hospital_min_admissions",
        "hospital_max_admissions",
        "hospital_q10_admissions",
        "hospital_q90_admissions",
        "hospital_mean_los",
        "hospital_median_los",
        "hospital_min_los",
        "hospital_max_los",
        "hospital_q10_los",
        "hospital_q90_los",
        "admissions_ratio_1",
        "admissions_ratio_4",
        "los_ratio_1",
    ]


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

    def lag(series: pd.Series, n: int):
        if len(series) < n:
            return None
        return float(series.iloc[-n])

    def roll_mean(series: pd.Series, window: int):
        tail = series.tail(window).dropna()
        if tail.empty:
            return None
        return float(tail.mean())

    def roll_std(series: pd.Series, window: int):
        tail = series.tail(window).dropna()
        if tail.empty:
            return None
        value = float(tail.std()) if len(tail) > 1 else 0.0
        return 0.0 if math.isnan(value) else value

    iso = future_week_start.isocalendar()
    week_num = int(iso.week)

    same_week_hist = hist[hist["week"] == week_num]

    admissions_lag_1 = lag(admissions_series, 1)
    admissions_lag_2 = lag(admissions_series, 2)
    roll_mean_4 = roll_mean(admissions_series, 4)

    los_lag_1 = lag(los_series, 1)
    los_lag_2 = lag(los_series, 2)

    row = {
        "health_facility_registry_code": hospital,
        "week": week_num,
        "month": future_week_start.month,
        "quarter": ((future_week_start.month - 1) // 3) + 1,
        "week_sin": math.sin(2 * math.pi * week_num / 52),
        "week_cos": math.cos(2 * math.pi * week_num / 52),
        "admissions_lag_1": admissions_lag_1,
        "admissions_lag_2": admissions_lag_2,
        "admissions_lag_3": lag(admissions_series, 3),
        "admissions_lag_4": lag(admissions_series, 4),
        "admissions_lag_8": lag(admissions_series, 8),
        "admissions_lag_12": lag(admissions_series, 12),
        "admissions_lag_26": lag(admissions_series, 26),
        "admissions_lag_52": lag(admissions_series, 52),
        "deaths_lag_1": lag(deaths_series, 1),
        "deaths_lag_2": lag(deaths_series, 2),
        "deaths_lag_4": lag(deaths_series, 4),
        "icu_lag_1": lag(icu_series, 1),
        "icu_lag_2": lag(icu_series, 2),
        "icu_lag_4": lag(icu_series, 4),
        "paid_lag_1": lag(paid_series, 1),
        "paid_lag_2": lag(paid_series, 2),
        "los_lag_1": los_lag_1,
        "los_lag_2": los_lag_2,
        "los_lag_3": lag(los_series, 3),
        "los_lag_4": lag(los_series, 4),
        "los_lag_8": lag(los_series, 8),
        "los_lag_12": lag(los_series, 12),
        "los_lag_26": lag(los_series, 26),
        "los_lag_52": lag(los_series, 52),
        "roll_mean_2": roll_mean(admissions_series, 2),
        "roll_mean_4": roll_mean_4,
        "roll_mean_8": roll_mean(admissions_series, 8),
        "roll_mean_12": roll_mean(admissions_series, 12),
        "roll_std_2": roll_std(admissions_series, 2),
        "roll_std_4": roll_std(admissions_series, 4),
        "roll_std_8": roll_std(admissions_series, 8),
        "roll_std_12": roll_std(admissions_series, 12),
        "los_roll_mean_2": roll_mean(los_series, 2),
        "los_roll_mean_4": roll_mean(los_series, 4),
        "los_roll_mean_8": roll_mean(los_series, 8),
        "los_roll_mean_12": roll_mean(los_series, 12),
        "los_roll_std_2": roll_std(los_series, 2),
        "los_roll_std_4": roll_std(los_series, 4),
        "los_roll_std_8": roll_std(los_series, 8),
        "los_roll_std_12": roll_std(los_series, 12),
        "recent_mean_3": roll_mean(admissions_series, 3),
        "recent_mean_6": roll_mean(admissions_series, 6),
        "recent_los_mean_3": roll_mean(los_series, 3),
        "recent_los_mean_6": roll_mean(los_series, 6),
        "seasonal_week_mean_admissions": _safe_mean(same_week_hist["admissions_count"]),
        "seasonal_week_median_admissions": _safe_quantile(same_week_hist["admissions_count"], 0.50),
        "seasonal_week_mean_los": _safe_mean(same_week_hist["avg_length_of_stay_days"]),
        "seasonal_week_median_los": _safe_quantile(same_week_hist["avg_length_of_stay_days"], 0.50),
        "hospital_mean_admissions": _safe_mean(hist["admissions_count"]),
        "hospital_median_admissions": _safe_quantile(hist["admissions_count"], 0.50),
        "hospital_min_admissions": _safe_min(admissions_series),
        "hospital_max_admissions": _safe_max(admissions_series),
        "hospital_q10_admissions": _safe_quantile(hist["admissions_count"], 0.10),
        "hospital_q90_admissions": _safe_quantile(hist["admissions_count"], 0.90),
        "hospital_mean_los": _safe_mean(hist["avg_length_of_stay_days"]),
        "hospital_median_los": _safe_quantile(hist["avg_length_of_stay_days"], 0.50),
        "hospital_min_los": _safe_quantile(hist["avg_length_of_stay_days"], 0.00),
        "hospital_max_los": _safe_quantile(hist["avg_length_of_stay_days"], 1.00),
        "hospital_q10_los": _safe_quantile(hist["avg_length_of_stay_days"], 0.10),
        "hospital_q90_los": _safe_quantile(hist["avg_length_of_stay_days"], 0.90),
        "admissions_ratio_1": _safe_div_value(admissions_lag_1, admissions_lag_2),
        "admissions_ratio_4": _safe_div_value(admissions_lag_1, roll_mean_4),
        "los_ratio_1": _safe_div_value(los_lag_1, los_lag_2),
    }

    return row


def _blend_future_admissions(
    hospital_hist: pd.DataFrame,
    week_num: int,
    model_pred: float,
    horizon_index: int,
):
    admissions = pd.to_numeric(hospital_hist["admissions_count"], errors="coerce").dropna()

    if admissions.empty:
        return max(float(model_pred), 0.0)

    recent_4 = float(admissions.tail(4).mean())
    recent_8 = float(admissions.tail(8).mean()) if len(admissions) >= 8 else recent_4
    last_value = float(admissions.iloc[-1])
    hist_mean = float(admissions.mean())

    seasonal_hist = hospital_hist[hospital_hist["week"] == week_num]
    seasonal_mean = _safe_mean(seasonal_hist["admissions_count"])
    seasonal_median = _safe_quantile(seasonal_hist["admissions_count"], 0.50)

    seasonal_value = seasonal_mean if seasonal_mean is not None else seasonal_median
    if seasonal_value is None:
        seasonal_value = hist_mean

    if horizon_index <= 4:
        w_model, w_season, w_recent = 0.55, 0.25, 0.20
    elif horizon_index <= 8:
        w_model, w_season, w_recent = 0.45, 0.30, 0.25
    else:
        w_model, w_season, w_recent = 0.35, 0.35, 0.30

    recent_value = 0.6 * recent_4 + 0.4 * recent_8

    blended = (
        w_model * model_pred
        + w_season * seasonal_value
        + w_recent * recent_value
    )

    blended = 0.90 * blended + 0.10 * last_value

    return blended


def _blend_future_los(
    hospital_hist: pd.DataFrame,
    week_num: int,
    model_pred: float,
    horizon_index: int,
):
    los = pd.to_numeric(hospital_hist["avg_length_of_stay_days"], errors="coerce").dropna()

    if los.empty:
        return max(float(model_pred), 0.0)

    recent_4 = float(los.tail(4).mean())
    hist_mean = float(los.mean())

    seasonal_hist = hospital_hist[hospital_hist["week"] == week_num]
    seasonal_mean = _safe_mean(seasonal_hist["avg_length_of_stay_days"])
    seasonal_value = seasonal_mean if seasonal_mean is not None else hist_mean

    if horizon_index <= 4:
        w_model, w_season, w_recent = 0.60, 0.20, 0.20
    elif horizon_index <= 8:
        w_model, w_season, w_recent = 0.50, 0.25, 0.25
    else:
        w_model, w_season, w_recent = 0.40, 0.30, 0.30

    return (
        w_model * model_pred
        + w_season * seasonal_value
        + w_recent * recent_4
    )


def _clip_admissions_with_history(predicted_value: float, hospital_hist: pd.DataFrame):
    admissions = pd.to_numeric(hospital_hist["admissions_count"], errors="coerce").dropna()

    if admissions.empty:
        return max(float(predicted_value), 0.0)

    hist_mean = float(admissions.mean())
    hist_min = float(admissions.min())
    hist_max = float(admissions.max())
    q10 = _safe_quantile(admissions, 0.10)
    q90 = _safe_quantile(admissions, 0.90)
    recent_8 = float(admissions.tail(8).mean()) if len(admissions) >= 8 else hist_mean

    lower = max(
        hist_min * 0.90,
        q10 * 0.90 if q10 is not None else 0.0,
        hist_mean * 0.65,
        recent_8 * 0.70,
    )
    upper = max(
        hist_max * 1.15,
        q90 * 1.10 if q90 is not None else hist_max,
        hist_mean * 1.35,
    )

    clipped = float(min(max(predicted_value, lower), upper))
    return max(clipped, 0.0)


def _clip_los_with_history(predicted_value: float, hospital_hist: pd.DataFrame):
    los = pd.to_numeric(hospital_hist["avg_length_of_stay_days"], errors="coerce").dropna()

    if los.empty:
        return max(float(predicted_value), 0.0)

    hist_mean = float(los.mean())
    hist_min = float(los.min())
    hist_max = float(los.max())
    q10 = _safe_quantile(los, 0.10)
    q90 = _safe_quantile(los, 0.90)

    lower = max(0.0, min(hist_min * 0.85, (q10 if q10 is not None else hist_mean) * 0.85))
    upper = max(hist_max * 1.15, (q90 if q90 is not None else hist_mean) * 1.15)

    clipped = float(min(max(predicted_value, lower), upper))
    return max(clipped, 0.0)


def _iterative_forecast(
    model_adm,
    model_los,
    base_df: pd.DataFrame,
    start_date: date,
    horizon_days: int,
    feature_cols: list[str],
):
    history_df = base_df.copy()
    history_df["week_start"] = pd.to_datetime(history_df["week_start"]).dt.date
    history_df = history_df.sort_values(
        ["health_facility_registry_code", "week_start"]
    ).reset_index(drop=True)

    future_weeks = _make_future_week_starts(start_date, horizon_days)
    hospitals = sorted(history_df["health_facility_registry_code"].unique().tolist())

    output_rows = []

    for horizon_index, future_ws in enumerate(future_weeks, start=1):
        step_rows = []

        for hospital in hospitals:
            step_rows.append(
                _build_future_feature_row(
                    hospital=hospital,
                    future_week_start=future_ws,
                    history_df=history_df,
                )
            )

        step_df = pd.DataFrame(step_rows)

        pred_log_adm = model_adm.predict(step_df[feature_cols])
        pred_los = model_los.predict(step_df[feature_cols])

        pred_total_raw = np.expm1(pred_log_adm)

        for idx, hospital in enumerate(hospitals):
            hospital_hist = history_df[
                history_df["health_facility_registry_code"] == hospital
            ].sort_values("week_start")

            week_num = int(future_ws.isocalendar().week)

            model_total = float(pred_total_raw[idx])
            model_los_value = float(pred_los[idx])

            blended_total = _blend_future_admissions(
                hospital_hist=hospital_hist,
                week_num=week_num,
                model_pred=model_total,
                horizon_index=horizon_index,
            )

            blended_los = _blend_future_los(
                hospital_hist=hospital_hist,
                week_num=week_num,
                model_pred=model_los_value,
                horizon_index=horizon_index,
            )

            final_total = _clip_admissions_with_history(
                predicted_value=blended_total,
                hospital_hist=hospital_hist,
            )

            final_los = _clip_los_with_history(
                predicted_value=blended_los,
                hospital_hist=hospital_hist,
            )

            last_hist = hospital_hist.iloc[-1]

            new_row = {
                "health_facility_registry_code": hospital,
                "week_start": future_ws,
                "year": future_ws.isocalendar().year,
                "week": int(future_ws.isocalendar().week),
                "month": future_ws.month,
                "quarter": ((future_ws.month - 1) // 3) + 1,
                "admissions_count": final_total,
                "deaths_count": float(pd.to_numeric(last_hist["deaths_count"], errors="coerce")),
                "icu_days_sum": float(pd.to_numeric(last_hist["icu_days_sum"], errors="coerce")),
                "total_amount_paid_sum_brl": float(
                    pd.to_numeric(last_hist["total_amount_paid_sum_brl"], errors="coerce")
                ),
                "avg_length_of_stay_days": final_los,
            }

            history_df = pd.concat(
                [history_df, pd.DataFrame([new_row])],
                ignore_index=True,
            )

            output_rows.append(
                {
                    "hospital": hospital,
                    "week_start": future_ws,
                    "estimated_total": round(final_total, 2),
                    "estimated_avg_length_of_stay": round(final_los, 2),
                }
            )

    out = pd.DataFrame(output_rows).sort_values(["hospital", "week_start"])
    return out


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

    df = _build_weekly_dataset(date_init=date_init, date_end=date_end)

    if df.empty:
        return {"ok": False, "error": "No data found"}

    df = _add_features(df, min_weeks=min_weeks)

    if df.empty:
        return {"ok": False, "error": "No hospitals with enough weeks to train"}

    feature_cols = _training_feature_cols()

    df["week_start"] = pd.to_datetime(df["week_start"]).dt.date
    available_years = sorted(df["year"].dropna().astype(int).unique().tolist())

    model_adm = _build_pipeline(feature_cols=feature_cols, n_estimators=1600)
    model_los = _build_pipeline(feature_cols=feature_cols, n_estimators=1000)

    if date_de_teste:
        test_year = int(date_de_teste.year)

        if test_year not in available_years:
            return {
                "ok": False,
                "error": f"Test year {test_year} not found in dataset",
                "available_years": available_years,
            }

        train_df = df[df["year"] < test_year].copy()
        test_df = df[df["year"] == test_year].copy()

        if train_df.empty:
            return {
                "ok": False,
                "error": f"No training data before test year {test_year}",
            }

        if test_df.empty:
            return {
                "ok": False,
                "error": f"No rows for test year {test_year}",
            }

        y_train_adm = np.log1p(train_df["admissions_count"].astype(float))
        y_test_adm = test_df["admissions_count"].astype(float)

        los_train_mask = train_df["avg_length_of_stay_days"].notna()
        los_test_mask = test_df["avg_length_of_stay_days"].notna()

        model_adm.fit(train_df[feature_cols], y_train_adm)

        if los_train_mask.any():
            model_los.fit(
                train_df.loc[los_train_mask, feature_cols],
                train_df.loc[los_train_mask, "avg_length_of_stay_days"].astype(float),
            )
        else:
            return {"ok": False, "error": "No LOS data available to train"}

        pred_total = np.expm1(model_adm.predict(test_df[feature_cols]))
        pred_los = model_los.predict(test_df[feature_cols])

        mae = mean_absolute_error(y_test_adm, pred_total)
        rmse = root_mean_squared_error(y_test_adm, pred_total)
        r2 = r2_score(y_test_adm, pred_total)

        los_mae = None
        los_rmse = None
        los_r2 = None

        if los_test_mask.any():
            los_true = test_df.loc[los_test_mask, "avg_length_of_stay_days"].astype(float)
            los_pred = pd.Series(pred_los, index=test_df.index).loc[los_test_mask]

            los_mae = float(mean_absolute_error(los_true, los_pred))
            los_rmse = float(root_mean_squared_error(los_true, los_pred))
            los_r2 = float(r2_score(los_true, los_pred))

        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_adm": model_adm,
                "model_los": model_los,
                "feature_cols": feature_cols,
            },
            model_out,
        )

        out_df = test_df[
            [
                "health_facility_registry_code",
                "week_start",
                "admissions_count",
                "avg_length_of_stay_days",
            ]
        ].copy()

        out_df["estimated_total"] = pd.Series(pred_total, index=out_df.index).round(2)
        out_df["estimated_avg_length_of_stay"] = pd.Series(pred_los, index=out_df.index).round(2)

        out_df = out_df.rename(
            columns={
                "health_facility_registry_code": "hospital",
                "admissions_count": "real_total",
                "avg_length_of_stay_days": "real_avg_length_of_stay",
            }
        )

        out_df["hospital"] = out_df["hospital"].astype(str).str.strip().str.upper()

        out_df = out_df[
            [
                "hospital",
                "week_start",
                "real_total",
                "estimated_total",
                "real_avg_length_of_stay",
                "estimated_avg_length_of_stay",
            ]
        ].sort_values(["hospital", "week_start"])

        Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(csv_out, index=False)

        return {
            "ok": True,
            "mode": "test",
            "test_year": test_year,
            "available_years": available_years,
            "model_out": model_out,
            "csv_out": csv_out,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "los_mae": los_mae,
            "los_rmse": los_rmse,
            "los_r2": los_r2,
            "rows": int(len(out_df)),
        }

    y_all_adm = np.log1p(df["admissions_count"].astype(float))
    los_all_mask = df["avg_length_of_stay_days"].notna()

    model_adm.fit(df[feature_cols], y_all_adm)

    if los_all_mask.any():
        model_los.fit(
            df.loc[los_all_mask, feature_cols],
            df.loc[los_all_mask, "avg_length_of_stay_days"].astype(float),
        )
    else:
        return {"ok": False, "error": "No LOS data available to train"}

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_adm": model_adm,
            "model_los": model_los,
            "feature_cols": feature_cols,
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

    forecast_df = _iterative_forecast(
        model_adm=model_adm,
        model_los=model_los,
        base_df=base_df,
        start_date=start_date,
        horizon_days=int(horizon_days),
        feature_cols=feature_cols,
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
    }