import math
from pathlib import Path
from datetime import date, datetime, timedelta

import joblib
import pandas as pd

from celery import shared_task
from django.db import transaction
from django.db.models import Sum, Count, Case, When, IntegerField, Value
from django.db.models.functions import TruncWeek, ExtractYear, ExtractWeek, ExtractMonth, ExtractQuarter

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

    qs = (
        qs.annotate(week_start=TruncWeek("admission_date"))
        .annotate(year=ExtractYear("week_start"))
        .annotate(week=ExtractWeek("week_start"))
        .annotate(month=ExtractMonth("week_start"))
        .annotate(quarter=ExtractQuarter("week_start"))
        .values(
            "health_facility_registry_code",
            "week_start",
            "year",
            "week",
            "month",
            "quarter",
        )
        .annotate(
            admissions_count=Count("record_identifier"),
            deaths_count=Sum(
                Case(
                    When(death_during_admission=True, then=Value(1)),
                    default=Value(0),
                    output_field=IntegerField(),
                )
            ),
            icu_days_sum=Sum("intensive_care_total_days"),
            total_amount_paid_sum_brl=Sum("total_amount_paid_brl"),
        )
        .order_by("health_facility_registry_code", "week_start")
    )

    data = list(qs)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    df["health_facility_registry_code"] = (
        df["health_facility_registry_code"].astype(str).str.strip().str.upper()
    )

    df["deaths_count"] = df["deaths_count"].fillna(0)
    df["icu_days_sum"] = df["icu_days_sum"].fillna(0)
    df["total_amount_paid_sum_brl"] = df["total_amount_paid_sum_brl"].fillna(0)

    df = df[df["health_facility_registry_code"].astype(str).str.strip() != ""]
    df = df.sort_values(["health_facility_registry_code", "week_start"])

    return df


def _add_features(df: pd.DataFrame, min_weeks: int) -> pd.DataFrame:
    group = "health_facility_registry_code"

    for lag in [1, 2, 3, 4, 8, 12, 26, 52]:
        df[f"admissions_lag_{lag}"] = df.groupby(group)["admissions_count"].shift(lag)
        df[f"deaths_lag_{lag}"] = df.groupby(group)["deaths_count"].shift(lag)
        df[f"icu_lag_{lag}"] = df.groupby(group)["icu_days_sum"].shift(lag)
        df[f"paid_lag_{lag}"] = df.groupby(group)["total_amount_paid_sum_brl"].shift(lag)

    for window in [2, 4, 8, 12]:
        base = df.groupby(group)["admissions_count"].shift(1)

        df[f"roll_mean_{window}"] = (
            base.groupby(df[group]).rolling(window).mean().reset_index(level=0, drop=True)
        )

        df[f"roll_std_{window}"] = (
            base.groupby(df[group]).rolling(window).std().reset_index(level=0, drop=True)
        )

    df["week_sin"] = df["week"].apply(lambda x: math.sin(2 * math.pi * x / 52))
    df["week_cos"] = df["week"].apply(lambda x: math.cos(2 * math.pi * x / 52))

    counts = df.groupby(group)["week_start"].nunique()
    keep = counts[counts >= min_weeks].index

    return df[df[group].isin(keep)].copy()


def _build_pipeline(feature_cols: list[str]):
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
        n_estimators=1200,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("pre", preprocessor),
            ("model", model),
        ]
    )


def _iterative_forecast(
    pipe,
    base_df: pd.DataFrame,
    start_date: date,
    horizon_days: int,
    feature_cols: list[str],
):
    base_df = base_df.copy()
    base_df["week_start"] = pd.to_datetime(base_df["week_start"]).dt.date

    future_weeks = _make_future_week_starts(start_date, horizon_days)

    future_rows = []
    hospitals = sorted(base_df["health_facility_registry_code"].unique().tolist())

    for hospital in hospitals:
        for ws in future_weeks:
            future_rows.append(
                {
                    "health_facility_registry_code": hospital,
                    "week_start": ws,
                    "year": ws.isocalendar().year,
                    "week": ws.isocalendar().week,
                    "month": ws.month,
                    "quarter": ((ws.month - 1) // 3) + 1,
                    "admissions_count": None,
                    "deaths_count": None,
                    "icu_days_sum": None,
                    "total_amount_paid_sum_brl": None,
                }
            )

    future_df = pd.DataFrame(future_rows)

    full = pd.concat([base_df, future_df], ignore_index=True)
    full = full.sort_values(["health_facility_registry_code", "week_start"]).reset_index(drop=True)

    mask_future = full["week_start"].isin(future_weeks)

    for ws in future_weeks:
        step_mask = mask_future & (full["week_start"] == ws)
        tmp = _add_features(full.copy(), min_weeks=1)
        X_step = tmp.loc[step_mask, feature_cols]
        preds = pipe.predict(X_step)
        full.loc[step_mask, "admissions_count"] = preds

    out = full[mask_future].copy()
    out["estimated_total"] = pd.to_numeric(out["admissions_count"], errors="coerce").round(2)

    out = out.rename(
        columns={
            "health_facility_registry_code": "hospital",
        }
    )

    out = out[
        [
            "hospital",
            "week_start",
            "estimated_total",
        ]
    ].sort_values(["hospital", "week_start"])

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

            audit_objects.append(
                HospitalOccupancyPredictionAudit(
                    run=run,
                    hospital=hospital,
                    week_start=week_start,
                    estimated_total=estimated_total,
                )
            )

            current_objects.append(
                HospitalOccupancyPrediction(
                    hospital=hospital,
                    week_start=week_start,
                    estimated_total=estimated_total,
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

    feature_cols = [
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
        "roll_mean_2",
        "roll_mean_4",
        "roll_mean_8",
        "roll_mean_12",
        "roll_std_2",
        "roll_std_4",
        "roll_std_8",
        "roll_std_12",
    ]

    target = "admissions_count"
    pipe = _build_pipeline(feature_cols=feature_cols)

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

        train_df = df[df["year"] < test_year]
        test_df = df[df["year"] == test_year]

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

        pipe.fit(train_df[feature_cols], train_df[target])

        preds = pipe.predict(test_df[feature_cols])

        mae = mean_absolute_error(test_df[target], preds)
        rmse = root_mean_squared_error(test_df[target], preds)
        r2 = r2_score(test_df[target], preds)

        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_out)

        out_df = test_df[
            [
                "health_facility_registry_code",
                "week_start",
                "admissions_count",
            ]
        ].copy()

        out_df["estimated_total"] = preds.round(2)

        out_df = out_df.rename(
            columns={
                "health_facility_registry_code": "hospital",
                "admissions_count": "real_total",
            }
        )

        out_df["hospital"] = out_df["hospital"].astype(str).str.strip().str.upper()

        out_df = out_df[
            [
                "hospital",
                "week_start",
                "real_total",
                "estimated_total",
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
            "rows": int(len(out_df)),
        }

    pipe.fit(df[feature_cols], df[target])

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)

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
        ]
    ].copy()

    forecast_df = _iterative_forecast(
        pipe=pipe,
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