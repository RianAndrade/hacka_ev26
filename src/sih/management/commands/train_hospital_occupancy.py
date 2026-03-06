from pathlib import Path
import math

import joblib
import pandas as pd

from django.core.management.base import BaseCommand
from django.db.models import Sum, Count, Case, When, IntegerField, Value
from django.db.models.functions import TruncWeek, ExtractYear, ExtractWeek, ExtractMonth, ExtractQuarter

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from sih.models import HospitalAdmission


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument("--train-from", type=int, default=2021)
        parser.add_argument("--train-to", type=int, default=2023)
        parser.add_argument("--test-year", type=int, default=2024)
        parser.add_argument("--min-weeks", type=int, default=20)
        parser.add_argument("--csv-out", type=str, default="/app/models/hospital_pred_vs_real.csv")
        parser.add_argument("--model-out", type=str, default="/app/models/hospital_occupancy.joblib")

    def handle(self, *args, **opts):

        train_from = opts["train_from"]
        train_to = opts["train_to"]
        test_year = opts["test_year"]
        min_weeks = opts["min_weeks"]
        csv_out = opts["csv_out"]
        model_out = opts["model_out"]

        qs = (
            HospitalAdmission.objects.exclude(admission_date__isnull=True)
            .annotate(week_start=TruncWeek("admission_date"))
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
            self.stdout.write(self.style.ERROR("No data found"))
            return

        df = pd.DataFrame(data)

        df["health_facility_registry_code"] = (
            df["health_facility_registry_code"].astype(str).str.strip().str.upper()
        )

        df["deaths_count"] = df["deaths_count"].fillna(0)
        df["icu_days_sum"] = df["icu_days_sum"].fillna(0)
        df["total_amount_paid_sum_brl"] = df["total_amount_paid_sum_brl"].fillna(0)

        df = df.sort_values(["health_facility_registry_code", "week_start"])

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
        df = df[df[group].isin(keep)]

        train_df = df[(df["year"] >= train_from) & (df["year"] <= train_to)]
        test_df = df[df["year"] == test_year]

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

        X_train = train_df[feature_cols]
        y_train = train_df[target]

        X_test = test_df[feature_cols]
        y_test = test_df[target]

        categorical = ["health_facility_registry_code"]
        numeric = [c for c in feature_cols if c not in categorical]

        preprocessor = ColumnTransformer(
            transformers=[
                ("hospital", OneHotEncoder(handle_unknown="ignore"), categorical),
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median"))
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

        pipe = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("model", model),
            ]
        )

        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        Path(model_out).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipe, model_out)

        pred_df = test_df[[
            "health_facility_registry_code",
            "week_start",
            "admissions_count"
        ]].copy()

        pred_df["estimated_total"] = preds

        pred_df = pred_df.rename(
            columns={
                "health_facility_registry_code": "hospital",
                "admissions_count": "real_total",
            }
        )

        pred_df["estimated_total"] = pred_df["estimated_total"].round(2)

        pred_df = pred_df[[
            "hospital",
            "week_start",
            "real_total",
            "estimated_total"
        ]]

        Path(csv_out).parent.mkdir(parents=True, exist_ok=True)

        pred_df.to_csv(csv_out, index=False)

        self.stdout.write(
            self.style.SUCCESS(
                f"MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f} | model={model_out}"
            )
        )