import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from arbovirus.models import DengueWeeklyDataset


def train_and_evaluate(municipality_code: str):

    qs = DengueWeeklyDataset.objects.order_by("week_start").values(
        "week_start",
        "epidemiological_year",
        "epidemiological_week",
        "temp_mean",
        "humidity_mean",
        "rain_sum",
        "pressure_mean",
        "wind_speed_mean",
        "solar_radiation_sum",
        "optimal_days",
        "dengue_cases"
    )

    data = list(qs)

    if not data:
        raise ValueError("dataset empty for municipality")

    df = pd.DataFrame(data)

    df["dengue_cases_next"] = df["dengue_cases"].shift(-1)

    df = df.dropna(subset=["dengue_cases_next"])

    features = [
        "temp_mean",
        "humidity_mean",
        "rain_sum",
        "pressure_mean",
        "wind_speed_mean",
        "solar_radiation_sum",
        "optimal_days",
        "dengue_cases"
    ]

    train = df[df["epidemiological_year"].between(2019, 2021)]
    test = df[df["epidemiological_year"] == 2022]

    X_train = train[features]
    y_train = train["dengue_cases_next"]

    X_test = test[features]
    y_test = test["dengue_cases_next"]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    joblib.dump(model, "/app/models/dengue_model.joblib")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "train_samples": int(len(train)),
        "test_samples": int(len(test))
    }