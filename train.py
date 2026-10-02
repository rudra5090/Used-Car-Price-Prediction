"""
Training pipeline for the Used Car Price Predictor.

Mirrors "Used_Car_Price_Prediction_Improved (1).ipynb" exactly (same cleaning,
same features, same IQR outlier rule, same CatBoost hyper-parameters) and saves
everything the Streamlit UI (app.py) needs:

  used_car_price_catboost.cbm   trained CatBoost model
  used_car_model_metadata.pkl   features, medians, metrics, price stats
  ui_options.json               cascading dropdown options + sample cars

Run:  python train.py
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor

DATA_PATH = "used_cars.csv"
CURRENT_YEAR = 2026

# ---------------------------------------------------------------- 1. Load + clean
df = pd.read_csv(DATA_PATH)

data = df.copy()
data = data.drop_duplicates().reset_index(drop=True)


def clean_number(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )


data["milage"] = clean_number(data["milage"])
data["price"] = clean_number(data["price"])

categorical_raw = data.select_dtypes(include="object").columns
for col in categorical_raw:
    data[col] = data[col].fillna("Unknown").astype(str).str.strip()

data = data.dropna(subset=["price", "milage"]).reset_index(drop=True)

# ---------------------------------------------------------------- 2. Feature engineering
data["car_age"] = (CURRENT_YEAR - data["model_year"]).clip(lower=0)

engine_text = data["engine"].astype(str)
data["horsepower"] = pd.to_numeric(
    engine_text.str.extract(r"(\d+(?:\.\d+)?)HP", expand=False), errors="coerce"
)
data["engine_liters"] = pd.to_numeric(
    engine_text.str.extract(r"(\d+(?:\.\d+)?)L", expand=False), errors="coerce"
)
data["cylinders"] = pd.to_numeric(
    engine_text.str.extract(r"(\d+)\s*Cylinder", expand=False), errors="coerce"
)

for col in ["horsepower", "engine_liters", "cylinders"]:
    data[col] = data[col].fillna(data[col].median())

data["mileage_per_year"] = data["milage"] / data["car_age"].clip(lower=1)

# ---------------------------------------------------------------- 3. IQR outlier rule
q1 = data["price"].quantile(0.25)
q3 = data["price"].quantile(0.75)
iqr = q3 - q1
lower_bound = max(0, q1 - 1.5 * iqr)
upper_bound = q3 + 1.5 * iqr
before = len(data)
data = data[
    (data["price"] >= lower_bound) & (data["price"] <= upper_bound)
].reset_index(drop=True)
print(f"IQR bounds: ${lower_bound:,.0f} - ${upper_bound:,.0f}  |  rows removed: {before - len(data)}")

# ---------------------------------------------------------------- 4. Train
target = "price"
features = [
    "brand", "model", "model_year", "milage",
    "fuel_type", "engine", "transmission",
    "ext_col", "int_col", "accident", "clean_title",
    "car_age", "horsepower", "engine_liters",
    "cylinders", "mileage_per_year",
]

X = data[features].copy()
y = data[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

categorical_features = X.select_dtypes(include="object").columns.tolist()
cat_indices = [X_train.columns.get_loc(col) for col in categorical_features]

catboost_model = CatBoostRegressor(
    iterations=1000,
    depth=8,
    learning_rate=0.04,
    loss_function="RMSE",
    l2_leaf_reg=6,
    random_seed=42,
    verbose=False,
)
catboost_model.fit(X_train, y_train, cat_features=cat_indices)

cat_pred = catboost_model.predict(X_test)
cat_r2 = r2_score(y_test, cat_pred)
cat_mae = mean_absolute_error(y_test, cat_pred)
cat_rmse = mean_squared_error(y_test, cat_pred) ** 0.5

print("CatBoost Regressor")
print(f"R2   : {cat_r2:.4f}")
print(f"MAE  : ${cat_mae:,.2f}")
print(f"RMSE : ${cat_rmse:,.2f}")

# ---------------------------------------------------------------- 5. Save artifacts
MODEL_FILE = "used_car_price_catboost.cbm"
catboost_model.save_model(MODEL_FILE)

metadata = {
    "features": features,
    "current_year": CURRENT_YEAR,
    "price_lower_bound": float(data["price"].min()),
    "price_upper_bound": float(data["price"].max()),
    "medians": {
        "horsepower": float(data["horsepower"].median()),
        "engine_liters": float(data["engine_liters"].median()),
        "cylinders": float(data["cylinders"].median()),
    },
    "metrics": {"r2": float(cat_r2), "mae": float(cat_mae), "rmse": float(cat_rmse)},
    "rows_trained": int(len(data)),
    "price_stats": {
        "median": float(data["price"].median()),
        "mean": float(data["price"].mean()),
        "q25": float(data["price"].quantile(0.25)),
        "q75": float(data["price"].quantile(0.75)),
    },
    "price_sorted": np.sort(data["price"].to_numpy()).tolist(),
}
joblib.dump(metadata, "used_car_model_metadata.pkl")

# ---------------------------------------------------------------- 6. UI options
# Cascading dropdown options, filtered by brand+model (with global fallback),
# so the UI only ever offers combinations that exist in the data.
DETAIL_SEP = "||"


def uniq(series):
    return sorted(series.dropna().astype(str).unique().tolist())


option_cols = [
    "fuel_type", "transmission", "engine",
    "ext_col", "int_col", "accident", "clean_title",
]

details = {}
for (brand, model_name), grp in data.groupby(["brand", "model"]):
    details[f"{brand}{DETAIL_SEP}{model_name}"] = {
        "years": sorted(grp["model_year"].unique().tolist(), reverse=True),
        **{c: uniq(grp[c]) for c in option_cols},
    }

samples = []
for b in data["brand"].value_counts().head(6).index:
    row = data[data["brand"] == b].iloc[0]
    samples.append({
        "label": f"{b} {int(row['model_year'])}",
        "brand": row["brand"],
        "model": row["model"],
        "model_year": int(row["model_year"]),
        "milage": int(row["milage"]),
        "fuel_type": row["fuel_type"],
        "engine": row["engine"],
        "transmission": row["transmission"],
        "ext_col": row["ext_col"],
        "int_col": row["int_col"],
        "accident": row["accident"],
        "clean_title": row["clean_title"],
    })
    if len(samples) == 3:
        break

ui_options = {
    "brands": uniq(data["brand"]),
    "models_by_brand": {b: uniq(g["model"]) for b, g in data.groupby("brand")},
    "details": details,
    "global": {
        "years": sorted(data["model_year"].unique().tolist(), reverse=True),
        **{c: uniq(data[c]) for c in option_cols},
    },
    "mileage_max": int(np.ceil(data["milage"].max() / 10000) * 10000),
    "samples": samples,
}
with open("ui_options.json", "w", encoding="utf-8") as f:
    json.dump(ui_options, f, ensure_ascii=False, indent=1)

print("Saved:")
print(" -", MODEL_FILE)
print(" - used_car_model_metadata.pkl")
print(" - ui_options.json")
print("Ready. Run:  streamlit run app.py")
