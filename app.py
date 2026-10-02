"""
Used Car Price Predictor — Streamlit UI

Loads the model trained by train.py (or the notebook) and presents a polished,
cascading form with a live price-estimate card.

Run:  streamlit run app.py
"""

import json
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DETAIL_SEP = "||"

RAW_FIELDS = [
    "brand", "model", "model_year", "milage",
    "fuel_type", "engine", "transmission",
    "ext_col", "int_col", "accident", "clean_title",
]
MODEL_COLUMNS = RAW_FIELDS + [
    "car_age", "horsepower", "engine_liters", "cylinders", "mileage_per_year",
]


# ------------------------------------------------------------ artifacts
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model = CatBoostRegressor()
    model.load_model("used_car_price_catboost.cbm")
    meta = joblib.load("used_car_model_metadata.pkl")
    with open("ui_options.json", encoding="utf-8") as f:
        opts = json.load(f)
    return model, meta, opts


model, META, OPTS = load_artifacts()
YEAR = META["current_year"]
MEDIANS = META["medians"]
METRICS = META.get("metrics", {})
STATS = META.get("price_stats", {})
PRICE_SORTED = np.asarray(META.get("price_sorted", []))
MAE = METRICS.get("mae", 0.0)


# ------------------------------------------------------------ prediction
def _extract(pattern, text):
    m = re.search(pattern, str(text))
    return float(m.group(1)) if m else None


def predict_price(car: dict) -> float:
    hp = _extract(r"(\d+(?:\.\d+)?)HP", car["engine"])
    liters = _extract(r"(\d+(?:\.\d+)?)L", car["engine"])
    cyl = _extract(r"(\d+)\s*Cylinder", car["engine"])

    car_age = max(0, YEAR - int(car["model_year"]))
    row = {
        **{k: car[k] for k in RAW_FIELDS},
        "model_year": int(car["model_year"]),
        "milage": float(car["milage"]),
        "car_age": car_age,
        "horsepower": hp if hp is not None else MEDIANS["horsepower"],
        "engine_liters": liters if liters is not None else MEDIANS["engine_liters"],
        "cylinders": cyl if cyl is not None else MEDIANS["cylinders"],
        "mileage_per_year": float(car["milage"]) / max(1, car_age),
    }
    pred = model.predict(pd.DataFrame([row], columns=MODEL_COLUMNS))[0]
    return max(0.0, float(pred))


def percentile_of(price: float) -> float:
    if len(PRICE_SORTED) == 0:
        return 50.0
    return 100.0 * float(np.searchsorted(PRICE_SORTED, price)) / len(PRICE_SORTED)


USD_TO_INR = 96.378852


def money(v: float) -> str:
    amount = f"{v * USD_TO_INR:.0f}"
    if len(amount) <= 3:
        return f"₹{amount}"
    groups = [amount[-3:]]
    amount = amount[:-3]
    while amount:
        groups.insert(0, amount[-2:])
        amount = amount[:-2]
    return f"₹{','.join(groups)}"


# ------------------------------------------------------------ styles
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    .stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

    .hero {
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 45%, #06b6d4 100%);
        border-radius: 28px;
        padding: 36px 42px;
        margin-bottom: 22px;
        box-shadow: 0 18px 45px -18px rgba(79, 70, 229, .65);
    }
    .hero h1 { color: #fff; font-size: 2.3rem; font-weight: 800; margin: 0 0 6px; letter-spacing: -.5px; }
    .hero p  { color: rgba(255,255,255,.88); font-size: 1rem; margin: 0; }

    .section-label {
        font-size: .76rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase;
        color: #64748b; margin: 4px 0 10px;
    }

    .price-card {
        background: linear-gradient(165deg, #0f172a 0%, #1e293b 60%, #0f172a 100%);
        border-radius: 26px; padding: 32px 30px; color: #fff;
        box-shadow: 0 22px 50px -20px rgba(15, 23, 42, .75);
        border: 1px solid rgba(148, 163, 184, .18);
    }
    .price-card .cap { color: #94a3b8; font-size: .78rem; letter-spacing: .16em; text-transform: uppercase; }
    .price-card .big { font-size: 3.1rem; font-weight: 800; line-height: 1.12; margin: 6px 0 2px;
                       background: linear-gradient(90deg, #38bdf8, #a78bfa);
                       -webkit-background-clip: text; background-clip: text; color: transparent; }
    .price-card .range { color: #cbd5e1; font-size: .93rem; }
    .price-card .note { color: #64748b; font-size: .8rem; margin-top: 6px; }

    .chip { display: inline-block; background: rgba(56, 189, 248, .12); border: 1px solid rgba(56,189,248,.3);
            color: #7dd3fc; border-radius: 999px; padding: 5px 13px; margin: 12px 5px 0 0;
            font-size: .79rem; font-weight: 600; }
    .chip.alt { background: rgba(167, 139, 250, .12); border-color: rgba(167,139,250,.35); color: #c4b5fd; }

    .cmp { margin-top: 18px; }
    .cmp-row { display: flex; align-items: center; gap: 10px; margin: 9px 0; font-size: .84rem; color: #cbd5e1; }
    .cmp-row .lbl { width: 112px; flex: none; color: #94a3b8; }
    .cmp-row .track { flex: 1; height: 12px; background: rgba(148,163,184,.16); border-radius: 999px; overflow: hidden; }
    .cmp-row .fill { display: block; height: 100%; border-radius: 999px; }
    .cmp-row .val { width: 88px; flex: none; text-align: right; font-weight: 700; color: #e2e8f0; }
    .f-you { background: linear-gradient(90deg, #38bdf8, #818cf8); }
    .f-med { background: linear-gradient(90deg, #475569, #64748b); }
    .f-avg { background: linear-gradient(90deg, #7c3aed, #a78bfa); }

    .empty {
        border: 2px dashed #334155; border-radius: 26px; padding: 56px 24px; text-align: center;
        color: #94a3b8; background: rgba(148,163,184,.05);
    }
    .empty .ico { font-size: 2.8rem; margin-bottom: 8px; }
    .empty b { color: #cbd5e1; }

    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, .7); border: 1px solid #334155;
        border-radius: 18px; padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-weight: 800 !important; color: #f1f5f9 !important; font-size: 1.55rem !important; }
    div[data-testid="stMetricDelta"] { color: #7dd3fc !important; }

    [data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(120deg, #4f46e5, #7c3aed) !important;
        color: #fff !important; border: none !important; border-radius: 16px !important;
        font-weight: 700 !important; font-size: 1rem !important;
        padding: 13px 20px !important; width: 100% !important;
        box-shadow: 0 12px 26px -12px rgba(79,70,229,.7) !important;
        transition: transform .12s ease, box-shadow .12s ease !important;
    }
    [data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 30px -12px rgba(79,70,229,.85) !important;
        color: #fff !important; border: none !important;
        filter: brightness(1.12);
    }
    [data-testid="stButton"] > button[kind="primary"]:active { transform: translateY(0) !important; }

    [data-testid="stButton"] > button[kind="secondary"] {
        background: rgba(30, 41, 59, .85) !important; color: #a5b4fc !important;
        border: 1.5px solid rgba(99, 102, 241, .55) !important; border-radius: 999px !important;
        font-weight: 600 !important; font-size: .82rem !important;
        padding: 7px 12px !important; width: 100% !important; box-shadow: none !important;
        transition: background .12s ease, border-color .12s ease !important;
    }
    [data-testid="stButton"] > button[kind="secondary"]:hover {
        background: #27354d !important; border-color: #818cf8 !important; color: #c7d2fe !important;
    }

    label { font-weight: 600 !important; color: #cbd5e1 !important; }

    footer { color: #94a3b8; font-size: .78rem; text-align: center; margin-top: 26px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------ helpers
def ensure_valid(key: str, valid: list):
    """Reset a session_state selection before its widget is instantiated."""
    if valid and st.session_state.get(key) not in valid:
        st.session_state[key] = valid[0]


def detail_options(brand: str, model_name: str, field: str):
    det = OPTS["details"].get(f"{brand}{DETAIL_SEP}{model_name}", {})
    return det.get(field) or OPTS["global"].get(field, [])


# ------------------------------------------------------------ header
st.markdown(
    """
    <div class="hero">
      <h1>🚗 Used Car Price Prediction</h1>
      <p>Choose a car and get an instant, ML-powered market estimate —
         a CatBoost model trained on thousands of real listings.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------ input / result layout
inp, res = st.columns([3, 2], gap="large")

with inp:
    st.markdown('<div class="section-label">⚡ Quick samples</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    for slot, sample in zip([s1, s2, s3], OPTS["samples"]):
        with slot:
            if st.button(
                sample["label"], key=f"sample_{sample['label']}", type="secondary"
            ):
                for k in RAW_FIELDS:
                    st.session_state[k] = sample[k]

    st.markdown('<div class="section-label">🚗 Configure your car</div>', unsafe_allow_html=True)

    # --- cascading selects: brand → model → everything else ---
    brands = OPTS["brands"]
    ensure_valid("brand", brands)
    brand = st.selectbox("Brand", brands, key="brand")

    models = OPTS["models_by_brand"].get(brand, [])
    ensure_valid("model", models)
    model_name = st.selectbox("Model", models, key="model")

    r1a, r1b = st.columns(2)
    with r1a:
        years = [int(y) for y in detail_options(brand, model_name, "years")]
        ensure_valid("model_year", years)
        st.selectbox("Year", years, key="model_year")
    with r1b:
        mileage_max = OPTS["mileage_max"]
        if "milage" not in st.session_state:
            st.session_state["milage"] = 50000
        st.slider(
            "Mileage", 0, mileage_max, int(st.session_state["milage"]),
            step=1000, key="milage", format="%d mi",
        )

    r2a, r2b = st.columns(2)
    with r2a:
        fuels = detail_options(brand, model_name, "fuel_type")
        ensure_valid("fuel_type", fuels)
        st.selectbox("Fuel type", fuels, key="fuel_type")
    with r2b:
        trans = detail_options(brand, model_name, "transmission")
        ensure_valid("transmission", trans)
        st.selectbox("Transmission", trans, key="transmission")

    engines = detail_options(brand, model_name, "engine")
    ensure_valid("engine", engines)
    st.selectbox("Engine", engines, key="engine")

    r3a, r3b = st.columns(2)
    with r3a:
        exts = detail_options(brand, model_name, "ext_col")
        ensure_valid("ext_col", exts)
        st.selectbox("Exterior color", exts, key="ext_col")
    with r3b:
        ints = detail_options(brand, model_name, "int_col")
        ensure_valid("int_col", ints)
        st.selectbox("Interior color", ints, key="int_col")

    r4a, r4b = st.columns(2)
    with r4a:
        acc = detail_options(brand, model_name, "accident")
        ensure_valid("accident", acc)
        st.selectbox("Accident history", acc, key="accident")
    with r4b:
        titles = detail_options(brand, model_name, "clean_title")
        ensure_valid("clean_title", titles)
        st.selectbox("Clean title", titles, key="clean_title")

    if st.button("🔮  Predict Car Price", type="primary", use_container_width=True):
        car = {k: st.session_state[k] for k in RAW_FIELDS}
        try:
            price = predict_price(car)
            st.session_state["result"] = {"car": car, "price": price}
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# ------------------------------------------------------------ result card
with res:
    result = st.session_state.get("result")
    if result is None:
        st.markdown(
            """
            <div class="empty" style="margin-top:34px">
              <div class="ico">🎯</div>
              <b>No prediction yet</b><br>
              Configure a car on the left and hit <b>Predict Car Price</b><br>
              to see your instant market estimate here.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        car, price = result["car"], result["price"]
        lo, hi = max(0.0, price - MAE), price + MAE
        car_age = max(0, YEAR - int(car["model_year"]))
        hp = _extract(r"(\d+(?:\.\d+)?)HP", car["engine"]) or MEDIANS["horsepower"]
        liters = _extract(r"(\d+(?:\.\d+)?)L", car["engine"]) or MEDIANS["engine_liters"]
        pct = percentile_of(price)
        med = STATS.get("median", price)
        mean = STATS.get("mean", price)
        top = max(price, med, mean) or 1.0

        def w(v):
            return max(4.0, 100.0 * v / top)

        st.markdown(
            f"""
            <div class="price-card">
              <div class="cap">Estimated market price</div>
              <div class="big">{money(price)}</div>
              <div class="range">Likely range {money(lo)} – {money(hi)} &nbsp;·&nbsp; ± {money(MAE)} avg. error</div>
              <div class="note">Pricier than {pct:.0f}% of listings in the dataset</div>
              <div>
                <span class="chip">🗓 {car_age} yr old · {car['model_year']}</span>
                <span class="chip">🐎 {hp:,.0f} HP</span>
                <span class="chip">⚙ {liters:g}L</span>
                <span class="chip">{int(car['milage']):,} mi</span>
                <span class="chip alt">{car['fuel_type']}</span>
                <span class="chip alt">{car['transmission']}</span>
              </div>
              <div class="cmp">
                <div class="cmp-row"><span class="lbl">This car</span>
                  <span class="track"><span class="fill f-you" style="width:{w(price)}%"></span></span>
                  <span class="val">{money(price)}</span></div>
                <div class="cmp-row"><span class="lbl">Market median</span>
                  <span class="track"><span class="fill f-med" style="width:{w(med)}%"></span></span>
                  <span class="val">{money(med)}</span></div>
                <div class="cmp-row"><span class="lbl">Market average</span>
                  <span class="track"><span class="fill f-avg" style="width:{w(mean)}%"></span></span>
                  <span class="val">{money(mean)}</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"Estimate for a **{car['brand']} {car['model']} ({car['model_year']})** "
            f"with **{int(car['milage']):,} mi** — same feature engineering as the training notebook."
        )

# ------------------------------------------------------------ dataset context
st.markdown('<div class="section-label">📊 Dataset &amp; model context</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Market median", money(STATS.get("median", 0)))
c2.metric("Market average", money(STATS.get("mean", 0)))
c3.metric(
    "Middle 50%",
    f"{money(STATS.get('q25', 0))} – {money(STATS.get('q75', 0))}",
)
c4.metric("Cars trained on", f"{META.get('rows_trained', 0):,}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Model R²", f"{METRICS.get('r2', 0):.3f}")
c6.metric("Avg. error (MAE)", money(MAE))
c7.metric("RMSE", money(METRICS.get("rmse", 0)))
c8.metric("Features", str(len(MODEL_COLUMNS)))

st.markdown(
    "<footer>Built with Streamlit · Model: CatBoost Regressor · "
    "Pipeline: cleaning → IQR outlier filter → engineered specs → 16 predictors · "
    "USD converted at ₹96.38/USD (Oct 2, 2026); estimates reflect US listings, not Indian market values</footer>",
    unsafe_allow_html=True,
)
