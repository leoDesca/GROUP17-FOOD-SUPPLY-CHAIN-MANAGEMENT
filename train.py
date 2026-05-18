"""
train.py  —  Group 17 Supply Chain: HDBSCAN Training
======================================================
Builds the cafeteria-day feature matrix, fits HDBSCAN,
and saves the model + scaler to disk.

Key point — why we use the standalone `hdbscan` package
and NOT sklearn's HDBSCAN:
  sklearn's HDBSCAN only has fit_predict() — it cannot
  classify NEW data points after training.
  The standalone `hdbscan` package has approximate_predict(),
  which is exactly what a deployed API needs.

Run once before starting the server:
  once before starting the server   python train.py

Outputs written to model/:
    hdbscan_model.pkl    — fitted HDBSCAN clusterer
    scaler.pkl           — fitted StandardScaler
    cluster_info.json    — cluster names + supply actions
    feature_names.json   — ordered list of 41 feature columns
    metrics.json         — evaluation scores from training
"""

import json
import os

import hdbscan           # pip install hdbscan  (standalone package, NOT sklearn)
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


def _convert_scaler_to_float32(scaler_obj: StandardScaler) -> None:
    """Shrink scaler artifact memory footprint while preserving behavior."""
    for attr in ("mean_", "var_", "scale_"):
        value = getattr(scaler_obj, attr, None)
        if value is not None:
            setattr(scaler_obj, attr, np.asarray(value, dtype=np.float32))

# ── 1. Load cafeteria data ───────────────────────────────────────────────────
DATA_FILE = "makerere_Cafeteria_synthetic.csv"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"'{DATA_FILE}' not found — run the notebook data-generation cell first."
    )

print(f"Loading {DATA_FILE}...")
cafe = pd.read_csv(DATA_FILE)
cafe["Date"] = pd.to_datetime(cafe["Date"])

for col in ["Price_UGX", "Revenue_UGX", "Ingredient_Cost_UGX",
            "Waste_Cost_UGX", "Gross_Profit_UGX"]:
    cafe[col] = pd.to_numeric(
        cafe[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
cafe["Is_Weekend"] = (
    cafe["Is_Weekend"].astype(str).str.lower()
    .map({"true": 1, "false": 0}).fillna(0).astype(int)
)
print(f"  Rows: {len(cafe):,}  |  Cafeterias: {cafe['Cafeteria_ID'].nunique()}")

# ── 2. Feature engineering (identical to notebook FE v2) ────────────────────
print("\nBuilding feature matrix...")

cafe["Sellout_Rate"]     = cafe["Portions_Sold"]       / cafe["Portions_Prepared"].replace(0, np.nan)
cafe["Waste_Rate"]       = cafe["Waste_Portions"]      / cafe["Portions_Prepared"].replace(0, np.nan)
cafe["Profit_Margin"]    = cafe["Gross_Profit_UGX"]    / cafe["Revenue_UGX"].replace(0, np.nan)
cafe["Food_Cost_Ratio"]  = cafe["Ingredient_Cost_UGX"] / cafe["Revenue_UGX"].replace(0, np.nan)
cafe["Waste_Cost_Ratio"] = cafe["Waste_Cost_UGX"]      / cafe["Revenue_UGX"].replace(0, np.nan)
cafe["Rev_per_Portion"]  = cafe["Revenue_UGX"]         / cafe["Portions_Sold"].replace(0, np.nan)

cafe_daily = cafe.groupby(["Date", "Cafeteria_ID"], observed=True).agg(
    Daily_Prepared        = ("Portions_Prepared",    "sum"),
    Daily_Sold            = ("Portions_Sold",        "sum"),
    Daily_Waste           = ("Waste_Portions",       "sum"),
    Daily_Revenue         = ("Revenue_UGX",          "sum"),
    Daily_Ingredient_Cost = ("Ingredient_Cost_UGX",  "sum"),
    Daily_Waste_Cost      = ("Waste_Cost_UGX",       "sum"),
    Daily_Profit          = ("Gross_Profit_UGX",     "sum"),
    Avg_Waste_Pct         = ("Waste_Pct",            "mean"),
    Waste_Pct_Std         = ("Waste_Pct",            "std"),
).reset_index()

cafe_daily["Daily_Sellout_Rate"]    = cafe_daily["Daily_Sold"]    / cafe_daily["Daily_Prepared"].replace(0, np.nan)
cafe_daily["Daily_Waste_Rate"]      = cafe_daily["Daily_Waste"]   / cafe_daily["Daily_Prepared"].replace(0, np.nan)
cafe_daily["Daily_Profit_Margin"]   = cafe_daily["Daily_Profit"]  / cafe_daily["Daily_Revenue"].replace(0, np.nan)
cafe_daily["Rev_per_Prepared"]      = cafe_daily["Daily_Revenue"] / cafe_daily["Daily_Prepared"].replace(0, np.nan)
cafe_daily["Ingredient_Efficiency"] = cafe_daily["Daily_Profit"]  / cafe_daily["Daily_Ingredient_Cost"].replace(0, np.nan)
cafe_daily["Waste_Cost_Share"]      = cafe_daily["Daily_Waste_Cost"] / cafe_daily["Daily_Revenue"].replace(0, np.nan)

ctx = cafe[["Date", "Cafeteria_ID", "Academic_Period", "Is_Weekend"]].drop_duplicates()
cafe_daily = cafe_daily.merge(ctx, on=["Date", "Cafeteria_ID"], how="left")

# Per-meal sold pivot
sold_piv = cafe.pivot_table(
    index=["Date", "Cafeteria_ID"], columns="Meal",
    values="Portions_Sold", aggfunc="sum"
).reset_index().fillna(0)
sold_piv.columns.name = None
sold_piv.rename(columns={
    "Posho & Beans":            "Sold_Posho_Beans",
    "Matooke & Groundnut Stew": "Sold_Matooke_Stew",
    "Rice & Chicken":           "Sold_Rice_Chicken",
    "Katogo (Offal+Matoke)":    "Sold_Katogo",
    "Chips & Eggs":             "Sold_Chips_Eggs",
    "Rolex":                    "Sold_Rolex",
}, inplace=True)
cafe_daily = cafe_daily.merge(sold_piv, on=["Date", "Cafeteria_ID"], how="left")

# Per-meal waste pivot
waste_piv = cafe.pivot_table(
    index=["Date", "Cafeteria_ID"], columns="Meal",
    values="Waste_Pct", aggfunc="mean"
).reset_index().fillna(0)
waste_piv.columns.name = None
waste_piv.rename(columns={
    "Posho & Beans":            "WastePct_Posho_Beans",
    "Matooke & Groundnut Stew": "WastePct_Matooke_Stew",
    "Rice & Chicken":           "WastePct_Rice_Chicken",
    "Katogo (Offal+Matoke)":    "WastePct_Katogo",
    "Chips & Eggs":             "WastePct_Chips_Eggs",
    "Rolex":                    "WastePct_Rolex",
}, inplace=True)
cafe_daily = cafe_daily.merge(waste_piv, on=["Date", "Cafeteria_ID"], how="left")

sold_cols = ["Sold_Posho_Beans","Sold_Matooke_Stew","Sold_Rice_Chicken",
             "Sold_Katogo","Sold_Chips_Eggs","Sold_Rolex"]
row_tot = cafe_daily[sold_cols].sum(axis=1).replace(0, np.nan)
shares  = cafe_daily[sold_cols].div(row_tot, axis=0).fillna(0)
cafe_daily["Meal_Entropy"]   = -(shares * np.log(shares.replace(0, np.nan))).sum(axis=1).fillna(0)
cafe_daily["Top_Meal_Share"] = cafe_daily[sold_cols].max(axis=1) / row_tot.fillna(1)

cafe_daily["Month"]        = cafe_daily["Date"].dt.month
cafe_daily["DayOfWeekNum"] = cafe_daily["Date"].dt.dayofweek
cafe_daily["Month_sin"]    = np.sin(2 * np.pi * (cafe_daily["Month"] - 1) / 12)
cafe_daily["Month_cos"]    = np.cos(2 * np.pi * (cafe_daily["Month"] - 1) / 12)
cafe_daily["DOW_sin"]      = np.sin(2 * np.pi * cafe_daily["DayOfWeekNum"] / 7)
cafe_daily["DOW_cos"]      = np.cos(2 * np.pi * cafe_daily["DayOfWeekNum"] / 7)

PR = {"Sem1_Teaching":1,"Sem1_Exams":2,"Sem_Break":3,
      "Sem2_Teaching":4,"Sem2_Exams":5,"Xmas_Break":6,"Other":0}
cafe_daily["Period_Code"]    = cafe_daily["Academic_Period"].map(PR).fillna(0).astype(int)
cafe_daily["Is_Exam_Period"] = cafe_daily["Academic_Period"].isin(["Sem1_Exams","Sem2_Exams"]).astype(int)
cafe_daily["Is_Break"]       = cafe_daily["Academic_Period"].isin(["Sem_Break","Xmas_Break"]).astype(int)

cafe_daily = cafe_daily.sort_values(["Cafeteria_ID","Date"])
grp = cafe_daily.groupby("Cafeteria_ID", observed=True)
cafe_daily["Sold_Roll7_Mean"]    = grp["Daily_Sold"].transform(lambda s: s.rolling(7,min_periods=2).mean())
cafe_daily["Sold_Roll7_Std"]     = grp["Daily_Sold"].transform(lambda s: s.rolling(7,min_periods=2).std())
cafe_daily["Waste_Roll7_Mean"]   = grp["Daily_Waste_Rate"].transform(lambda s: s.rolling(7,min_periods=2).mean())
cafe_daily["Profit_Roll7_Mean"]  = grp["Daily_Profit_Margin"].transform(lambda s: s.rolling(7,min_periods=2).mean())
cafe_daily["Revenue_Roll7_Mean"] = grp["Daily_Revenue"].transform(lambda s: s.rolling(7,min_periods=2).mean())
cafe_daily["Demand_Z"]           = grp["Daily_Sold"].transform(lambda s: (s - s.mean()) / s.std())
cafe_daily["Revenue_vs_Roll7"]   = (
    (cafe_daily["Daily_Revenue"] - cafe_daily["Revenue_Roll7_Mean"])
    / cafe_daily["Revenue_Roll7_Mean"].replace(0, np.nan)
)

# Simplified feature set: only features computable from UI inputs (6 fields)
# Can derive: daily_prepared, daily_sold, daily_waste, daily_revenue, daily_profit
#            + dow, month → DOW_sin, DOW_cos, Month_sin, Month_cos, Is_Weekend (via dow)
FEATURE_NAMES = [
    "Daily_Prepared",
    "Daily_Sold",
    "Daily_Waste",
    "Daily_Revenue",
    "Daily_Profit",
    "Daily_Sellout_Rate",
    "Daily_Waste_Rate",
    "Daily_Profit_Margin",
    "DOW_sin",
    "DOW_cos",
    "Month_sin",
    "Month_cos",
    "Is_Weekend",
]

avail = [f for f in FEATURE_NAMES if f in cafe_daily.columns]
X_df  = cafe_daily[avail].replace([np.inf,-np.inf], np.nan).dropna(
    subset=["Daily_Prepared", "Daily_Sold"]
)
X_raw = np.asarray(X_df.fillna(0).values, dtype=np.float32)
print(f"  Feature matrix: {X_raw.shape[0]:,} cafeteria-days × {X_raw.shape[1]} features")

# ── 3. Scale ─────────────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = np.asarray(scaler.fit_transform(X_raw), dtype=np.float32)
_convert_scaler_to_float32(scaler)

# ── 4. Fit HDBSCAN (standalone package — has approximate_predict) ─────────────
# IMPORTANT: we use prediction_data=True so approximate_predict works later
print("\nFitting HDBSCAN (min_cluster_size=20, min_samples=20)...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,
    min_samples=20,
    prediction_data=True,   # ← required for approximate_predict at inference time
)
labels = clusterer.fit_predict(X_scaled)

n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
noise_count = int((labels == -1).sum())
noise_pct   = noise_count / len(labels) * 100
non_noise   = labels != -1

sil = silhouette_score(X_scaled[non_noise], labels[non_noise])
dbi = davies_bouldin_score(X_scaled[non_noise], labels[non_noise])
chi = calinski_harabasz_score(X_scaled[non_noise], labels[non_noise])

print(f"  Clusters found       : {n_clusters}")
print(f"  Noise points         : {noise_count:,}  ({noise_pct:.1f}%)")
print(f"  Silhouette Score     : {sil:.4f}  (from notebook: 0.5043)")
print(f"  Davies-Bouldin Index : {dbi:.4f}  (from notebook: 0.6632)")
print(f"  Calinski-Harabasz    : {chi:.1f}  (from notebook: 1486.5)")

# ── 5. Cluster definitions (from notebook interpretation) ─────────────────────
valid_rows = cafe_daily.loc[X_df.index].copy()
valid_rows["HDBSCAN_Cluster"] = labels

cluster_info = {}
for c in sorted(set(labels)):
    sub  = valid_rows[valid_rows["HDBSCAN_Cluster"] == c]
    key  = str(c)

    if c == -1:
        cluster_info[key] = {
            "name":          "Anomalous / Atypical Day",
            "supply_alert":  "REVIEW",
            "avg_daily_sold": 0,
            "supply_action": (
                "This day does not match any known operational pattern. "
                "It is likely a break-period day, a public holiday, or an "
                "unusually low-traffic day. Flag for manual procurement review. "
                "Default to minimum stock until the cause is confirmed."
            ),
            "size": int(len(sub)),
        }
    else:
        avg_sold   = float(sub["Daily_Sold"].mean())
        avg_waste  = float(sub["Daily_Waste_Rate"].mean())
        avg_profit = float(sub["Daily_Profit_Margin"].mean())
        pct_break  = float(sub["Is_Break"].mean())
        pct_exam   = float(sub["Is_Exam_Period"].mean())

        if avg_sold > 1000:
            name   = "High-Volume Demand Days"
            alert  = "CRITICAL"
            action = (
                "Demand is significantly above the campus baseline. "
                "Place ingredient orders at least 2 days in advance. "
                "Start service with full menu availability and assign extra staff. "
                "Track sellout every 2 hours and replenish quickly where needed."
            )
        else:
            name   = "Standard Operational Days"
            alert  = "NORMAL"
            action = (
                "Normal procurement using 7-day rolling average as the base. "
                "Adjust up by 10% for teaching days, down by 55% for weekends. "
                "Reduce by 60% during semester breaks. "
                "Check waste percentage daily — if above 15%, reduce next order by 10%."
            )

        cluster_info[key] = {
            "name":               name,
            "supply_alert":       alert,
            "avg_daily_sold":     round(avg_sold, 1),
            "avg_waste_rate_pct": round(avg_waste * 100, 2),
            "avg_profit_margin":  round(avg_profit * 100, 2),
            "pct_break_days":     round(pct_break, 3),
            "pct_exam_days":      round(pct_exam, 3),
            "supply_action":      action,
            "size":               int(len(sub)),
        }

    print(f"\n  Cluster {c}: {cluster_info[key]['name']}  ({len(sub):,} days)")

# ── 6. Save everything ───────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

joblib.dump(clusterer, "model/hdbscan_model.pkl", compress=3)
joblib.dump(scaler,    "model/scaler.pkl", compress=3)

with open("model/cluster_info.json", "w") as f:
    json.dump(cluster_info, f, indent=2)

with open("model/feature_names.json", "w") as f:
    json.dump(avail, f, indent=2)

with open("model/metrics.json", "w") as f:
    model_bytes = os.path.getsize("model/hdbscan_model.pkl")
    scaler_bytes = os.path.getsize("model/scaler.pkl")
    json.dump({
        "model":                "HDBSCAN (standalone hdbscan package)",
        "sklearn_hdbscan_note": "sklearn HDBSCAN has no .predict() — we use the standalone hdbscan package which has approximate_predict()",
        "min_cluster_size":     20,
        "min_samples":          20,
        "prediction_data":      True,
        "n_clusters":           n_clusters,
        "noise_count":          noise_count,
        "noise_pct":            round(noise_pct, 2),
        "silhouette":           round(sil, 4),
        "davies_bouldin":       round(dbi, 4),
        "calinski_harabasz":    round(chi, 1),
        "n_samples":            int(X_raw.shape[0]),
        "n_features":           int(X_raw.shape[1]),
        "compression": {
            "joblib_compress_level": 3,
            "numeric_dtype": "float32",
            "model_file_bytes": int(model_bytes),
            "scaler_file_bytes": int(scaler_bytes),
        },
    }, f, indent=2)

print("\n\nSaved:")
print("  model/hdbscan_model.pkl")
print("  model/scaler.pkl")
print("  model/cluster_info.json")
print("  model/feature_names.json")
print("  model/metrics.json")
print(f"  compressed model size  : {model_bytes / 1024:.1f} KB")
print(f"  compressed scaler size : {scaler_bytes / 1024:.1f} KB")
print("\nTraining complete. Now run:")
print("  gunicorn app:application --bind 0.0.0.0:8000 --workers 2")
