"""
Test 2: Model Compression Pipeline
---------------------------------
Implements three compression techniques for the deployed cluster predictor:
1) Knowledge distillation (teacher -> student)
2) Cost-complexity pruning
3) Post-training quantization (parameter rounding)

Outputs:
- model/compression_results.json
- model/distilled_student.pkl
- model/pruned_student.pkl
- model/quantized_student.pkl
- model/compressed_model_bundle.pkl (selected production model)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import hdbscan
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_FILE = BASE_DIR / "makerere_Cafeteria_synthetic.csv"

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


def build_feature_matrix() -> np.ndarray:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    cafe = pd.read_csv(DATA_FILE)
    cafe["Date"] = pd.to_datetime(cafe["Date"], errors="coerce")

    for col in ["Price_UGX", "Revenue_UGX", "Ingredient_Cost_UGX", "Waste_Cost_UGX", "Gross_Profit_UGX"]:
        cafe[col] = pd.to_numeric(cafe[col].astype(str).str.replace(",", "", regex=False), errors="coerce")

    cafe["Is_Weekend"] = (
        cafe["Is_Weekend"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)
    )

    cafe_daily = cafe.groupby(["Date", "Cafeteria_ID"], observed=True).agg(
        Daily_Prepared=("Portions_Prepared", "sum"),
        Daily_Sold=("Portions_Sold", "sum"),
        Daily_Waste=("Waste_Portions", "sum"),
        Daily_Revenue=("Revenue_UGX", "sum"),
        Daily_Profit=("Gross_Profit_UGX", "sum"),
    ).reset_index()

    cafe_daily["Daily_Sellout_Rate"] = cafe_daily["Daily_Sold"] / cafe_daily["Daily_Prepared"].replace(0, np.nan)
    cafe_daily["Daily_Waste_Rate"] = cafe_daily["Daily_Waste"] / cafe_daily["Daily_Prepared"].replace(0, np.nan)
    cafe_daily["Daily_Profit_Margin"] = cafe_daily["Daily_Profit"] / cafe_daily["Daily_Revenue"].replace(0, np.nan)

    ctx = cafe[["Date", "Cafeteria_ID", "Is_Weekend"]].drop_duplicates()
    cafe_daily = cafe_daily.merge(ctx, on=["Date", "Cafeteria_ID"], how="left")

    cafe_daily["Month"] = cafe_daily["Date"].dt.month
    cafe_daily["DayOfWeekNum"] = cafe_daily["Date"].dt.dayofweek
    cafe_daily["Month_sin"] = np.sin(2 * np.pi * (cafe_daily["Month"] - 1) / 12)
    cafe_daily["Month_cos"] = np.cos(2 * np.pi * (cafe_daily["Month"] - 1) / 12)
    cafe_daily["DOW_sin"] = np.sin(2 * np.pi * cafe_daily["DayOfWeekNum"] / 7)
    cafe_daily["DOW_cos"] = np.cos(2 * np.pi * cafe_daily["DayOfWeekNum"] / 7)

    x_df = cafe_daily[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).dropna()
    return x_df.astype(float).values


def evaluate(model, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    pred = model.predict(x_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
    }


def model_size_kb(path: Path) -> float:
    return round(path.stat().st_size / 1024.0, 3)


def save_model(model, path: Path) -> float:
    joblib.dump(model, path, compress=9)
    return model_size_kb(path)


def pick_best_candidate(results: dict) -> str:
    best_name = None
    best_score = None

    for name, info in results.items():
        f1 = info["metrics"]["macro_f1"]
        size_kb = info["size_kb"]

        # Higher F1 is better, then smaller size.
        score = (f1, -size_kb)
        if best_score is None or score > best_score:
            best_score = score
            best_name = name

    if best_name is None:
        raise RuntimeError("No candidate models available.")
    return best_name


def main() -> None:
    np.random.seed(42)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    hdbscan_model_path = MODEL_DIR / "hdbscan_model.pkl"
    hdbscan_scaler_path = MODEL_DIR / "scaler.pkl"
    if not hdbscan_model_path.exists() or not hdbscan_scaler_path.exists():
        raise FileNotFoundError(
            "Missing baseline model artifacts. Run train.py first to generate hdbscan_model.pkl and scaler.pkl"
        )

    clusterer = joblib.load(hdbscan_model_path)
    scaler = joblib.load(hdbscan_scaler_path)

    x_raw = build_feature_matrix()
    x_hdb = scaler.transform(x_raw)
    labels, _ = hdbscan.approximate_predict(clusterer, x_hdb)
    y = labels.astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x_raw,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    teacher = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    teacher.fit(x_train, y_train)
    teacher_metrics = evaluate(teacher, x_test, y_test)
    print("Teacher metrics:", teacher_metrics)

    # Technique 1: Knowledge distillation
    pseudo_train = teacher.predict(x_train)
    distilled = DecisionTreeClassifier(max_depth=7, min_samples_leaf=20, random_state=42)
    distilled.fit(x_train, pseudo_train)

    # Technique 2: Pruning (cost complexity pruning on distilled student)
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.25,
        random_state=42,
        stratify=y_train,
    )

    initial_student = DecisionTreeClassifier(max_depth=7, min_samples_leaf=20, random_state=42)
    initial_student.fit(x_fit, teacher.predict(x_fit))

    path = initial_student.cost_complexity_pruning_path(x_fit, teacher.predict(x_fit))
    candidate_alphas = np.unique(np.round(path.ccp_alphas, 6))

    best_alpha = 0.0
    best_f1 = -1.0
    for alpha in candidate_alphas:
        candidate = DecisionTreeClassifier(
            max_depth=7,
            min_samples_leaf=20,
            ccp_alpha=float(alpha),
            random_state=42,
        )
        candidate.fit(x_fit, teacher.predict(x_fit))
        pred_val = candidate.predict(x_val)
        score = f1_score(y_val, pred_val, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_alpha = float(alpha)

    pruned = DecisionTreeClassifier(
        max_depth=7,
        min_samples_leaf=20,
        ccp_alpha=best_alpha,
        random_state=42,
    )
    pruned.fit(x_train, teacher.predict(x_train))

    # Technique 3: Quantization (round tree parameters to reduced precision)
    quantized = copy.deepcopy(pruned)
    tree = quantized.tree_
    tree.threshold[:] = np.round(tree.threshold.astype(np.float32), 3)
    tree.value[:] = np.round(tree.value.astype(np.float32), 3)

    distilled_path = MODEL_DIR / "distilled_student.pkl"
    pruned_path = MODEL_DIR / "pruned_student.pkl"
    quantized_path = MODEL_DIR / "quantized_student.pkl"

    distilled_size = save_model(distilled, distilled_path)
    pruned_size = save_model(pruned, pruned_path)
    quantized_size = save_model(quantized, quantized_path)

    results = {
        "distillation": {
            "artifact": distilled_path.name,
            "size_kb": distilled_size,
            "metrics": evaluate(distilled, x_test, y_test),
            "description": "Student tree trained on teacher pseudo-labels.",
        },
        "pruning": {
            "artifact": pruned_path.name,
            "size_kb": pruned_size,
            "metrics": evaluate(pruned, x_test, y_test),
            "description": f"Cost-complexity pruned student (ccp_alpha={best_alpha:.6f}).",
        },
        "quantization": {
            "artifact": quantized_path.name,
            "size_kb": quantized_size,
            "metrics": evaluate(quantized, x_test, y_test),
            "description": "Pruned student with rounded tree thresholds/leaf values.",
        },
        "teacher_reference": {
            "model": "RandomForestClassifier",
            "metrics": teacher_metrics,
        },
        "target_label_source": "hdbscan.approximate_predict",
    }

    winner = pick_best_candidate({k: v for k, v in results.items() if k in {"distillation", "pruning", "quantization"}})
    winner_model = {
        "distillation": distilled,
        "pruning": pruned,
        "quantization": quantized,
    }[winner]

    bundle = {
        "model_type": "compressed_classifier",
        "winner": winner,
        "feature_names": FEATURE_NAMES,
        "model": winner_model,
        "training_summary": {
            "teacher": teacher_metrics,
            "winner_metrics": results[winner]["metrics"],
            "winner_size_kb": results[winner]["size_kb"],
            "techniques_tested": ["distillation", "pruning", "quantization"],
        },
    }

    bundle_path = MODEL_DIR / "compressed_model_bundle.pkl"
    joblib.dump(bundle, bundle_path, compress=9)
    results["selected_for_production"] = {
        "winner": winner,
        "bundle": bundle_path.name,
        "bundle_size_kb": model_size_kb(bundle_path),
    }

    out_json = MODEL_DIR / "compression_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nTest 2 completed.")
    print(f"Teacher      : Acc={teacher_metrics['accuracy']:.4f}, F1={teacher_metrics['macro_f1']:.4f}")
    for name in ["distillation", "pruning", "quantization"]:
        m = results[name]["metrics"]
        print(f"{name:12s}: Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}, Size={results[name]['size_kb']:.2f} KB")
    print(f"Selected model for production: {winner}")
    print(f"Results file: {out_json}")


if __name__ == "__main__":
    main()
