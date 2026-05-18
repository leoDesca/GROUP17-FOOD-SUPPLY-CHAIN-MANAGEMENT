from pathlib import Path
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional

import hdbscan
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from preprocess import preprocess_row

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

app = FastAPI(title="Group 17 Food Supply Chain Management API")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

clusterer = joblib.load(MODEL_DIR / "hdbscan_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")

compressed_bundle = None
compressed_model = None
compressed_feature_names = []
compressed_winner = None

compressed_bundle_path = MODEL_DIR / "compressed_model_bundle.pkl"
if compressed_bundle_path.exists():
    try:
        compressed_bundle = joblib.load(compressed_bundle_path)
        compressed_model = compressed_bundle.get("model")
        compressed_feature_names = compressed_bundle.get("feature_names", [])
        compressed_winner = compressed_bundle.get("winner")
    except Exception:
        compressed_bundle = None
        compressed_model = None
        compressed_feature_names = []
        compressed_winner = None

with open(MODEL_DIR / "cluster_info.json", "r", encoding="utf-8") as f:
    CLUSTER_INFO = json.load(f)


class SupplierIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    contact: str = Field(min_length=3, max_length=120)
    location: str = Field(min_length=2, max_length=120)
    lead_time_days: int = Field(ge=1, le=60)
    reliability_score: float = Field(ge=0.0, le=1.0)

    @field_validator("contact", mode="before")
    @classmethod
    def normalize_contact(cls, value):
        if value is None:
            return value
        return str(value).strip()


class Supplier(SupplierIn):
    id: int
    created_at: str


class InventoryIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    category: str = Field(min_length=2, max_length=80)
    unit: str = Field(min_length=1, max_length=20)
    current_stock: float = Field(ge=0)
    reorder_level: float = Field(ge=0)
    unit_cost: float = Field(ge=0)
    supplier_id: Optional[int] = None


class InventoryItem(InventoryIn):
    id: int
    last_updated: str


class OrderLine(BaseModel):
    inventory_id: int
    quantity: float = Field(gt=0)


class PurchaseOrderIn(BaseModel):
    supplier_id: int
    expected_delivery: str
    items: List[OrderLine] = Field(min_length=1)


class PurchaseOrder(BaseModel):
    id: int
    supplier_id: int
    expected_delivery: str
    created_at: str
    status: Literal["pending", "approved", "in_transit", "received", "cancelled"]
    items: List[OrderLine]
    total_cost: float


class OrderStatusUpdate(BaseModel):
    status: Literal["pending", "approved", "in_transit", "received", "cancelled"]


class PredictInput(BaseModel):
    daily_prepared: float
    daily_sold: float
    daily_revenue: float
    daily_profit: float
    dow: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)


suppliers: Dict[int, Supplier] = {}
inventory: Dict[int, InventoryItem] = {}
orders: Dict[int, PurchaseOrder] = {}

supplier_seq = 0
inventory_seq = 0
order_seq = 0


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _require_supplier(supplier_id: int) -> Supplier:
    supplier = suppliers.get(supplier_id)
    if not supplier:
        raise HTTPException(status_code=404, detail=f"Supplier {supplier_id} not found")
    return supplier


def _require_inventory(item_id: int) -> InventoryItem:
    item = inventory.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Inventory item {item_id} not found")
    return item


def _calculate_order_total(items: List[OrderLine]) -> float:
    total = 0.0
    for line in items:
        item = _require_inventory(line.inventory_id)
        total += line.quantity * item.unit_cost
    return round(total, 2)


def _seed_data() -> None:
    global supplier_seq, inventory_seq, order_seq
    if suppliers or inventory or orders:
        return

    seed_suppliers = [
        Supplier(
            id=1,
            name="Kampala Fresh Produce Ltd",
            contact="+256-700-111000",
            location="Kampala",
            lead_time_days=2,
            reliability_score=0.92,
            created_at=_utc_now(),
        ),
        Supplier(
            id=2,
            name="Mukono Grains Cooperative",
            contact="+256-702-456789",
            location="Mukono",
            lead_time_days=3,
            reliability_score=0.86,
            created_at=_utc_now(),
        ),
    ]
    for supplier in seed_suppliers:
        suppliers[supplier.id] = supplier

    seed_inventory = [
        InventoryItem(
            id=1,
            name="Rice",
            category="Staples",
            unit="kg",
            current_stock=180,
            reorder_level=120,
            unit_cost=4500,
            supplier_id=2,
            last_updated=_utc_now(),
        ),
        InventoryItem(
            id=2,
            name="Chicken",
            category="Protein",
            unit="kg",
            current_stock=65,
            reorder_level=80,
            unit_cost=13500,
            supplier_id=1,
            last_updated=_utc_now(),
        ),
        InventoryItem(
            id=3,
            name="Cooking Oil",
            category="Kitchen",
            unit="liters",
            current_stock=45,
            reorder_level=50,
            unit_cost=8200,
            supplier_id=1,
            last_updated=_utc_now(),
        ),
    ]
    for item in seed_inventory:
        inventory[item.id] = item

    sample_order = PurchaseOrder(
        id=1,
        supplier_id=1,
        expected_delivery=(datetime.utcnow() + timedelta(days=2)).date().isoformat(),
        created_at=_utc_now(),
        status="pending",
        items=[OrderLine(inventory_id=2, quantity=25), OrderLine(inventory_id=3, quantity=30)],
        total_cost=round((25 * 13500) + (30 * 8200), 2),
    )
    orders[sample_order.id] = sample_order

    supplier_seq = len(suppliers)
    inventory_seq = len(inventory)
    order_seq = len(orders)


_seed_data()

# Compatibility alias for gunicorn app:application if needed
application = app


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )

def _normalize_keys(payload: dict) -> dict:
    key_map = {
        "daily_prepared": "Daily_Prepared",
        "daily_sold": "Daily_Sold",
        "daily_waste": "Daily_Waste",
        "daily_revenue": "Daily_Revenue",
        "daily_profit": "Daily_Profit",
        "daily_sellout_rate": "Daily_Sellout_Rate",
        "daily_waste_rate": "Daily_Waste_Rate",
        "daily_profit_margin": "Daily_Profit_Margin",
        "rev_per_prepared": "Rev_per_Prepared",
        "ingredient_efficiency": "Ingredient_Efficiency",
        "waste_cost_share": "Waste_Cost_Share",
        "avg_waste_pct": "Avg_Waste_Pct",
        "waste_pct_std": "Waste_Pct_Std",
        "sold_posho_beans": "Sold_Posho_Beans",
        "sold_matooke_stew": "Sold_Matooke_Stew",
        "sold_rice_chicken": "Sold_Rice_Chicken",
        "sold_katogo": "Sold_Katogo",
        "sold_chips_eggs": "Sold_Chips_Eggs",
        "sold_rolex": "Sold_Rolex",
        "meal_entropy": "Meal_Entropy",
        "top_meal_share": "Top_Meal_Share",
        "wastepct_posho_beans": "WastePct_Posho_Beans",
        "wastepct_matooke_stew": "WastePct_Matooke_Stew",
        "wastepct_rice_chicken": "WastePct_Rice_Chicken",
        "wastepct_katogo": "WastePct_Katogo",
        "wastepct_chips_eggs": "WastePct_Chips_Eggs",
        "wastepct_rolex": "WastePct_Rolex",
        "period_code": "Period_Code",
        "is_weekend": "Is_Weekend",
        "is_exam_period": "Is_Exam_Period",
        "is_break": "Is_Break",
        "sold_roll7_mean": "Sold_Roll7_Mean",
        "sold_roll7_std": "Sold_Roll7_Std",
        "waste_roll7_mean": "Waste_Roll7_Mean",
        "profit_roll7_mean": "Profit_Roll7_Mean",
        "demand_z": "Demand_Z",
        "revenue_vs_roll7": "Revenue_vs_Roll7",
        "dow": "DayOfWeekNum",
        "month": "Month",
    }

    normalized = {}
    for k, v in payload.items():
        normalized[key_map.get(k, k)] = v
    return normalized


def _confidence(strength: float) -> str:
    if strength >= 0.75:
        return "high"
    if strength >= 0.45:
        return "medium"
    return "low"


def _fallback_cluster_by_demand(daily_sold: float) -> Optional[int]:
    candidates = []
    for key, meta in CLUSTER_INFO.items():
        if key == "-1":
            continue
        avg = meta.get("avg_daily_sold")
        if avg is None:
            continue
        try:
            candidates.append((int(key), float(avg)))
        except (TypeError, ValueError):
            continue

    if not candidates:
        return None

    best_cluster, _ = min(candidates, key=lambda item: abs(item[1] - float(daily_sold)))
    return best_cluster


@app.get("/health")
def health():
    model_path = MODEL_DIR / "hdbscan_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"

    return {
        "status": "ok",
        "service": "food-supply-chain-management",
        "timestamp": _utc_now(),
        "model_artifacts": {
            "hdbscan_model_exists": model_path.exists(),
            "scaler_exists": scaler_path.exists(),
            "hdbscan_model_bytes": int(os.path.getsize(model_path)) if model_path.exists() else None,
            "scaler_bytes": int(os.path.getsize(scaler_path)) if scaler_path.exists() else None,
        },
        "entities": {
            "suppliers": len(suppliers),
            "inventory_items": len(inventory),
            "orders": len(orders),
        },
    }


@app.get("/clusters")
def clusters():
    return CLUSTER_INFO


@app.get("/api/dashboard")
def dashboard():
    low_stock_items = [item for item in inventory.values() if item.current_stock <= item.reorder_level]
    pending_orders = [order for order in orders.values() if order.status in {"pending", "approved", "in_transit"}]
    monthly_purchase_cost = round(sum(order.total_cost for order in orders.values()), 2)

    return {
        "kpis": {
            "total_suppliers": len(suppliers),
            "total_inventory_items": len(inventory),
            "low_stock_count": len(low_stock_items),
            "active_orders": len(pending_orders),
            "monthly_purchase_cost": monthly_purchase_cost,
        },
        "low_stock_items": low_stock_items,
        "active_orders": pending_orders,
    }


@app.get("/api/inventory")
def list_inventory():
    return list(inventory.values())


@app.post("/api/inventory")
def create_inventory_item(payload: InventoryIn):
    global inventory_seq
    if payload.supplier_id is not None:
        _require_supplier(payload.supplier_id)

    inventory_seq += 1
    item = InventoryItem(id=inventory_seq, last_updated=_utc_now(), **payload.model_dump())
    inventory[item.id] = item
    return item


@app.put("/api/inventory/{item_id}")
def update_inventory_item(item_id: int, payload: InventoryIn):
    _require_inventory(item_id)
    if payload.supplier_id is not None:
        _require_supplier(payload.supplier_id)

    item = InventoryItem(id=item_id, last_updated=_utc_now(), **payload.model_dump())
    inventory[item_id] = item
    return item


@app.get("/api/inventory/alerts")
def inventory_alerts():
    alerts = []
    for item in inventory.values():
        if item.current_stock <= item.reorder_level:
            alerts.append(
                {
                    "inventory_id": item.id,
                    "name": item.name,
                    "current_stock": item.current_stock,
                    "reorder_level": item.reorder_level,
                    "severity": "critical" if item.current_stock < item.reorder_level * 0.6 else "warning",
                    "message": f"{item.name} is below reorder level. Trigger procurement.",
                }
            )
    return alerts


@app.get("/api/suppliers")
def list_suppliers():
    return list(suppliers.values())


@app.post("/api/suppliers")
def create_supplier(payload: SupplierIn):
    global supplier_seq
    supplier_seq += 1
    supplier = Supplier(id=supplier_seq, created_at=_utc_now(), **payload.model_dump())
    suppliers[supplier.id] = supplier
    return supplier


@app.get("/api/orders")
def list_orders(status: Optional[str] = None):
    values = list(orders.values())
    if status:
        values = [order for order in values if order.status == status]
    return values


@app.post("/api/orders")
def create_order(payload: PurchaseOrderIn):
    global order_seq
    _require_supplier(payload.supplier_id)
    total_cost = _calculate_order_total(payload.items)

    order_seq += 1
    order = PurchaseOrder(
        id=order_seq,
        supplier_id=payload.supplier_id,
        expected_delivery=payload.expected_delivery,
        created_at=_utc_now(),
        status="pending",
        items=payload.items,
        total_cost=total_cost,
    )
    orders[order.id] = order
    return order


@app.patch("/api/orders/{order_id}/status")
def update_order_status(order_id: int, payload: OrderStatusUpdate):
    order = orders.get(order_id)
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

    order.status = payload.status
    if payload.status == "received":
        for line in order.items:
            item = _require_inventory(line.inventory_id)
            item.current_stock = round(item.current_stock + line.quantity, 3)
            item.last_updated = _utc_now()
            inventory[item.id] = item

    orders[order_id] = order
    return order


def _predict_from_payload(payload: dict):
    try:
        normalized = _normalize_keys(payload)
        features = preprocess_row(normalized)
        arr = np.array(features, dtype=float).reshape(1, -1)

        # If Test 2 produced a compressed production model, use it first.
        if compressed_model is not None:
            cluster = int(compressed_model.predict(arr)[0])
            strength = 0.5
            if hasattr(compressed_model, "predict_proba"):
                proba = compressed_model.predict_proba(arr)
                strength = float(np.max(proba[0]))
            fallback_applied = False
            inference_backend = f"compressed:{compressed_winner or 'unknown'}"
        else:
            scaled = scaler.transform(arr)
            labels, strengths = hdbscan.approximate_predict(clusterer, scaled)

            cluster = int(labels[0])
            strength = float(strengths[0])

            fallback_applied = False
            if cluster == -1:
                fallback_cluster = _fallback_cluster_by_demand(normalized.get("Daily_Sold", 0))
                if fallback_cluster is not None:
                    cluster = fallback_cluster
                    strength = max(strength, 0.35)
                    fallback_applied = True
            inference_backend = "hdbscan"

        cluster_meta = CLUSTER_INFO.get(str(cluster), CLUSTER_INFO.get("-1", {}))

        return {
            "cluster": cluster,
            "is_noise": cluster == -1,
            "cluster_name": cluster_meta.get("name", "Unknown"),
            "supply_alert": cluster_meta.get("supply_alert", "REVIEW"),
            "supply_action": cluster_meta.get("supply_action", "Manual review recommended."),
            "membership_strength": round(strength, 4),
            "confidence": _confidence(strength),
            "fallback_applied": fallback_applied,
            "inference_backend": inference_backend,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict")
def predict(cafe_day: dict):
    required = ["daily_prepared", "daily_sold", "daily_revenue", "daily_profit", "dow", "month"]
    missing = [k for k in required if k not in cafe_day]
    if missing:
        raise HTTPException(status_code=400, detail={"missing_fields": missing})

    payload = {}
    for key in required:
        try:
            payload[key] = float(cafe_day[key])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Field '{key}' must be numeric")

    payload["dow"] = int(payload["dow"])
    payload["month"] = int(payload["month"])
    return _predict_from_payload(payload)


@app.post("/api/forecast-procurement")
def forecast_procurement(cafe_day: PredictInput):
    prediction = _predict_from_payload(cafe_day.model_dump())
    alerts = inventory_alerts()

    suggested_actions = []
    for alert in alerts:
        item = inventory[alert["inventory_id"]]
        top_up_qty = max(item.reorder_level * 1.6 - item.current_stock, 0)
        suggested_actions.append(
            {
                "inventory_id": item.id,
                "name": item.name,
                "supplier_id": item.supplier_id,
                "recommended_quantity": round(top_up_qty, 2),
                "estimated_cost": round(top_up_qty * item.unit_cost, 2),
            }
        )

    return {
        "prediction": prediction,
        "inventory_alerts": alerts,
        "suggested_procurement": suggested_actions,
    }