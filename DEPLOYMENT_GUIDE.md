# Group 17 — Deployment Guide
## HDBSCAN Supply Chain Prediction API

---

## Deployment challenge and approach

sklearn's `HDBSCAN` only exposes `fit_predict()` and doesn't provide a way to classify new points after training. That makes live inference harder when the service receives new cafeteria-day records.

How we addressed this: use the standalone `hdbscan` package (`pip install hdbscan`), which provides:

```python
labels, strengths = hdbscan.approximate_predict(clusterer, X_new)
```

This requires training with `prediction_data=True` so the model keeps the data needed for membership inference:

```python
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,
    min_samples=20,
    prediction_data=True,   # ← this line makes live inference possible
)
```

After that, deployment is straightforward: scale incoming records with the saved `StandardScaler`, call `approximate_predict`, and return the cluster id and membership strength.

---

## Model results (from notebook)

| Metric | Score | Benchmark |
|---|---|---|
| Silhouette Score | **0.5043** | > 0.5 = strong |
| Davies-Bouldin Index | **0.6632** | < 1.0 = well separated |
| Calinski-Harabasz | **1,486.5** | > 1000 = dense clusters |
| Clusters found | **2** | |
| Noise points | **2,513 (25.7%)** | anomalous/break days |

## Cluster meanings

| Cluster | Name | What it means | Supply action |
|---|---|---|---|
| 0 | Standard Operational Days | All 13 regular cafeterias, all periods | Normal rolling-average procurement |
| 1 | High-Volume Hub Days | Africa Hall (C13) only, teaching/exam days | 5× volume — order 2 days in advance |
| -1 | Noise / Anomalous | Break days, irregular days | Minimum stock, manual review |

---

## Stack

```
Client  →  Gunicorn (WSGI server)  →  app.py (pure Python WSGI)
                                             ↓
                              hdbscan.approximate_predict()
                                             ↓
                              model/hdbscan_model.pkl + scaler.pkl
```

We implement the WSGI `application(environ, start_response)` callable directly to keep the runtime minimal and avoid extra framework layers.

---

## Option A — Run locally

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
#    NOTE: hdbscan compiles Cython extensions — needs gcc (Linux/Mac)
#    On Windows: install Microsoft C++ Build Tools first
pip install -r requirements.txt

# 3. Train and save the model
python train.py
# 3b. (Optional) Run the compression pipeline to produce a compressed production model
python test2_model_compression.py

# 4. Start the server
gunicorn app:application --bind 0.0.0.0:8000 --workers 2

# 5. In a second terminal, test all endpoints
python test_api.py
```

---

## Option B — Docker (recommended)

```bash
# Build (runs train.py and the compression pipeline inside the container — model and compressed bundle are baked in)
docker build -t makerere-api .

# Run
docker run -p 8000:8000 makerere-api

# Test
python test_api.py

# Stop
docker ps                       # get container ID
docker stop <container_id>
```

---

## Option C — Render (public live URL)

Use Render if you want a shareable live web address without managing a VM.

1. Push this repository to GitHub.
2. In Render, choose **New +** then **Blueprint**.
3. Connect the GitHub repo and let Render read [render.yaml](render.yaml).
4. Deploy the `group17-food-supply-chain-management` web service.
5. Render will build the Docker image, train the model during build, and expose the app on a public `https://...onrender.com` URL.

Once deployed, the main app is available at `/`, and the health check is at `/health`.

---

## Manual curl tests

```bash
# Health check
curl http://localhost:8000/health

# Cluster definitions
curl http://localhost:8000/clusters

# Predict — Africa Hall on a teaching day
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "daily_prepared": 2200, "daily_sold": 2067, "daily_waste": 133,
    "daily_revenue": 8268000, "daily_profit": 3180000,
    "daily_sellout_rate": 0.94, "daily_waste_rate": 0.06,
    "daily_profit_margin": 0.38, "rev_per_prepared": 3758,
    "ingredient_efficiency": 1.42, "waste_cost_share": 0.07,
    "avg_waste_pct": 6.0, "waste_pct_std": 1.2,
    "sold_posho_beans": 415, "sold_matooke_stew": 330,
    "sold_rice_chicken": 280, "sold_katogo": 210,
    "sold_chips_eggs": 250, "sold_rolex": 582,
    "meal_entropy": 1.76, "top_meal_share": 0.28,
    "wastepct_posho_beans": 6.1, "wastepct_matooke_stew": 5.8,
    "wastepct_rice_chicken": 6.4, "wastepct_katogo": 6.2,
    "wastepct_chips_eggs": 5.9, "wastepct_rolex": 5.7,
    "dow": 1, "month": 3,
    "is_weekend": 0, "period_code": 1, "is_exam_period": 0, "is_break": 0,
    "sold_roll7_mean": 2010, "sold_roll7_std": 120,
    "waste_roll7_mean": 0.061, "profit_roll7_mean": 0.375,
    "demand_z": 0.8, "revenue_vs_roll7": 0.05
  }'
```

Expected response:
```json
{
  "cluster": 1,
  "is_noise": false,
  "cluster_name": "High-Volume Hub Days (Africa Hall)",
  "supply_alert": "CRITICAL",
  "supply_action": "Africa Hall cafeteria only. Volume is ~5x the campus average...",
  "membership_strength": 0.87,
  "confidence": "high"
}
```

---

## Notes for assessment

Q: Why not use Flask or Django?
We kept the stack minimal and implemented the WSGI entry point directly. That avoids adding an extra dependency while still providing the standard `application(environ, start_response)` interface for WSGI servers.

Q: How do you get predictions if sklearn's HDBSCAN has no `predict`?
We use the `hdbscan` package's `approximate_predict(clusterer, X_new)`. Training with `prediction_data=True` ensures the model holds the information needed for membership inference; the function returns a cluster label and a membership strength.

Q: What operational meaning does a noise label (-1) have?
A noise label indicates the day does not match the learned operational patterns — often break days or very small cafeterias. The API marks these for manual review and uses conservative procurement recommendations.

---

## File structure

```
deployment/
├── train.py                         — fit HDBSCAN, save to model/
├── app.py                           — pure Python WSGI (no framework)
├── Dockerfile                       — build and run the container
├── requirements.txt                 — 6 packages
├── test_api.py                      — validates all endpoints
├── DEPLOYMENT_GUIDE.md              — this file
├── makerere_Cafeteria_synthetic.csv — training data
└── model/                           — created by train.py
    ├── hdbscan_model.pkl
    ├── scaler.pkl
    ├── cluster_info.json
    ├── feature_names.json
    └── metrics.json
```
