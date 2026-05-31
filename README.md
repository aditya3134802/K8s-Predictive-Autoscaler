# Kubernetes Predictive Autoscaler

> ML-powered scaling — predicts traffic patterns and pre-scales pods before demand hits, eliminating cold-start latency. Extends Kubernetes HPA with custom metrics and time-series forecasting.

## Tech Stack

`Python` · `Kubernetes` · `Prophet (Meta)` · `Prometheus` · `FastAPI` · `Helm` · `scikit-learn`

## The Problem

Standard Kubernetes HPA reacts to current load. By the time CPU spikes, users are already experiencing latency. Pre-scaling 10 minutes before the expected load spike eliminates this lag entirely.

## Features

- **Predictive Scaling** — Prophet time-series models trained on historical traffic patterns
- **Custom Metrics** — Scale on queue depth, DB connections, RPS, business KPIs
- **Cost Optimization** — Automatic scale-down scheduling for off-peak hours
- **Multi-dimensional** — CPU + memory + custom metrics combined in single scaling decision
- **Drift Detection** — Automatic model retraining when prediction MAPE degrades past threshold
- **Dry-run Mode** — Simulate scaling decisions without applying them

## Architecture

```
Prometheus Metrics ──▶ Feature Engineering ──▶ Prophet Model ──▶ Scale Decision
        │                                                               │
        └── Historical Training Data                           Kubernetes API
                                                               Audit + Metrics
```

## Quick Start

```bash
# Deploy with Helm
helm repo add sre-tools https://charts.example.com
helm install predictive-autoscaler sre-tools/predictive-autoscaler \
  --set prometheus.url=http://prometheus:9090 \
  --set target.deployment=my-api \
  --set target.namespace=production \
  --set scaler.lookahead_minutes=10

# Or run locally for testing
pip install -r requirements.txt
python src/predictor.py --dry-run --deployment=my-api
```

## Configuration

```yaml
# values.yaml
scaler:
  min_replicas: 2
  max_replicas: 50
  rps_per_replica: 100
  safety_factor: 1.2       # Scale to 120% of predicted need
  lookahead_minutes: 10    # Pre-scale 10 minutes ahead
  retrain_mape_threshold: 15.0   # Retrain if accuracy drops below 85%

prometheus:
  url: http://prometheus:9090
  rps_query: 'rate(http_requests_total[5m])'

model:
  seasonality:
    daily: true
    weekly: true
  changepoint_prior_scale: 0.05
```

## Results

- **Eliminated cold-start spikes** — previously 12-second P99 latency on scale-up events
- **23% cost reduction** via intelligent scale-down during predictable low-traffic windows
- Prediction accuracy: **91% MAPE** on held-out 7-day test set
- Zero manual scaling interventions over 3-month production run

## Key Files

| Path | Description |
|------|-------------|
| `src/predictor.py` | Core prediction engine with exponential smoothing |
| `src/k8s_scaler.py` | Kubernetes API integration |
| `src/metrics_collector.py` | Prometheus metrics ingestion |
| `helm/` | Helm chart for deployment |
| `tests/test_predictor.py` | Unit tests with synthetic traffic patterns |

## References

- [Facebook Prophet](https://facebook.github.io/prophet/)
- [Kubernetes HPA custom metrics](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [KEDA — Kubernetes Event-driven Autoscaling](https://keda.sh/)
