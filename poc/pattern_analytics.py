"""
Pattern Analytics: Clustering and Time-Series Anomaly Detection.

Demonstrates two additional ML techniques from the topic prompt:
- K-Means Clustering: Group claims into risk profiles
- Time-Series Anomaly Detection: Detect billing spikes over time windows
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


SIMULATED_DAILY_CLAIMS = [
    {"day": 1, "count": 42, "avg_amount": 380, "flag_rate": 0.12},
    {"day": 2, "count": 38, "avg_amount": 410, "flag_rate": 0.11},
    {"day": 3, "count": 45, "avg_amount": 365, "flag_rate": 0.14},
    {"day": 4, "count": 40, "avg_amount": 390, "flag_rate": 0.10},
    {"day": 5, "count": 44, "avg_amount": 375, "flag_rate": 0.13},
    {"day": 6, "count": 15, "avg_amount": 320, "flag_rate": 0.05},
    {"day": 7, "count": 12, "avg_amount": 290, "flag_rate": 0.04},
    {"day": 8, "count": 41, "avg_amount": 395, "flag_rate": 0.11},
    {"day": 9, "count": 43, "avg_amount": 400, "flag_rate": 0.12},
    {"day": 10, "count": 39, "avg_amount": 385, "flag_rate": 0.10},
    {"day": 11, "count": 47, "avg_amount": 420, "flag_rate": 0.15},
    {"day": 12, "count": 78, "avg_amount": 890, "flag_rate": 0.45},
    {"day": 13, "count": 82, "avg_amount": 920, "flag_rate": 0.52},
    {"day": 14, "count": 55, "avg_amount": 610, "flag_rate": 0.28},
    {"day": 15, "count": 44, "avg_amount": 405, "flag_rate": 0.14},
    {"day": 16, "count": 42, "avg_amount": 390, "flag_rate": 0.12},
    {"day": 17, "count": 40, "avg_amount": 375, "flag_rate": 0.11},
    {"day": 18, "count": 38, "avg_amount": 360, "flag_rate": 0.10},
    {"day": 19, "count": 14, "avg_amount": 310, "flag_rate": 0.05},
    {"day": 20, "count": 11, "avg_amount": 285, "flag_rate": 0.03},
    {"day": 21, "count": 43, "avg_amount": 395, "flag_rate": 0.12},
    {"day": 22, "count": 41, "avg_amount": 380, "flag_rate": 0.11},
    {"day": 23, "count": 46, "avg_amount": 410, "flag_rate": 0.13},
    {"day": 24, "count": 44, "avg_amount": 400, "flag_rate": 0.12},
    {"day": 25, "count": 48, "avg_amount": 430, "flag_rate": 0.14},
    {"day": 26, "count": 85, "avg_amount": 950, "flag_rate": 0.55},
    {"day": 27, "count": 52, "avg_amount": 580, "flag_rate": 0.24},
    {"day": 28, "count": 42, "avg_amount": 390, "flag_rate": 0.12},
    {"day": 29, "count": 40, "avg_amount": 385, "flag_rate": 0.11},
    {"day": 30, "count": 43, "avg_amount": 395, "flag_rate": 0.13},
]


def cluster_claims(claims_list):
    """Apply K-Means clustering to group claims into risk profiles.

    Uses billing amount, procedure count, and emergency flag to identify
    natural groupings in claims data for risk stratification.
    """
    from rules_engine import run_rules_engine
    from anomaly_detector import get_anomaly_score

    features = []
    risk_scores = []
    for claim in claims_list:
        is_emergency = 1 if claim["provider_type"] in ["Emergency Medicine", "Urgent Care"] else 0
        rules = run_rules_engine(claim)
        anomaly = get_anomaly_score(claim)
        combined_risk = rules["risk_score"] * 0.5 + anomaly["anomaly_score"] * 0.5
        risk_scores.append(combined_risk)
        features.append([
            len(claim["procedures"]),
            claim["billed_amount"],
            is_emergency,
            combined_risk,
        ])

    X = np.array(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(3, len(claims_list))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    cluster_names = ["Low Risk", "Moderate Risk", "High Risk"]
    cluster_risk = []
    for i in range(n_clusters):
        mask = labels == i
        cluster_risk.append(np.mean([risk_scores[j] for j in range(len(claims_list)) if mask[j]]))

    sorted_indices = np.argsort(cluster_risk)
    label_map = {sorted_indices[i]: i for i in range(n_clusters)}
    mapped_labels = [label_map[l] for l in labels]

    results = []
    for i, claim in enumerate(claims_list):
        results.append({
            "Claim ID": claim["id"],
            "Cluster": cluster_names[mapped_labels[i]],
            "Cluster ID": mapped_labels[i],
            "Billed Amount": claim["billed_amount"],
            "Procedures": len(claim["procedures"]),
            "Age": claim["patient_age"],
        })

    return pd.DataFrame(results)


def detect_time_series_anomalies(window_size=5, threshold=1.5):
    """Detect anomalies in daily claims volume using rolling statistics.

    Uses a moving average + standard deviation approach (Z-score method)
    to identify days with unusual billing patterns.
    """
    df = pd.DataFrame(SIMULATED_DAILY_CLAIMS)

    df["rolling_mean"] = df["count"].rolling(window=window_size, min_periods=1).mean()
    df["rolling_std"] = df["count"].rolling(window=window_size, min_periods=1).std().fillna(5)
    df["z_score"] = (df["count"] - df["rolling_mean"]) / df["rolling_std"].replace(0, 1)
    df["is_anomaly"] = df["z_score"].abs() > threshold

    df["amount_rolling_mean"] = df["avg_amount"].rolling(window=window_size, min_periods=1).mean()
    df["amount_z_score"] = (df["avg_amount"] - df["amount_rolling_mean"]) / df["avg_amount"].rolling(window=window_size, min_periods=1).std().fillna(50)
    df["amount_anomaly"] = df["amount_z_score"].abs() > threshold

    df["combined_anomaly"] = df["is_anomaly"] | df["amount_anomaly"]

    return df
