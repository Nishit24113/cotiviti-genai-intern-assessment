"""
Statistical anomaly detection for claims data.
Uses Isolation Forest (unsupervised ML) to detect outlier billing patterns
without requiring labeled fraud data.

This demonstrates unsupervised anomaly detection - a key technique for
identifying novel fraud patterns that rule-based systems miss.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


HISTORICAL_CLAIMS = np.array([
    # [age, num_procedures, billed_amount, is_emergency, diagnosis_complexity]
    [45, 1, 250, 0, 1],   # routine diabetes
    [50, 1, 180, 0, 1],   # routine visit
    [67, 2, 420, 0, 2],   # cardiology follow-up
    [35, 1, 150, 0, 1],   # simple visit
    [72, 2, 380, 0, 2],   # multi-procedure routine
    [55, 1, 200, 0, 1],   # standard visit
    [40, 2, 350, 1, 1],   # ER moderate
    [28, 2, 450, 1, 1],   # ER with imaging
    [60, 1, 220, 0, 2],   # specialist visit
    [48, 2, 500, 0, 2],   # moderate complexity
    [33, 1, 130, 0, 1],   # young routine
    [58, 2, 600, 0, 2],   # older moderate
    [42, 3, 800, 1, 2],   # ER multiple procedures
    [70, 2, 450, 0, 2],   # elderly routine
    [38, 1, 180, 0, 1],   # young routine
    [65, 3, 900, 0, 3],   # complex elderly
    [52, 2, 380, 0, 2],   # moderate
    [44, 1, 200, 0, 1],   # routine
    [75, 2, 500, 0, 2],   # elderly follow-up
    [30, 1, 160, 0, 1],   # young simple
])


def get_anomaly_score(claim):
    is_emergency = 1 if claim["provider_type"] in ["Emergency Medicine", "Urgent Care"] else 0

    diagnosis_code = claim["primary_diagnosis"].split(" ")[0]
    if any(c in diagnosis_code for c in ["E11", "I10", "J06"]):
        complexity = 1
    elif any(c in diagnosis_code for c in ["M54", "S93"]):
        complexity = 2
    else:
        complexity = 2

    claim_features = np.array([[
        claim["patient_age"],
        len(claim["procedures"]),
        claim["billed_amount"],
        is_emergency,
        complexity,
    ]])

    scaler = StandardScaler()
    training_data = scaler.fit_transform(HISTORICAL_CLAIMS)
    claim_scaled = scaler.transform(claim_features)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.15,
        random_state=42,
    )
    model.fit(training_data)

    anomaly_score = model.decision_function(claim_scaled)[0]
    prediction = model.predict(claim_scaled)[0]

    normalized_score = max(0, min(1, (0.5 - anomaly_score) / 0.5))

    feature_contributions = []
    feature_names = ["Age", "Procedure Count", "Billed Amount", "Emergency Visit", "Diagnosis Complexity"]
    means = HISTORICAL_CLAIMS.mean(axis=0)
    stds = HISTORICAL_CLAIMS.std(axis=0)

    raw_features = [claim["patient_age"], len(claim["procedures"]),
                    claim["billed_amount"], is_emergency, complexity]

    for i, (name, val, mean, std) in enumerate(zip(feature_names, raw_features, means, stds)):
        if std > 0:
            z_score = abs(val - mean) / std
            contribution = min(z_score / 3.0, 1.0)
        else:
            contribution = 0
        feature_contributions.append({
            "feature": name,
            "value": val,
            "mean": round(mean, 1),
            "deviation": round(contribution, 3),
            "direction": "above" if val > mean else "below",
        })

    feature_contributions.sort(key=lambda x: x["deviation"], reverse=True)

    return {
        "anomaly_score": round(normalized_score, 3),
        "is_anomaly": prediction == -1,
        "raw_isolation_score": round(anomaly_score, 4),
        "feature_contributions": feature_contributions,
        "interpretation": _interpret_score(normalized_score, feature_contributions),
    }


def _interpret_score(score, contributions):
    if score < 0.3:
        level = "Normal"
        desc = "This claim falls within typical billing patterns."
    elif score < 0.6:
        level = "Moderate Outlier"
        desc = "Some features deviate from typical patterns."
    else:
        level = "Strong Anomaly"
        desc = "This claim significantly deviates from expected billing patterns."

    top_drivers = [c for c in contributions[:2] if c["deviation"] > 0.3]
    if top_drivers:
        drivers_text = ", ".join(
            f"{c['feature']} ({c['direction']} average)" for c in top_drivers
        )
        desc += f" Primary drivers: {drivers_text}."

    return {"level": level, "description": desc}
