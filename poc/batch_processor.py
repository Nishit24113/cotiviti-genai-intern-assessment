"""
Batch processing module for analyzing multiple claims at once.
Demonstrates scalability thinking — production systems process
thousands of claims per hour through this pipeline.
"""

import pandas as pd
from rules_engine import run_rules_engine
from anomaly_detector import get_anomaly_score


def process_batch(claims_list):
    """Process a batch of claims through rules + anomaly layers.
    Returns a DataFrame with results for each claim.
    """
    results = []

    for claim in claims_list:
        rules = run_rules_engine(claim)
        anomaly = get_anomaly_score(claim)

        results.append({
            "Claim ID": claim["id"],
            "Patient": f"{claim['patient_age']}yo {claim['gender']}",
            "Diagnosis": claim["primary_diagnosis"].split(" - ")[0],
            "Procedures": len(claim["procedures"]),
            "Billed Amount": claim["billed_amount"],
            "Provider": claim["provider_type"],
            "Rules Risk": rules["risk_score"],
            "Rule Flags": len(rules["flags"]),
            "Anomaly Score": anomaly["anomaly_score"],
            "Is Outlier": anomaly["is_anomaly"],
            "Pre-Decision": rules["pre_decision"],
            "Top Flag": rules["flags"][0]["rule"] if rules["flags"] else "None",
        })

    return pd.DataFrame(results)


def get_batch_summary(df):
    """Generate summary statistics from batch results."""
    total = len(df)
    flagged = len(df[df["Rule Flags"] > 0])
    outliers = len(df[df["Is Outlier"]])
    avg_risk = df["Rules Risk"].mean()
    total_billed = df["Billed Amount"].sum()
    high_risk = len(df[df["Rules Risk"] >= 0.5])

    return {
        "total_claims": total,
        "flagged_claims": flagged,
        "outlier_claims": outliers,
        "high_risk_claims": high_risk,
        "avg_risk_score": avg_risk,
        "total_billed": total_billed,
        "flag_rate": flagged / total if total > 0 else 0,
        "outlier_rate": outliers / total if total > 0 else 0,
    }
