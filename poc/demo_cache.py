"""
Pre-cached analysis results for demo/deployment mode.

When running without AWS credentials (e.g., on Streamlit Community Cloud),
the app uses these cached results to demonstrate the full pipeline
without making API calls. This ensures the demo is always available
and no credentials are exposed publicly.
"""

CACHED_RESULTS = {
    "CLM-2024-001": {
        "rules": {
            "flags": [],
            "risk_score": 0.0,
            "pre_decision": "LOW_RISK",
            "procedures_checked": ["99214"],
            "diagnosis_prefix": "E11",
        },
        "guidelines": [
            {
                "guideline_id": "CG-005",
                "category": "Diabetes Management",
                "guideline": "For Type 2 Diabetes without complications (E11.9), routine management visits every 3-6 months are standard of care. A moderate complexity visit (99214) is appropriate when managing diabetes with medication adjustments, reviewing A1C results, and assessing for complications.",
                "source": "ADA Standards of Medical Care in Diabetes",
                "relevance_score": 0.62,
            },
            {
                "guideline_id": "CG-009",
                "category": "Medical Necessity - General",
                "guideline": "Medical necessity requires that services be: (1) appropriate for the symptoms and diagnosis, (2) not more costly than alternative services that are equally effective, (3) not experimental or investigational, and (4) furnished at the most appropriate level of service.",
                "source": "CMS Medicare Benefit Policy Manual, Chapter 16",
                "relevance_score": 0.38,
            },
            {
                "guideline_id": "CG-003",
                "category": "E&M Coding - Visit Complexity",
                "guideline": "CPT 99215 (high complexity office visit) requires at least one of: high-complexity medical decision-making, or 40+ minutes of total time on the date of encounter.",
                "source": "AMA CPT Guidelines / CMS E&M Documentation Standards",
                "relevance_score": 0.31,
            },
        ],
        "anomaly": {
            "anomaly_score": 0.219,
            "is_anomaly": False,
            "raw_isolation_score": 0.1127,
            "feature_contributions": [
                {"feature": "Billed Amount", "value": 250.0, "mean": 370.0, "deviation": 0.179, "direction": "below"},
                {"feature": "Emergency Visit", "value": 0, "mean": 0.2, "deviation": 0.158, "direction": "below"},
                {"feature": "Procedure Count", "value": 1, "mean": 1.6, "deviation": 0.125, "direction": "below"},
                {"feature": "Age", "value": 45, "mean": 48.4, "deviation": 0.075, "direction": "below"},
                {"feature": "Diagnosis Complexity", "value": 1, "mean": 1.6, "deviation": 0.108, "direction": "below"},
            ],
            "interpretation": {
                "level": "Normal",
                "description": "This claim falls within typical billing patterns.",
            },
        },
        "llm": {
            "steps": [
                {
                    "step_number": 1,
                    "title": "Clinical Context Analysis",
                    "reasoning": "The patient is a 45-year-old male with Type 2 Diabetes Mellitus without complications (E11.9). This is a common chronic condition requiring regular monitoring. The visit is for routine diabetes management with A1C review, which is standard of care.",
                    "findings": "Routine diabetes follow-up visit. Patient demographics and diagnosis are consistent with expected clinical patterns.",
                },
                {
                    "step_number": 2,
                    "title": "Procedure-Diagnosis Alignment",
                    "reasoning": "CPT 99214 (established patient, moderate complexity) is appropriate for diabetes management visits involving medication review, A1C result interpretation, and complication screening. This code aligns well with E11.9.",
                    "findings": "CPT 99214 is clinically aligned with the diagnosis E11.9 and the described clinical scenario.",
                },
                {
                    "step_number": 3,
                    "title": "Billing Pattern Analysis",
                    "reasoning": "The billed amount of $250.00 for a 99214 visit falls within the typical range of $180-$350 for moderate complexity established patient visits in Family Medicine. No anomalies detected.",
                    "findings": "No anomalous billing patterns detected. Billed amount is consistent with market rates.",
                },
                {
                    "step_number": 4,
                    "title": "Medical Necessity Assessment",
                    "reasoning": "Routine monitoring of Type 2 Diabetes with A1C review is medically necessary per ADA guidelines, which recommend visits every 3-6 months for patients on medication therapy.",
                    "findings": "Visit is medically necessary per standard diabetes management guidelines.",
                },
                {
                    "step_number": 5,
                    "title": "Final Decision",
                    "reasoning": "All analysis layers confirm this is a legitimate, appropriately coded routine diabetes management visit. No flags from rules, anomaly detection, or clinical reasoning.",
                    "findings": "Claim is appropriate for approval without further review.",
                },
            ],
            "decision": "APPROVE",
            "confidence_score": 0.96,
            "risk_factors": [],
            "recommendation": "Approve claim. Routine diabetes management visit with appropriate coding and billing within normal parameters.",
        },
    },
    "CLM-2024-002": {
        "rules": {
            "flags": [
                {
                    "rule": "PROCEDURE_DIAGNOSIS_MISMATCH",
                    "severity": "HIGH",
                    "detail": "CPT 71046 is not typically associated with ICD prefix J06",
                }
            ],
            "risk_score": 0.3,
            "pre_decision": "NEEDS_ANALYSIS",
            "procedures_checked": ["99213", "71046"],
            "diagnosis_prefix": "J06",
        },
        "guidelines": [
            {
                "guideline_id": "CG-002",
                "category": "Imaging - Chest X-ray",
                "guideline": "Chest X-ray for acute upper respiratory infection (common cold) is not recommended unless symptoms persist beyond 10 days, the patient has high fever above 102F, or there are signs of lower respiratory involvement such as productive cough, shortness of breath, or abnormal lung sounds on auscultation.",
                "source": "American College of Radiology / AAFP Clinical Guidelines",
                "relevance_score": 0.71,
            },
            {
                "guideline_id": "CG-008",
                "category": "Billing Integrity - Upcoding",
                "guideline": "Upcoding is billing for a higher level of service than what was actually provided or documented.",
                "source": "OIG Work Plan / CMS RAC Program Guidelines",
                "relevance_score": 0.35,
            },
            {
                "guideline_id": "CG-009",
                "category": "Medical Necessity - General",
                "guideline": "Medical necessity requires that services be appropriate for the symptoms and diagnosis.",
                "source": "CMS Medicare Benefit Policy Manual, Chapter 16",
                "relevance_score": 0.29,
            },
        ],
        "anomaly": {
            "anomaly_score": 0.65,
            "is_anomaly": True,
            "raw_isolation_score": -0.085,
            "feature_contributions": [
                {"feature": "Billed Amount", "value": 680.0, "mean": 370.0, "deviation": 0.463, "direction": "above"},
                {"feature": "Procedure Count", "value": 2, "mean": 1.6, "deviation": 0.083, "direction": "above"},
                {"feature": "Age", "value": 32, "mean": 48.4, "deviation": 0.363, "direction": "below"},
                {"feature": "Emergency Visit", "value": 1, "mean": 0.2, "deviation": 0.632, "direction": "above"},
                {"feature": "Diagnosis Complexity", "value": 1, "mean": 1.6, "deviation": 0.108, "direction": "below"},
            ],
            "interpretation": {
                "level": "Moderate Outlier",
                "description": "Some features deviate from typical patterns. Primary drivers: Emergency Visit (above average), Billed Amount (above average).",
            },
        },
        "llm": {
            "steps": [
                {
                    "step_number": 1,
                    "title": "Clinical Context Analysis",
                    "reasoning": "A 32-year-old female presenting to urgent care with acute upper respiratory infection (J06.9). This is typically a self-limiting viral illness not requiring imaging.",
                    "findings": "Common cold in young healthy patient. Low acuity presentation.",
                },
                {
                    "step_number": 2,
                    "title": "Procedure-Diagnosis Alignment",
                    "reasoning": "The office visit (99213) is appropriate for a URI evaluation. However, the chest X-ray (71046) is not standard for uncomplicated URI. Per ACR guidelines, imaging is only indicated if symptoms persist >10 days or there are signs of lower respiratory involvement.",
                    "findings": "Chest X-ray is NOT typically aligned with acute URI diagnosis without documented complications.",
                },
                {
                    "step_number": 3,
                    "title": "Billing Pattern Analysis",
                    "reasoning": "Total billed amount of $680 seems elevated for a simple URI visit. The chest X-ray adds $350-400 to what should be a $150-200 visit. This pattern of adding imaging to simple viral visits warrants scrutiny.",
                    "findings": "Billing pattern suggests potential over-utilization of imaging for a low-complexity diagnosis.",
                },
                {
                    "step_number": 4,
                    "title": "Medical Necessity Assessment",
                    "reasoning": "Based on the clinical description ('common cold symptoms'), there is no documentation of fever, prolonged symptoms, or signs of pneumonia that would justify chest imaging.",
                    "findings": "Chest X-ray likely not medically necessary based on available clinical information.",
                },
                {
                    "step_number": 5,
                    "title": "Final Decision",
                    "reasoning": "The office visit is appropriate, but the chest X-ray lacks medical necessity documentation for an uncomplicated URI in a young patient. Flagging for clinical review to verify if additional symptoms justified imaging.",
                    "findings": "Flag for review - imaging may not be medically necessary.",
                },
            ],
            "decision": "FLAG_FOR_REVIEW",
            "confidence_score": 0.87,
            "risk_factors": [
                "Chest X-ray not standard for uncomplicated URI",
                "No documented clinical indication for imaging",
                "Elevated billing for low-complexity diagnosis",
            ],
            "recommendation": "Request clinical documentation supporting the chest X-ray. If no signs of lower respiratory involvement are documented, deny CPT 71046 and approve 99213 only.",
        },
    },
    "CLM-2024-005": {
        "rules": {
            "flags": [
                {
                    "rule": "PROCEDURE_DIAGNOSIS_MISMATCH",
                    "severity": "HIGH",
                    "detail": "CPT 72148 is not typically associated with ICD prefix S93",
                },
                {
                    "rule": "HIGH_PROCEDURE_COUNT",
                    "severity": "LOW",
                    "detail": "4 procedures billed in single encounter",
                },
                {
                    "rule": "AGE_PROCEDURE_MISMATCH",
                    "severity": "MEDIUM",
                    "detail": "High-cost imaging ordered for 28yo patient - may lack medical necessity without prior conservative treatment",
                },
            ],
            "risk_score": 0.55,
            "pre_decision": "FLAG_FOR_REVIEW",
            "procedures_checked": ["99283", "73610", "29540", "72148"],
            "diagnosis_prefix": "S93",
        },
        "guidelines": [
            {
                "guideline_id": "CG-004",
                "category": "Emergency Department - Ankle Injuries",
                "guideline": "Ottawa Ankle Rules dictate that X-ray is indicated for ankle injuries only when there is bony tenderness at the posterior edge or tip of either malleolus, or inability to bear weight immediately and in the ED. MRI is not indicated for acute ankle sprains and should only be considered if symptoms persist after 4-6 weeks of conservative management.",
                "source": "Ottawa Ankle Rules / ACEP Clinical Policy",
                "relevance_score": 0.72,
            },
            {
                "guideline_id": "CG-001",
                "category": "Imaging - Lumbar Spine",
                "guideline": "MRI of the lumbar spine is not recommended as initial imaging for acute low back pain in patients under 50 without red flags. Conservative treatment of 4-6 weeks should be attempted first.",
                "source": "American College of Radiology Appropriateness Criteria",
                "relevance_score": 0.58,
            },
            {
                "guideline_id": "CG-010",
                "category": "Fraud Indicators - Pattern Detection",
                "guideline": "Common fraud indicators include: procedures unrelated to the primary diagnosis, unusually high billing volume, and clustering of high-cost procedures without clinical escalation documentation.",
                "source": "FBI Healthcare Fraud Report / OIG Semiannual Report",
                "relevance_score": 0.45,
            },
        ],
        "anomaly": {
            "anomaly_score": 1.0,
            "is_anomaly": True,
            "raw_isolation_score": -0.312,
            "feature_contributions": [
                {"feature": "Procedure Count", "value": 4, "mean": 1.6, "deviation": 1.0, "direction": "above"},
                {"feature": "Billed Amount", "value": 4500.0, "mean": 370.0, "deviation": 1.0, "direction": "above"},
                {"feature": "Emergency Visit", "value": 1, "mean": 0.2, "deviation": 0.632, "direction": "above"},
                {"feature": "Age", "value": 28, "mean": 48.4, "deviation": 0.452, "direction": "below"},
                {"feature": "Diagnosis Complexity", "value": 2, "mean": 1.6, "deviation": 0.072, "direction": "above"},
            ],
            "interpretation": {
                "level": "Strong Anomaly",
                "description": "This claim significantly deviates from expected billing patterns. Primary drivers: Procedure Count (above average), Billed Amount (above average).",
            },
        },
        "llm": {
            "steps": [
                {
                    "step_number": 1,
                    "title": "Clinical Context Analysis",
                    "reasoning": "A 28-year-old male presenting to the Emergency Department with an ankle sprain (S93.401A). This is an acute musculoskeletal injury commonly seen in young active patients. Standard workup involves physical exam, possible X-ray per Ottawa rules, and conservative treatment.",
                    "findings": "Acute ankle sprain in young male. Standard low-complexity ER presentation.",
                },
                {
                    "step_number": 2,
                    "title": "Procedure-Diagnosis Alignment",
                    "reasoning": "The ankle X-ray (73610) and strapping (29540) are appropriate for ankle sprain. The ER visit (99283) is reasonable. However, the lumbar MRI (72148) has NO clinical relationship to an ankle sprain diagnosis. There is no documented back complaint.",
                    "findings": "CRITICAL MISMATCH: Lumbar MRI (72148) is completely unrelated to ankle sprain diagnosis (S93). This is a major red flag.",
                },
                {
                    "step_number": 3,
                    "title": "Billing Pattern Analysis",
                    "reasoning": "Total billed amount of $4,500 is extremely high for an ankle sprain. The lumbar MRI alone accounts for approximately $2,000 of unnecessary charges. This pattern of adding unrelated high-cost imaging to routine ER visits is a known fraud indicator.",
                    "findings": "Billing pattern consistent with known fraud indicators: unrelated high-cost procedure bundled with legitimate visit.",
                },
                {
                    "step_number": 4,
                    "title": "Medical Necessity Assessment",
                    "reasoning": "Per Ottawa Ankle Rules, X-ray is appropriate if clinical criteria are met. Strapping is standard treatment. However, lumbar MRI has ZERO medical necessity for an ankle sprain. No documented back pain, no trauma mechanism suggesting spinal involvement.",
                    "findings": "Lumbar MRI fails medical necessity on all criteria. No supporting diagnosis or clinical indication.",
                },
                {
                    "step_number": 5,
                    "title": "Final Decision",
                    "reasoning": "Three of four procedures are appropriate for ankle sprain management. The lumbar MRI is completely unjustified, represents a significant billing anomaly, and matches fraud patterns. Recommend denial of the MRI with referral to Special Investigations Unit.",
                    "findings": "DENY lumbar MRI. Flag provider for SIU review. Approve remaining procedures.",
                },
            ],
            "decision": "DENY",
            "confidence_score": 0.95,
            "risk_factors": [
                "Lumbar MRI completely unrelated to ankle sprain diagnosis",
                "No documented clinical indication for spinal imaging",
                "Billing pattern matches known fraud indicators",
                "Total charges 12x higher than typical ankle sprain ER visit",
            ],
            "recommendation": "DENY CPT 72148 (MRI Lumbar Spine) - no medical necessity. Approve 99283, 73610, 29540. Refer provider to Special Investigations Unit for pattern analysis of unrelated imaging add-ons.",
        },
    },
}
