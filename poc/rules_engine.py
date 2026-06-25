"""
Rule-based pre-screening engine for healthcare claims.
Implements deterministic business rules that flag common billing anomalies
BEFORE sending to the LLM for deeper chain-of-thought analysis.

This demonstrates the hybrid approach: rules + ML + GenAI working together.
"""

PROCEDURE_DIAGNOSIS_MAP = {
    "E11": ["99213", "99214", "99215", "83036", "80053", "80048", "36415"],
    "J06": ["99213", "99212", "87880", "87081"],
    "M54": ["99213", "99214", "99215", "97110", "97140", "72148", "72149"],
    "I10": ["99213", "99214", "93000", "93005", "80053", "80048"],
    "S93": ["99281", "99282", "99283", "73610", "73600", "29540", "29550"],
}

COST_THRESHOLDS = {
    "99212": 120, "99213": 200, "99214": 350, "99215": 500,
    "99281": 300, "99282": 500, "99283": 800, "99284": 1200, "99285": 1800,
    "71046": 400, "72148": 2000, "73610": 250, "93000": 200,
    "29540": 150, "87880": 50, "83036": 80,
}

HIGH_COST_PROCEDURES = ["72148", "72149", "70553", "74177", "71250"]


def extract_cpt_code(procedure_str):
    return procedure_str.split(" ")[0].strip()


def extract_icd_prefix(diagnosis_str):
    code = diagnosis_str.split(" ")[0].strip()
    return code.split(".")[0]


def run_rules_engine(claim):
    flags = []
    risk_score = 0.0

    icd_prefix = extract_icd_prefix(claim["primary_diagnosis"])
    cpt_codes = [extract_cpt_code(p) for p in claim["procedures"]]

    allowed_cpts = PROCEDURE_DIAGNOSIS_MAP.get(icd_prefix, [])
    if allowed_cpts:
        for cpt in cpt_codes:
            if cpt not in allowed_cpts:
                flags.append({
                    "rule": "PROCEDURE_DIAGNOSIS_MISMATCH",
                    "severity": "HIGH",
                    "detail": f"CPT {cpt} is not typically associated with ICD prefix {icd_prefix}",
                })
                risk_score += 0.3

    expected_cost = sum(COST_THRESHOLDS.get(cpt, 300) for cpt in cpt_codes)
    if claim["billed_amount"] > expected_cost * 1.5:
        flags.append({
            "rule": "COST_EXCEEDS_THRESHOLD",
            "severity": "MEDIUM",
            "detail": f"Billed ${claim['billed_amount']:.2f} exceeds expected ~${expected_cost:.2f} by {((claim['billed_amount']/expected_cost)-1)*100:.0f}%",
        })
        risk_score += 0.2

    high_cost_count = sum(1 for cpt in cpt_codes if cpt in HIGH_COST_PROCEDURES)
    if high_cost_count > 1:
        flags.append({
            "rule": "MULTIPLE_HIGH_COST_PROCEDURES",
            "severity": "HIGH",
            "detail": f"{high_cost_count} high-cost imaging/procedures in single visit",
        })
        risk_score += 0.25

    if len(cpt_codes) >= 4:
        flags.append({
            "rule": "HIGH_PROCEDURE_COUNT",
            "severity": "LOW",
            "detail": f"{len(cpt_codes)} procedures billed in single encounter",
        })
        risk_score += 0.1

    if claim["patient_age"] < 40 and any(cpt in HIGH_COST_PROCEDURES for cpt in cpt_codes):
        flags.append({
            "rule": "AGE_PROCEDURE_MISMATCH",
            "severity": "MEDIUM",
            "detail": f"High-cost imaging ordered for {claim['patient_age']}yo patient — may lack medical necessity without prior conservative treatment",
        })
        risk_score += 0.15

    risk_score = min(risk_score, 1.0)

    if risk_score >= 0.5:
        pre_decision = "FLAG_FOR_REVIEW"
    elif risk_score >= 0.3:
        pre_decision = "NEEDS_ANALYSIS"
    else:
        pre_decision = "LOW_RISK"

    return {
        "flags": flags,
        "risk_score": risk_score,
        "pre_decision": pre_decision,
        "procedures_checked": cpt_codes,
        "diagnosis_prefix": icd_prefix,
    }
