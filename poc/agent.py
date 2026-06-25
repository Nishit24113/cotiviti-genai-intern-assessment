import json
import boto3
from dotenv import load_dotenv
import os
from knowledge_base import retrieve_relevant_guidelines

load_dotenv()

AWS_PROFILE = os.getenv("AWS_PROFILE", "sandbox2025")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("MODEL_ID", "us.anthropic.claude-sonnet-4-6")

SYSTEM_PROMPT = """You are a Clinical Claims Decision Agent working for a healthcare payment integrity company.
Your role is to analyze medical claims using chain-of-thought reasoning to determine if a claim should be approved, flagged for review, or denied.

You must analyze each claim through these sequential reasoning steps:

**Step 1 - Clinical Context Analysis:** Understand the patient demographics, diagnosis, and clinical scenario.

**Step 2 - Procedure-Diagnosis Alignment:** Evaluate whether the procedures billed are medically appropriate for the stated diagnosis. Consider standard clinical guidelines.

**Step 3 - Billing Pattern Analysis:** Check for anomalies such as:
- Unusually high costs for the diagnosis
- Procedures unrelated to the primary diagnosis
- Upcoding (billing a higher complexity than warranted)
- Unbundling (billing separately for procedures that should be combined)

**Step 4 - Medical Necessity Assessment:** Determine if each procedure was medically necessary given the clinical context.

**Step 5 - Final Decision:** Based on all prior reasoning, classify the claim.

IMPORTANT: Respond ONLY with valid JSON. Do not use special characters like em-dashes (use hyphens instead). Do not wrap in markdown code fences.

You MUST respond in the following JSON format:
{
    "steps": [
        {
            "step_number": 1,
            "title": "Clinical Context Analysis",
            "reasoning": "Your detailed reasoning here",
            "findings": "Key findings from this step"
        },
        {
            "step_number": 2,
            "title": "Procedure-Diagnosis Alignment",
            "reasoning": "Your detailed reasoning here",
            "findings": "Key findings from this step"
        },
        {
            "step_number": 3,
            "title": "Billing Pattern Analysis",
            "reasoning": "Your detailed reasoning here",
            "findings": "Key findings from this step"
        },
        {
            "step_number": 4,
            "title": "Medical Necessity Assessment",
            "reasoning": "Your detailed reasoning here",
            "findings": "Key findings from this step"
        },
        {
            "step_number": 5,
            "title": "Final Decision",
            "reasoning": "Your detailed reasoning here",
            "findings": "Summary of decision rationale"
        }
    ],
    "decision": "APPROVE" or "FLAG_FOR_REVIEW" or "DENY",
    "confidence_score": 0.0 to 1.0,
    "risk_factors": ["list of identified risk factors if any"],
    "recommendation": "Brief recommendation for the claims reviewer"
}"""


def get_bedrock_client():
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return session.client("bedrock-runtime")


def _parse_json_response(text):
    """Robustly extract JSON from LLM response, handling markdown fences and special chars."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        last_fence = cleaned.rfind("```")
        if last_fence > first_newline:
            cleaned = cleaned[first_newline + 1:last_fence].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON object found")

    json_str = cleaned[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    json_str = json_str.replace("—", "-").replace("–", "-")
    json_str = json_str.replace("‘", "'").replace("’", "'")
    json_str = json_str.replace("“", '"').replace("”", '"')

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    import re
    json_str = re.sub(r'(?<!\\)"([^"]*?)(?<!\\)"(\s*[,}\]])',
                      lambda m: '"' + m.group(1).replace('"', '\\"') + '"' + m.group(2),
                      json_str)

    return json.loads(json_str)


def analyze_claim(claim_data: dict, guidelines=None) -> dict:
    client = get_bedrock_client()

    if guidelines is None:
        guidelines = retrieve_relevant_guidelines(claim_data)

    guidelines_text = ""
    if guidelines:
        guidelines_text = "\n\nRELEVANT CLINICAL GUIDELINES (retrieved via RAG):\n"
        for g in guidelines:
            guidelines_text += f"- [{g['source']}] {g['guideline']}\n"

    claim_text = f"""Please analyze the following healthcare claim:

Claim ID: {claim_data['id']}
Patient Age: {claim_data['patient_age']}
Patient Gender: {claim_data['gender']}
Primary Diagnosis: {claim_data['primary_diagnosis']}
Procedures Billed: {', '.join(claim_data['procedures'])}
Total Billed Amount: ${claim_data['billed_amount']:.2f}
Provider Type: {claim_data['provider_type']}
Clinical Description: {claim_data['description']}
{guidelines_text}
Use the retrieved guidelines above to inform your analysis. Perform your full chain-of-thought analysis and provide your decision."""

    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": claim_text}],
            }
        ),
    )

    response_body = json.loads(response["body"].read())
    assistant_text = response_body["content"][0]["text"]

    try:
        result = _parse_json_response(assistant_text)
    except (json.JSONDecodeError, ValueError):
        result = {
            "steps": [
                {
                    "step_number": 1,
                    "title": "Analysis Complete",
                    "reasoning": assistant_text[:500],
                    "findings": "Response generated but structured parsing failed. See reasoning above.",
                }
            ],
            "decision": "FLAG_FOR_REVIEW",
            "confidence_score": 0.5,
            "risk_factors": ["Structured output parse issue"],
            "recommendation": "Manual review recommended — AI analysis completed but output format was non-standard.",
        }

    return result


def analyze_custom_claim(description: str) -> dict:
    client = get_bedrock_client()

    claim_text = f"""Please analyze the following healthcare claim scenario described by the user:

{description}

Extract what you can about the diagnosis, procedures, and billing context. Then perform your full chain-of-thought analysis and provide your decision."""

    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": claim_text}],
            }
        ),
    )

    response_body = json.loads(response["body"].read())
    assistant_text = response_body["content"][0]["text"]

    try:
        result = _parse_json_response(assistant_text)
    except (json.JSONDecodeError, ValueError):
        result = {
            "steps": [
                {
                    "step_number": 1,
                    "title": "Analysis Complete",
                    "reasoning": assistant_text[:500],
                    "findings": "Response generated but structured parsing failed. See reasoning above.",
                }
            ],
            "decision": "FLAG_FOR_REVIEW",
            "confidence_score": 0.5,
            "risk_factors": ["Structured output parse issue"],
            "recommendation": "Manual review recommended — AI analysis completed but output format was non-standard.",
        }

    return result
