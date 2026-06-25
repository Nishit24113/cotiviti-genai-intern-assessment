"""
RAG-based Clinical Knowledge Retrieval using Vector Embeddings.

This module implements a lightweight Retrieval-Augmented Generation (RAG) system
that stores clinical coding guidelines as vector embeddings and retrieves
relevant context for the AI agent's decision-making process.

Technologies demonstrated:
- AWS Bedrock Titan Embeddings (vector embedding generation)
- Cosine similarity search (vector retrieval)
- In-memory vector store (production would use Pinecone/pgvector/OpenSearch)
"""

import json
import numpy as np
import boto3
from dotenv import load_dotenv
import os

load_dotenv()

AWS_PROFILE = os.getenv("AWS_PROFILE", "sandbox2025")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIM = 256

CLINICAL_GUIDELINES = [
    {
        "id": "CG-001",
        "category": "Imaging - Lumbar Spine",
        "guideline": "MRI of the lumbar spine is not recommended as initial imaging for acute low back pain in patients under 50 without red flags (fever, weight loss, trauma, neurological deficits). Conservative treatment of 4-6 weeks should be attempted first. Red flags warranting immediate imaging include: cauda equina syndrome, progressive neurological deficit, suspected spinal infection, or history of cancer with new back pain.",
        "source": "American College of Radiology Appropriateness Criteria",
    },
    {
        "id": "CG-002",
        "category": "Imaging - Chest X-ray",
        "guideline": "Chest X-ray for acute upper respiratory infection (common cold) is not recommended unless symptoms persist beyond 10 days, the patient has high fever above 102F, or there are signs of lower respiratory involvement such as productive cough, shortness of breath, or abnormal lung sounds on auscultation.",
        "source": "American College of Radiology / AAFP Clinical Guidelines",
    },
    {
        "id": "CG-003",
        "category": "E&M Coding - Visit Complexity",
        "guideline": "CPT 99215 (high complexity office visit) requires at least one of: high-complexity medical decision-making (multiple diagnoses, extensive data review, high-risk management), or 40+ minutes of total time on the date of encounter. A first visit for uncomplicated low back pain typically qualifies for 99213 or 99214 unless comorbidities or complications elevate complexity.",
        "source": "AMA CPT Guidelines / CMS E&M Documentation Standards",
    },
    {
        "id": "CG-004",
        "category": "Emergency Department - Ankle Injuries",
        "guideline": "Ottawa Ankle Rules dictate that X-ray is indicated for ankle injuries only when there is bony tenderness at the posterior edge or tip of either malleolus, or inability to bear weight immediately and in the ED. MRI is not indicated for acute ankle sprains and should only be considered if symptoms persist after 4-6 weeks of conservative management or if there is clinical suspicion of osteochondral lesion.",
        "source": "Ottawa Ankle Rules / ACEP Clinical Policy",
    },
    {
        "id": "CG-005",
        "category": "Diabetes Management",
        "guideline": "For Type 2 Diabetes without complications (E11.9), routine management visits every 3-6 months are standard of care. A moderate complexity visit (99214) is appropriate when managing diabetes with medication adjustments, reviewing A1C results, and assessing for complications. This includes metabolic panel review and foot examination.",
        "source": "ADA Standards of Medical Care in Diabetes",
    },
    {
        "id": "CG-006",
        "category": "Cardiology - Hypertension",
        "guideline": "For essential hypertension (I10) follow-up visits, ECG (93000) is appropriate for annual cardiovascular screening in patients over 65, or when starting/adjusting antihypertensive medications, or when symptoms suggest cardiac involvement. A moderate complexity visit (99214) is appropriate for hypertension management with medication adjustments.",
        "source": "ACC/AHA Hypertension Clinical Practice Guidelines",
    },
    {
        "id": "CG-007",
        "category": "Billing Integrity - Unbundling",
        "guideline": "Unbundling occurs when a provider bills separately for procedures that should be billed as a single comprehensive code. Common examples include: billing separately for each component of a panel test, billing both a comprehensive and component procedure, or billing modifier-25 with E&M when the procedure was the primary reason for the visit without separate documentation of a significant, separately identifiable service.",
        "source": "OIG Compliance Program Guidance / CMS NCCI Edits",
    },
    {
        "id": "CG-008",
        "category": "Billing Integrity - Upcoding",
        "guideline": "Upcoding is billing for a higher level of service than what was actually provided or documented. Indicators include: E&M level inconsistent with diagnosis complexity (e.g., 99215 for uncomplicated URI), pattern of exclusively billing highest-level codes, and documentation that does not support the billed level of medical decision-making. Statistical analysis should compare provider coding patterns against specialty-specific benchmarks.",
        "source": "OIG Work Plan / CMS RAC Program Guidelines",
    },
    {
        "id": "CG-009",
        "category": "Medical Necessity - General",
        "guideline": "Medical necessity requires that services be: (1) appropriate for the symptoms and diagnosis, (2) not more costly than alternative services that are equally effective, (3) not experimental or investigational, and (4) furnished at the most appropriate level of service. Services that do not meet these criteria may be denied as not medically necessary under Medicare/Medicaid guidelines.",
        "source": "CMS Medicare Benefit Policy Manual, Chapter 16",
    },
    {
        "id": "CG-010",
        "category": "Fraud Indicators - Pattern Detection",
        "guideline": "Common fraud indicators in healthcare claims include: procedures unrelated to the primary diagnosis, unusually high billing volume per provider, services billed on dates the provider was unavailable, identical services billed for multiple patients on the same day beyond reasonable capacity, and clustering of high-cost procedures without clinical escalation documentation. Time-series analysis of billing patterns can reveal sudden spikes indicating potential fraud onset.",
        "source": "FBI Healthcare Fraud Report / OIG Semiannual Report",
    },
]


class VectorKnowledgeBase:
    def __init__(self):
        self._embeddings_cache = {}
        self._client = None

    @property
    def client(self):
        if self._client is None:
            session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
            self._client = session.client("bedrock-runtime")
        return self._client

    def _get_embedding(self, text):
        if text in self._embeddings_cache:
            return self._embeddings_cache[text]

        response = self.client.invoke_model(
            modelId=EMBEDDING_MODEL,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text[:2000], "dimensions": EMBEDDING_DIM}),
        )
        body = json.loads(response["body"].read())
        embedding = np.array(body["embedding"])
        self._embeddings_cache[text] = embedding
        return embedding

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def build_index(self):
        """Generate embeddings for all clinical guidelines."""
        self._index = []
        for doc in CLINICAL_GUIDELINES:
            text = f"{doc['category']}: {doc['guideline']}"
            embedding = self._get_embedding(text)
            self._index.append({"doc": doc, "embedding": embedding})
        return len(self._index)

    def retrieve(self, query, top_k=3):
        """Retrieve most relevant clinical guidelines for a given claim context."""
        if not hasattr(self, "_index") or not self._index:
            self.build_index()

        query_embedding = self._get_embedding(query)

        scores = []
        for item in self._index:
            sim = self._cosine_similarity(query_embedding, item["embedding"])
            scores.append((sim, item["doc"]))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scores[:top_k]:
            results.append({
                "guideline_id": doc["id"],
                "category": doc["category"],
                "guideline": doc["guideline"],
                "source": doc["source"],
                "relevance_score": round(float(score), 3),
            })

        return results


_kb_instance = None


def get_knowledge_base():
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = VectorKnowledgeBase()
    return _kb_instance


def retrieve_relevant_guidelines(claim_data):
    """Build a query from claim data and retrieve relevant guidelines."""
    query = (
        f"Patient age {claim_data['patient_age']} {claim_data['gender']} "
        f"diagnosed with {claim_data['primary_diagnosis']} "
        f"procedures: {', '.join(claim_data['procedures'])} "
        f"provider: {claim_data['provider_type']} "
        f"billed: ${claim_data['billed_amount']}"
    )

    kb = get_knowledge_base()
    return kb.retrieve(query, top_k=3)
