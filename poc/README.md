# Clinical Claims Decision Agent - Proof of Concept

An agentic AI system for healthcare claims decision-making that demonstrates autonomous reasoning, RAG-based knowledge retrieval, and multi-layered pattern recognition for payment integrity.

## Architecture

This POC implements a **four-layer agentic pipeline** that mirrors how an expert claims auditor would analyze a healthcare claim:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLAIM INPUT                                    │
└─────────────────────┬───────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: RULES ENGINE (Deterministic)                           │
│  • CPT-ICD procedure-diagnosis alignment                         │
│  • Cost threshold validation                                     │
│  • Known fraud pattern detection                                 │
└─────────────────────┬───────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: RAG KNOWLEDGE RETRIEVAL (Vector Embeddings)            │
│  • AWS Titan Embeddings generates vector representations         │
│  • Cosine similarity retrieves relevant clinical guidelines      │
│  • Guidelines passed as context to the reasoning agent           │
└─────────────────────┬───────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: ANOMALY DETECTION (Unsupervised ML)                    │
│  • Isolation Forest identifies statistical billing outliers       │
│  • Feature contribution analysis (explainability)                │
│  • Detects novel patterns without labeled training data           │
└─────────────────────┬───────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4: AGENTIC GenAI REASONING (LLM + RAG Context)            │
│  • 5-step chain-of-thought clinical reasoning                    │
│  • Uses retrieved guidelines for grounded decisions              │
│  • Autonomous classification: APPROVE / FLAG / DENY              │
│  • Generates human-readable audit narratives                     │
└─────────────────────┬───────────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 DECISION + AUDIT NARRATIVE                        │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| UI | Streamlit | Interactive web dashboard |
| LLM | AWS Bedrock (Claude) | Agentic chain-of-thought reasoning |
| Embeddings | AWS Titan Embed v2 | Vector representations for RAG |
| ML | scikit-learn (Isolation Forest) | Unsupervised anomaly detection |
| Rules | Custom Python | Deterministic business logic |
| Infrastructure | AWS (boto3) | Cloud-native ML/AI services |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure AWS credentials
aws configure --profile sandbox2025

# 3. Create .env file
cp .env.example .env

# 4. Run the application
streamlit run app.py
```

## Key Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit dashboard — main entry point |
| `agent.py` | AWS Bedrock LLM integration with RAG-augmented prompting |
| `knowledge_base.py` | Vector embedding store + cosine similarity retrieval |
| `rules_engine.py` | Deterministic CPT/ICD business rules |
| `anomaly_detector.py` | Isolation Forest statistical anomaly detection |
| `sample_claims.py` | 5 pre-loaded clinical claim scenarios |

## Demo Scenarios

The system includes 5 claims demonstrating different risk levels:

1. **CLM-2024-001** — Routine diabetes visit → APPROVE (low risk)
2. **CLM-2024-002** — Cold with unnecessary chest X-ray → FLAG (medium risk)
3. **CLM-2024-003** — First-visit back pain with immediate MRI → FLAG (high risk)
4. **CLM-2024-004** — Hypertension follow-up with ECG → APPROVE (low risk)
5. **CLM-2024-005** — Ankle sprain with unrelated lumbar MRI → DENY (fraud indicators)

## Author

**Nishit Pankajkumar Patel**  
M.S. Computer Science, Arizona State University  
Cotiviti GenAI Intern Assessment — Topic 2: Clinical Decision Making & Pattern Recognition
