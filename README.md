# Clinical Decision Making and Pattern Recognition in Health Care

## Cotiviti Intern Assessment — Nishit Pankajkumar Patel

**Topic:** Clinical Decision Making and Pattern Recognition in Health Care — Agentic Generative AI, Chain Reasoning, Classification, and Anomaly Detection for Treatment, Payment, & Operations (TPO)

---

## Live Demo

[View Live Application](https://cotiviti-genai-intern-assessment.streamlit.app/) (Demo mode — no credentials required)

---

## Repository Structure

```
├── .streamlit/                 # Streamlit Cloud configuration
│   └── config.toml             # Theme and server settings
├── poc/                        # Hackathon Proof of Concept
│   ├── app.py                  # Streamlit dashboard (main entry)
│   ├── agent.py                # AWS Bedrock LLM + RAG-augmented reasoning
│   ├── knowledge_base.py       # Vector embeddings + RAG retrieval
│   ├── rules_engine.py         # Deterministic business rules
│   ├── anomaly_detector.py     # Isolation Forest ML model
│   ├── batch_processor.py      # Batch analysis + statistics
│   ├── demo_cache.py           # Cached results for credential-free demo
│   ├── sample_claims.py        # Sample clinical claims data
│   ├── requirements.txt        # Python dependencies
│   └── README.md               # Architecture documentation
├── report/                     # Written Report
│   └── Cotiviti_Assessment_Report_Nishit_Patel.docx
├── presentation/               # PowerPoint Presentation
│   └── Cotiviti_Assessment_Presentation_Nishit_Patel.pptx
├── video/                      # Video Recording (MP4)
└── README.md                   # This file
```

## Proof of Concept: Clinical Claims Decision Agent

An **agentic AI system** demonstrating autonomous reasoning, goal-directed behavior, and multi-step decision-making for healthcare payment integrity.

### Architecture: Four-Layer Pipeline

| Layer | Technology | Function |
|-------|-----------|----------|
| 1. Rules Engine | Custom Python | Deterministic CPT-ICD validation, cost thresholds |
| 2. RAG Retrieval | AWS Titan Embeddings | Vector similarity search over clinical guidelines |
| 3. Anomaly Detection | scikit-learn (Isolation Forest) | Unsupervised statistical outlier detection |
| 4. Agentic GenAI | AWS Bedrock (Claude) | Chain-of-thought reasoning with RAG context |

### Key Features

- **Autonomous Decision-Making** — Agent performs 5-step clinical reasoning independently
- **RAG-Grounded Reasoning** — Decisions reference actual clinical coding guidelines
- **Explainable AI** — Complete audit trail and reasoning chain for every decision
- **Batch Processing** — Analyze multiple claims with aggregate statistics and Plotly visualizations
- **Export Functionality** — CSV export for downstream analysis
- **Demo Mode** — Runs without credentials using cached results for public access

### Quick Start (Local with AWS)

```bash
cd poc
pip install -r requirements.txt
aws configure --profile sandbox2025
streamlit run app.py
```

### Technologies

Python | AWS Bedrock (Claude + Titan Embeddings) | scikit-learn | Streamlit | Plotly | Pandas

---

## Deliverables

| # | Deliverable | Description |
|---|-------------|-------------|
| 1 | Written Report | 2-page analysis + bibliography (APA format) |
| 2 | POC Demo | Working agentic AI pipeline with Streamlit UI |
| 3 | Presentation | 11-slide PowerPoint with architecture overview |
| 4 | Video | Recorded presentation + live demo walkthrough |

---

## Author

**Nishit Pankajkumar Patel**  
M.S. Computer Science, Arizona State University (GPA: 4.0)  
nishitpatel24113@gmail.com | github.com/Nishit24113
