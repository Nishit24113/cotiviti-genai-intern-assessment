# Clinical Decision Making and Pattern Recognition in Health Care

## Cotiviti Intern Assessment — Nishit Pankajkumar Patel

**Topic:** Clinical Decision Making and Pattern Recognition in Health Care — Agentic Generative AI, Chain Reasoning, Classification, and Anomaly Detection for Treatment, Payment, & Operations (TPO)

---

## Repository Structure

```
├── poc/                        # Hackathon Proof of Concept
│   ├── app.py                  # Streamlit dashboard (main entry)
│   ├── agent.py                # AWS Bedrock LLM + RAG-augmented reasoning
│   ├── knowledge_base.py       # Vector embeddings + RAG retrieval
│   ├── rules_engine.py         # Deterministic business rules
│   ├── anomaly_detector.py     # Isolation Forest ML model
│   ├── sample_claims.py        # Sample clinical claims data
│   ├── requirements.txt        # Python dependencies
│   └── README.md               # POC architecture + setup
├── report/                     # Written Report
│   └── Cotiviti_Assessment_Report_Nishit_Patel.docx
├── presentation/               # PowerPoint Presentation
│   └── Cotiviti_Assessment_Presentation_Nishit_Patel.pptx
├── video/                      # Video Recording (MP4)
└── README.md                   # This file
```

## Proof of Concept: Clinical Claims Decision Agent

A four-layer **agentic AI pipeline** for healthcare payment integrity that demonstrates:

1. **Rules Engine** — Deterministic business rules for known billing patterns
2. **RAG Knowledge Retrieval** — Vector embeddings (AWS Titan) + cosine similarity for clinical guidelines
3. **Anomaly Detection** — Isolation Forest (unsupervised ML) for statistical outlier detection
4. **Agentic GenAI Reasoning** — Claude LLM chain-of-thought with RAG-augmented context

### Quick Start

```bash
cd poc
pip install -r requirements.txt
aws configure --profile sandbox2025
streamlit run app.py
```

### Technologies

Python | AWS Bedrock (Claude + Titan Embeddings) | scikit-learn | Streamlit | boto3

---

## Author

**Nishit Pankajkumar Patel**  
M.S. Computer Science, Arizona State University (GPA: 4.0)  
nishitpatel24113@gmail.com | github.com/Nishit24113
