import streamlit as st
import os
from rules_engine import run_rules_engine
from anomaly_detector import get_anomaly_score
from sample_claims import SAMPLE_CLAIMS
from demo_cache import CACHED_RESULTS


def is_demo_mode():
    """Check if AWS credentials are available. If not, run in demo mode."""
    try:
        import boto3
        session = boto3.Session(
            profile_name=os.getenv("AWS_PROFILE", "sandbox2025"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        sts = session.client("sts")
        sts.get_caller_identity()
        return False
    except Exception:
        return True


DEMO_MODE = is_demo_mode()

if not DEMO_MODE:
    from agent import analyze_claim, analyze_custom_claim
    from knowledge_base import retrieve_relevant_guidelines

st.set_page_config(
    page_title="Clinical Claims Decision Agent",
    page_icon="🏥",
    layout="wide",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 10px; color: white; text-align: center; margin: 5px 0;
    }
    .flag-HIGH { border-left: 4px solid #dc3545; background: #fff5f5; padding: 8px 12px; margin: 4px 0; border-radius: 4px; }
    .flag-MEDIUM { border-left: 4px solid #ffc107; background: #fffdf5; padding: 8px 12px; margin: 4px 0; border-radius: 4px; }
    .flag-LOW { border-left: 4px solid #17a2b8; background: #f5fdff; padding: 8px 12px; margin: 4px 0; border-radius: 4px; }
    .dashboard-header {
        background: linear-gradient(135deg, #4A154B 0%, #7B2D8B 100%);
        padding: 30px; border-radius: 12px; color: white; margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def display_pipeline_results(rules_result, anomaly_result, llm_result, guidelines=None):
    st.markdown("---")
    st.subheader("Analysis Pipeline Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        rule_score = rules_result["risk_score"]
        st.metric("Rules Risk", f"{rule_score:.0%}",
                  delta=f"{len(rules_result['flags'])} flags",
                  delta_color="inverse" if rules_result["flags"] else "off")
    with col2:
        if guidelines:
            st.metric("Guidelines Found", f"{len(guidelines)}",
                      delta=f"{guidelines[0]['relevance_score']:.0%} top match")
    with col3:
        anom_score = anomaly_result["anomaly_score"]
        st.metric("Anomaly Score", f"{anom_score:.0%}",
                  delta="Outlier" if anomaly_result["is_anomaly"] else "Normal",
                  delta_color="inverse" if anomaly_result["is_anomaly"] else "off")
    with col4:
        confidence = llm_result.get("confidence_score", 0)
        decision = llm_result.get("decision", "UNKNOWN")
        st.metric("AI Decision", decision, delta=f"{confidence:.0%} confidence")

    # --- Layer 1: Rules Engine ---
    st.markdown("---")
    st.subheader("Layer 1: Deterministic Rules Engine")
    st.caption("Business rules for known billing patterns — instant, explainable")

    if rules_result["flags"]:
        for flag in rules_result["flags"]:
            severity = flag["severity"]
            icon = "🔴" if severity == "HIGH" else "🟡" if severity == "MEDIUM" else "🔵"
            st.markdown(
                f'<div class="flag-{severity}">{icon} <b>[{flag["rule"]}]</b> — {flag["detail"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("No rule-based flags triggered. Claim passes initial screening.")

    # --- Layer 2: RAG Knowledge Retrieval ---
    if guidelines:
        st.markdown("---")
        st.subheader("Layer 2: RAG Knowledge Retrieval (Vector Embeddings)")
        st.caption("AWS Titan Embeddings + cosine similarity retrieve relevant clinical guidelines")

        for g in guidelines:
            score_pct = f"{g['relevance_score']:.0%}"
            with st.expander(f"[{g['category']}] — Relevance: {score_pct}", expanded=False):
                st.markdown(f"> {g['guideline']}")
                st.caption(f"Source: {g['source']}")

    # --- Layer 3: Anomaly Detection ---
    st.markdown("---")
    st.subheader("Layer 3: ML Anomaly Detection (Isolation Forest)")
    st.caption("Unsupervised ML identifies statistical outliers without labeled training data")

    interpretation = anomaly_result["interpretation"]
    if anomaly_result["is_anomaly"]:
        st.warning(f"**{interpretation['level']}** — {interpretation['description']}")
    else:
        st.success(f"**{interpretation['level']}** — {interpretation['description']}")

    st.markdown("**Feature Deviation Analysis:**")
    for contrib in anomaly_result["feature_contributions"]:
        bar_width = int(contrib["deviation"] * 100)
        st.markdown(
            f"- **{contrib['feature']}**: {contrib['value']} (avg: {contrib['mean']}) — "
            f"deviation: {'█' * max(1, bar_width // 10)}{'░' * (10 - bar_width // 10)} {contrib['deviation']:.1%}"
        )

    # --- Layer 4: LLM Chain Reasoning ---
    st.markdown("---")
    st.subheader("Layer 4: Agentic GenAI Chain-of-Thought Reasoning")
    st.caption("Claude LLM performs autonomous 5-step clinical analysis using retrieved guidelines")

    for step in llm_result.get("steps", []):
        step_num = step.get("step_number", "?")
        title = step.get("title", "Unknown")
        reasoning = step.get("reasoning", "")
        findings = step.get("findings", "")

        with st.expander(f"Step {step_num}: {title}", expanded=(step_num == 5)):
            st.markdown(f"**Reasoning:** {reasoning}")
            st.markdown(f"**Findings:** {findings}")

    # --- Final Decision ---
    st.markdown("---")
    st.subheader("Final Integrated Decision")

    decision = llm_result.get("decision", "UNKNOWN")
    risk_factors = llm_result.get("risk_factors", [])
    recommendation = llm_result.get("recommendation", "")

    col1, col2 = st.columns([3, 1])
    with col1:
        if decision == "APPROVE":
            st.success(f"### APPROVED")
        elif decision == "FLAG_FOR_REVIEW":
            st.warning(f"### FLAGGED FOR REVIEW")
        else:
            st.error(f"### DENIED")
        st.markdown(f"**Recommendation:** {recommendation}")

    with col2:
        st.metric("Confidence", f"{llm_result.get('confidence_score', 0):.0%}")
        combined_risk = (rules_result["risk_score"] * 0.3 +
                         anomaly_result["anomaly_score"] * 0.3 +
                         (1 - llm_result.get("confidence_score", 0.5)) * 0.4)
        st.metric("Combined Risk", f"{combined_risk:.0%}")

    if risk_factors:
        st.markdown("**Risk Factors:**")
        for rf in risk_factors:
            st.markdown(f"- {rf}")

    # --- Audit Narrative ---
    st.markdown("---")
    st.subheader("Auto-Generated Audit Narrative")
    narrative = _generate_narrative(rules_result, anomaly_result, llm_result)
    st.info(narrative)


def _generate_narrative(rules_result, anomaly_result, llm_result):
    decision = llm_result.get("decision", "UNKNOWN")
    confidence = llm_result.get("confidence_score", 0)

    parts = []
    parts.append(f"**Decision: {decision}** (Confidence: {confidence:.0%})")
    parts.append("")

    if rules_result["flags"]:
        parts.append(f"The rules engine identified {len(rules_result['flags'])} flag(s):")
        for f in rules_result["flags"]:
            parts.append(f"  - [{f['severity']}] {f['detail']}")
        parts.append("")

    interp = anomaly_result["interpretation"]
    parts.append(f"Statistical analysis: {interp['level']} — {interp['description']}")
    parts.append("")

    recommendation = llm_result.get("recommendation", "")
    if recommendation:
        parts.append(f"AI Recommendation: {recommendation}")

    return "\n".join(parts)


# ============================================================
# MAIN UI
# ============================================================

with st.sidebar:
    if DEMO_MODE:
        st.info("**Demo Mode** — Using cached results (no AWS credentials detected)")
    else:
        st.success("**Live Mode** — Connected to AWS Bedrock")
    st.markdown("---")
    st.header("Architecture")
    st.markdown(
        """
    ### Four-Layer Agentic Pipeline

    **1. Rules Engine**
    Deterministic CPT-ICD validation

    **2. RAG Retrieval**
    Vector embeddings + clinical guidelines

    **3. Anomaly Detection**
    Isolation Forest (unsupervised ML)

    **4. Agentic GenAI**
    Chain-of-thought + audit narratives

    ---
    ### Technologies
    - AWS Bedrock (Claude)
    - AWS Titan Embeddings
    - scikit-learn
    - Streamlit
    - Python
    """
    )
    st.markdown("---")
    st.markdown("**Built by Nishit Patel**")
    st.markdown("*Arizona State University*")
    st.markdown("*Cotiviti GenAI Intern Assessment*")

# --- Dashboard Header ---
st.markdown(
    '<div class="dashboard-header">'
    '<h1 style="color: white; margin: 0;">🏥 Clinical Claims Decision Agent</h1>'
    '<p style="color: #BB86FC; margin: 5px 0 0 0; font-size: 1.2em;">'
    'Agentic AI for Healthcare Payment Integrity</p>'
    '</div>',
    unsafe_allow_html=True,
)

# --- Dashboard Metrics ---
st.markdown("### System Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Knowledge Base", "10 Guidelines", delta="Vector indexed")
with col2:
    st.metric("ML Model", "Isolation Forest", delta="20 training samples")
with col3:
    st.metric("Rules Engine", "5 Rule Types", delta="Active")
with col4:
    st.metric("LLM Agent", "Claude (Bedrock)", delta="5-step reasoning")

st.markdown("---")

# --- Main Content ---
tab1, tab2, tab3 = st.tabs(["📋 Claim Analysis", "✏️ Custom Scenario", "ℹ️ About"])

with tab1:
    st.subheader("Select a Clinical Claim for Analysis")

    claim_options = {
        f"{c['id']}: {c['description']}": i for i, c in enumerate(SAMPLE_CLAIMS)
    }
    selected = st.selectbox("Select a claim:", list(claim_options.keys()))
    selected_claim = SAMPLE_CLAIMS[claim_options[selected]]

    with st.expander("Claim Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Claim ID:** {selected_claim['id']}")
            st.markdown(f"**Patient:** {selected_claim['patient_age']}yo {selected_claim['gender']}")
            st.markdown(f"**Diagnosis:** {selected_claim['primary_diagnosis']}")
        with col2:
            st.markdown(f"**Provider:** {selected_claim['provider_type']}")
            st.markdown(f"**Billed Amount:** ${selected_claim['billed_amount']:.2f}")
            st.markdown("**Procedures:**")
            for proc in selected_claim["procedures"]:
                st.markdown(f"  - {proc}")

    if st.button("Analyze Claim", type="primary", key="sample", use_container_width=True):
        progress = st.progress(0, text="Initializing four-layer analysis pipeline...")

        claim_id = selected_claim["id"]

        if DEMO_MODE and claim_id in CACHED_RESULTS:
            import time
            progress.progress(10, text="Layer 1: Executing rules engine...")
            time.sleep(0.3)
            rules_result = CACHED_RESULTS[claim_id]["rules"]

            progress.progress(30, text="Layer 2: Retrieving clinical guidelines via RAG...")
            time.sleep(0.3)
            guidelines = CACHED_RESULTS[claim_id]["guidelines"]

            progress.progress(50, text="Layer 3: Running anomaly detection model...")
            time.sleep(0.3)
            anomaly_result = CACHED_RESULTS[claim_id]["anomaly"]

            progress.progress(65, text="Layer 4: Agentic GenAI performing chain reasoning...")
            time.sleep(0.5)
            llm_result = CACHED_RESULTS[claim_id]["llm"]
        else:
            progress.progress(10, text="Layer 1: Executing rules engine...")
            rules_result = run_rules_engine(selected_claim)

            progress.progress(30, text="Layer 2: Retrieving clinical guidelines via RAG...")
            guidelines = retrieve_relevant_guidelines(selected_claim)

            progress.progress(50, text="Layer 3: Running anomaly detection model...")
            anomaly_result = get_anomaly_score(selected_claim)

            progress.progress(65, text="Layer 4: Agentic GenAI performing chain reasoning...")
            llm_result = analyze_claim(selected_claim, guidelines=guidelines)

        progress.progress(100, text="Analysis complete!")

        display_pipeline_results(rules_result, anomaly_result, llm_result, guidelines=guidelines)

with tab2:
    st.subheader("Enter a Custom Clinical Scenario")

    if DEMO_MODE:
        st.warning("Custom scenario analysis requires AWS Bedrock credentials. Run locally with configured AWS profile to use this feature.")
    else:
        st.markdown("Describe a clinical claim scenario in natural language for AI analysis:")

    custom_input = st.text_area(
        "Describe the claim:",
        placeholder="Example: A 35-year-old female presents to the ER with a minor ankle sprain. "
                    "The provider bills for an MRI of the lumbar spine, full metabolic panel, "
                    "and a high-complexity ER visit (99285). Total billed: $8,500.",
        height=150,
    )

    if st.button("Analyze Scenario", type="primary", key="custom", use_container_width=True, disabled=DEMO_MODE):
        if custom_input.strip():
            with st.spinner("Agentic AI performing chain-of-thought analysis..."):
                llm_result = analyze_custom_claim(custom_input)

            st.markdown("---")
            st.subheader("GenAI Chain-of-Thought Analysis")
            st.caption("Custom scenarios use GenAI-only analysis (structured data required for rules/ML layers)")

            for step in llm_result.get("steps", []):
                with st.expander(f"Step {step.get('step_number', '?')}: {step.get('title', '')}", expanded=True):
                    st.markdown(f"**Reasoning:** {step.get('reasoning', '')}")
                    st.markdown(f"**Findings:** {step.get('findings', '')}")

            decision = llm_result.get("decision", "UNKNOWN")
            if decision == "APPROVE":
                st.success(f"### APPROVED")
            elif decision == "FLAG_FOR_REVIEW":
                st.warning(f"### FLAGGED FOR REVIEW")
            else:
                st.error(f"### DENIED")

            st.markdown(f"**Confidence:** {llm_result.get('confidence_score', 0):.0%}")
            st.markdown(f"**Recommendation:** {llm_result.get('recommendation', '')}")
        else:
            st.warning("Please enter a clinical scenario to analyze.")

with tab3:
    st.subheader("About This System")
    st.markdown("""
    ### Clinical Claims Decision Agent

    This proof-of-concept demonstrates how **Agentic Generative AI** can be applied to healthcare
    payment integrity operations. The system implements autonomous, goal-directed reasoning to
    analyze clinical claims through a multi-step decision pipeline.

    #### Architecture

    | Layer | Technology | Function |
    |-------|-----------|----------|
    | 1. Rules Engine | Custom Python | Deterministic CPT-ICD validation, cost thresholds |
    | 2. RAG Retrieval | AWS Titan Embeddings | Vector similarity search over clinical guidelines |
    | 3. Anomaly Detection | scikit-learn (Isolation Forest) | Unsupervised statistical outlier detection |
    | 4. Agentic GenAI | AWS Bedrock (Claude) | Chain-of-thought reasoning with RAG context |

    #### Key Capabilities
    - **Autonomous Decision-Making**: The agent independently reasons through 5 analytical steps
    - **Grounded Reasoning**: RAG ensures decisions reference actual clinical guidelines
    - **Explainability**: Every decision includes a complete audit trail and reasoning chain
    - **Human-in-the-Loop**: Designed to augment (not replace) human clinical auditors
    - **Novel Pattern Detection**: Unsupervised ML catches emerging fraud schemes

    #### AI Governance Considerations
    - All decisions are fully traceable and auditable
    - Human review required for DENY decisions before action
    - Model outputs include confidence scores for risk-based routing
    - No patient PII is stored or transmitted beyond the analysis session
    - System maintains separation between detection and enforcement

    ---
    *Topic 2: Clinical Decision Making and Pattern Recognition in Health Care*

    *Built by Nishit Patel | Arizona State University | Cotiviti GenAI Intern Assessment*
    """)
