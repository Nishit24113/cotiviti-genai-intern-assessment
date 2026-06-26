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
    ### Pattern Analytics
    - K-Means Clustering (risk profiles)
    - Time-Series Anomaly Detection

    ---
    ### Technologies
    - AWS Bedrock (Claude)
    - AWS Titan Embeddings
    - scikit-learn
    - Streamlit | Plotly
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
st.markdown("### Pipeline Components")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Knowledge Base", "10 Guidelines", delta="Titan Embeddings")
with col2:
    st.metric("Anomaly Model", "Isolation Forest", delta="Unsupervised ML")
with col3:
    st.metric("Rules Engine", "5 Rule Types", delta="CPT-ICD Checks")
with col4:
    mode_label = "Demo Mode" if DEMO_MODE else "Live (Bedrock)"
    st.metric("LLM Agent", "Claude", delta=mode_label)

st.markdown("---")

# --- Main Content ---
tab1, tab2, tab3, tab4 = st.tabs(["📋 Claim Analysis", "📊 Batch Analysis", "✏️ Custom Scenario", "ℹ️ About"])

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
    st.subheader("Batch Claims Analysis")
    st.markdown("Analyze all sample claims simultaneously and view aggregate statistics.")

    from batch_processor import process_batch, get_batch_summary
    from pattern_analytics import cluster_claims, detect_time_series_anomalies
    import plotly.express as px
    import plotly.graph_objects as go

    if st.button("Run Batch Analysis", type="primary", key="batch", use_container_width=True):
        with st.spinner("Processing all claims through rules + anomaly detection..."):
            df = process_batch(SAMPLE_CLAIMS)
            summary = get_batch_summary(df)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Claims", summary["total_claims"])
        with col2:
            st.metric("Flagged", summary["flagged_claims"],
                      delta=f"{summary['flag_rate']:.0%} rate", delta_color="inverse")
        with col3:
            st.metric("Outliers", summary["outlier_claims"],
                      delta=f"{summary['outlier_rate']:.0%} rate", delta_color="inverse")
        with col4:
            st.metric("Avg Risk", f"{summary['avg_risk_score']:.0%}")

        st.markdown("---")

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                df, x="Claim ID", y="Rules Risk",
                color="Rules Risk",
                color_continuous_scale=["green", "yellow", "red"],
                title="Risk Score by Claim",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                df, x="Billed Amount", y="Anomaly Score",
                color="Is Outlier",
                size="Procedures",
                hover_data=["Claim ID", "Diagnosis"],
                title="Anomaly Score vs. Billed Amount",
                color_discrete_map={True: "red", False: "green"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        # --- K-Means Clustering ---
        st.markdown("---")
        st.subheader("K-Means Clustering: Risk Profiles")
        st.caption("Unsupervised clustering groups claims into risk categories based on billing features")

        cluster_df = cluster_claims(SAMPLE_CLAIMS)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(
                cluster_df, x="Billed Amount", y="Age",
                color="Cluster", size="Procedures",
                hover_data=["Claim ID"],
                title="Claims Clustered by Risk Profile",
                color_discrete_map={"Low Risk": "#28a745", "Moderate Risk": "#ffc107", "High Risk": "#dc3545"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            cluster_counts = cluster_df["Cluster"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster", "Count"]
            fig = px.pie(
                cluster_counts, values="Count", names="Cluster",
                title="Cluster Distribution",
                color="Cluster",
                color_discrete_map={"Low Risk": "#28a745", "Moderate Risk": "#ffc107", "High Risk": "#dc3545"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        # --- Time-Series Anomaly Detection ---
        st.markdown("---")
        st.subheader("Time-Series Anomaly Detection")
        st.caption("Rolling Z-score method detects billing volume spikes that may indicate fraud onset or system abuse")

        ts_df = detect_time_series_anomalies()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts_df["day"], y=ts_df["count"],
            mode="lines+markers", name="Daily Claims",
            line=dict(color="#667eea"),
        ))
        fig.add_trace(go.Scatter(
            x=ts_df["day"], y=ts_df["rolling_mean"],
            mode="lines", name="Rolling Average",
            line=dict(color="#999", dash="dash"),
        ))

        anomaly_days = ts_df[ts_df["combined_anomaly"]]
        fig.add_trace(go.Scatter(
            x=anomaly_days["day"], y=anomaly_days["count"],
            mode="markers", name="Anomaly Detected",
            marker=dict(color="red", size=14, symbol="x"),
        ))
        fig.update_layout(
            title="Daily Claims Volume with Anomaly Detection (30-Day Window)",
            xaxis_title="Day", yaxis_title="Claims Count",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            anomaly_count = len(anomaly_days)
            st.metric("Anomalous Days Detected", anomaly_count,
                      delta=f"{anomaly_count/30:.0%} of period")
        with col2:
            if len(anomaly_days) > 0:
                peak_day = anomaly_days.loc[anomaly_days["count"].idxmax()]
                st.metric("Peak Anomaly", f"Day {int(peak_day['day'])}",
                          delta=f"{int(peak_day['count'])} claims ({peak_day['count']/ts_df['count'].mean():.1f}x avg)")

        # Results table
        st.markdown("---")
        st.markdown("### Detailed Results")
        styled_df = df.style.background_gradient(
            subset=["Rules Risk", "Anomaly Score"],
            cmap="RdYlGn_r",
        )
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Export
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="claims_analysis_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

with tab3:
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

with tab4:
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

    #### Topic Coverage (Topic 2 Keywords)
    | Keyword | Implementation |
    |---------|---------------|
    | Chain Reasoning | 5-step chain-of-thought clinical analysis |
    | Agentic GenAI | Autonomous goal-directed LLM agent |
    | Classification | Approve / Flag for Review / Deny decisions |
    | Prediction & Inference | Confidence scoring + risk estimation |
    | Clustering | K-Means risk profile grouping |
    | Time-Series Anomaly | Rolling Z-score billing volume detection |
    | Anomaly Detection | Isolation Forest unsupervised outlier detection |

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
