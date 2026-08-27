import io
import json
import joblib
import pandas as pd
import streamlit as st
from agent import run_forensiq

st.set_page_config(page_title="ForensiQ | Autonomous ML Agent", page_icon="🕵️", layout="wide")

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        padding: 1.2rem;
        border-radius: 0.5rem;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-title { font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.3rem; }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #38bdf8; }
</style>
""", unsafe_allow_html=True)

st.title("🕵️ ForensiQ — Autonomous Data Forensics & ML Agent")
st.caption("Team **AgentForge** | Track A: Autonomous ML & Auto-EDA")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Agent Settings")
    target_f1 = st.slider("Target Minimum F1 Score", min_value=0.50, max_value=0.95, value=0.80, step=0.05)
    st.info("If the initial model falls below this F1 threshold, the agent will autonomously reflect and execute a self-correcting retry.")

# Main Interface
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview")
    st.dataframe(df.head(5), use_container_width=True)

    target_column = st.selectbox("🎯 Select Target Column to Predict", options=df.columns)

    if st.button("🚀 Launch Autonomous Investigation", type="primary"):
        st.write("### 🕵️ Autonomous Agent Execution Activity")
        log_container = st.empty()
        logs = []

        final_data = None
        for event in run_forensiq(df, target_column, target_f1_threshold=target_f1):
            logs.append(event["message"])
            log_container.code("\n".join(logs), language="markdown")
            if event["status"] == "complete":
                final_data = event

        if final_data:
            st.success("✨ Investigation & Pipeline Execution Complete!")
            res = final_data["final_result"]
            metrics = res["metrics"]

            # Key Metric Cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Final F1 Score</div><div class="metric-val">{metrics["f1_score"]}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Accuracy</div><div class="metric-val">{metrics["accuracy"]}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Precision</div><div class="metric-val">{metrics["precision"]}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Recall</div><div class="metric-val">{metrics["recall"]}</div></div>', unsafe_allow_html=True)

            st.write("---")
            tab1, tab2, tab3 = st.tabs(["📜 Execution History", "🔍 Data Profile Diagnostics", "💾 Export Artifacts"])

            with tab1:
                st.subheader("Pipeline Iteration Evolution")
                for i, hist in enumerate(final_data["history"]):
                    with st.expander(f"Iteration {i+1} (F1: {hist['metrics']['f1_score']})", expanded=(i == len(final_data['history'])-1)):
                        st.json({
                            "metrics": hist["metrics"],
                            "strategy": hist["strategy"],
                            "features_used": hist["features_used"],
                            "confusion_matrix": hist.get("confusion_matrix")
                        })

            with tab2:
                st.subheader("Dataset Diagnostics")
                st.json(final_data["profile"])

            with tab3:
                st.subheader("Download Production Artifacts")
                # 1. Download Model Object
                if "model_pipeline" in res:
                    model_buffer = io.BytesIO()
                    joblib.dump(res["model_pipeline"], model_buffer)
                    model_buffer.seek(0)
                    st.download_button(
                        label="📥 Download Best Trained Model (.joblib)",
                        data=model_buffer,
                        file_name="forensiq_best_model.joblib",
                        mime="application/octet-stream"
                    )

                # 2. Download JSON Forensic Report
                export_report = {
                    "team": "AgentForge",
                    "diagnostics": final_data["profile"],
                    "winning_strategy": res["strategy"],
                    "final_metrics": res["metrics"],
                    "iteration_count": len(final_data["history"])
                }
                st.download_button(
                    label="📄 Download Forensic Audit Report (JSON)",
                    data=json.dumps(export_report, indent=2),
                    file_name="forensiq_audit_report.json",
                    mime="application/json"
                )