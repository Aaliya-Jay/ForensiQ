import io
import json
import joblib
import pandas as pd
import streamlit as st
from agent import run_forensiq

st.set_page_config(
    page_title="ForensiQ | Autonomous ML Agent",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ ForensiQ — Autonomous ML Forensic Agent")
st.markdown("Automated diagnostic inspection, baseline execution, self-reflection, and iterative optimization.")

# Sidebar Settings
st.sidebar.header("⚙️ Agent Settings")
target_f1 = st.sidebar.slider("Target Minimum F1 Score", min_value=0.50, max_value=0.99, value=0.80, step=0.01)

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    
    target_column = st.selectbox("🎯 Select Target Column to Predict", df.columns.tolist())
    
    if st.button("🚀 Launch Autonomous Investigation", type="primary"):
        st.subheader("🕵️ Autonomous Agent Execution Activity")
        
        log_placeholder = st.empty()
        logs = []
        final_data = None
        
        for event in run_forensiq(df, target_column, target_f1_threshold=target_f1):
            if "message" in event:
                logs.append(event["message"])
                log_placeholder.markdown(
                    f"```text\n" + "\n".join(logs) + "\n```"
                )
            if event.get("status") == "finalized":
                final_data = event

        if final_data:
            st.success("✨ Investigation & Pipeline Execution Complete!")
            
            best_metrics = final_data["best_metrics"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final F1 Score", best_metrics["f1_score"])
            col2.metric("Accuracy", best_metrics["accuracy"])
            col3.metric("Precision", best_metrics["precision"])
            col4.metric("Recall", best_metrics["recall"])
            
            tab1, tab2, tab3 = st.tabs(["📜 Execution History", "🔍 Data Profile Diagnostics", "💾 Export Artifacts"])
            
            with tab1:
                st.subheader("Pipeline Iteration Evolution")
                for hist in final_data["execution_history"]:
                    with st.expander(f"Iteration {hist['iteration']} (F1: {hist['metrics']['f1_score']})", expanded=(hist['iteration'] == final_data['best_iteration'])):
                        st.json(hist)
            
            with tab2:
                st.subheader("Automated Diagnostic Profile")
                st.json(final_data["profile"])
                
            with tab3:
                st.subheader("Download Production Artifacts")
                
                # Model Buffer
                model_buffer = io.BytesIO()
                joblib.dump(final_data["best_pipeline"], model_buffer)
                model_buffer.seek(0)
                
                # Audit JSON Buffer
                audit_report = {
                    "best_iteration": final_data["best_iteration"],
                    "metrics": final_data["best_metrics"],
                    "history": final_data["execution_history"],
                    "profile": final_data["profile"]
                }
                json_buffer = io.StringIO()
                json.dump(audit_report, json_buffer, indent=2)
                
                st.download_button(
                    label="📥 Download Best Trained Model (.joblib)",
                    data=model_buffer,
                    file_name="forensiq_best_model.joblib",
                    mime="application/octet-stream"
                )
                
                st.download_button(
                    label="📄 Download Forensic Audit Report (JSON)",
                    data=json_buffer.getvalue(),
                    file_name="forensic_audit_report.json",
                    mime="application/json"
                )