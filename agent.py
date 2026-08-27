import os
import json
import streamlit as st
from google import genai
from google.genai import types
from tools import profile_dataset, execute_pipeline

# Resolve Gemini API Key from Streamlit Secrets or OS Environment
api_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.environ.get("GEMINI_API_KEY")

# Initialize the GenAI Client
client = genai.Client(api_key=api_key) if api_key else genai.Client()
MODEL_NAME = "gemini-2.5-flash"


def call_gemini_with_fallback(prompt: str) -> str:
    """Invokes Gemini with fallback options for model availability."""
    models_to_try = [MODEL_NAME, "gemini-1.5-flash", "gemini-1.5-pro"]
    last_error = None

    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            return response.text
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"All Gemini model calls failed. Last error: {last_error}")


def plan_initial_strategy(profile: dict) -> dict:
    """Stage 2: Plan - AI formulates the first ML execution plan."""
    prompt = f"""
You are ForensiQ, an autonomous ML forensic engineer.
Analyze the following dataset profile and provide an initial end-to-end classification strategy in JSON.

Dataset Profile:
{json.dumps(profile, indent=2)}

Requirements:
- Choose model_type: "logistic" or "random_forest" (use "logistic" first for baseline).
- imputation_strategy: "mean", "median", or "mode".
- scaling: true or false.
- handle_imbalance: "none" or "balanced".
- Provide a clear, concise forensic 'reasoning' string explaining your choices.

Return ONLY a JSON object matching this schema:
{{
  "model_type": "logistic",
  "imputation_strategy": "median",
  "scaling": true,
  "handle_imbalance": "none",
  "reasoning": "string explanation"
}}
"""
    raw_response = call_gemini_with_fallback(prompt)
    return json.loads(raw_response)


def reflect_and_revise(profile: dict, previous_strategy: dict, metrics: dict, threshold: float) -> dict:
    """Stage 4: Reflect & Self-Correct - AI critiques results and devises a revised strategy."""
    prompt = f"""
You are ForensiQ, an autonomous ML forensic engineer performing self-reflection.
The previous iteration failed to achieve the target F1 score threshold of {threshold}.

Dataset Profile:
{json.dumps(profile, indent=2)}

Previous Strategy:
{json.dumps(previous_strategy, indent=2)}

Observed Evaluation Metrics:
{json.dumps(metrics, indent=2)}

Analyze why the previous model performed poorly (e.g., class imbalance, non-linearity, unscaled features).
Formulate a revised, superior strategy to cross the target F1 threshold.

Return ONLY a JSON object matching this schema:
{{
  "critique": "Forensic critique of why previous strategy underperformed",
  "model_type": "random_forest",
  "imputation_strategy": "median",
  "scaling": true,
  "handle_imbalance": "balanced",
  "reasoning": "Detailed justification for the revised adjustments"
}}
"""
    raw_response = call_gemini_with_fallback(prompt)
    return json.loads(raw_response)


def run_forensiq(df, target_column: str, target_f1_threshold: float = 0.80):
    """
    Main Autonomous Loop (Generator yielding trace logs for real-time Streamlit streaming).
    Lifecycle: Investigate -> Plan -> Act -> Reflect -> Self-Correct -> Finalize
    """
    # 1. Investigate / Diagnostic Profiling
    yield {"status": "investigating", "message": "🔍 Profiling dataset and investigating anomalies..."}
    profile = profile_dataset(df, target_column)
    yield {
        "status": "profile_complete",
        "profile": profile,
        "message": f"📊 Profile: {profile['total_rows']} rows × {profile['total_cols']} cols. Issues: Missing values across {len(profile['missing_summary'])} column(s) | {profile['duplicate_rows']} duplicate row(s) | Target balance: {json.dumps(profile['target_distribution'])}"
    }

    # 2. Plan (Iteration 1)
    yield {"status": "planning_v1", "message": "🧠 AI Planner: Formulating initial data treatment strategy..."}
    plan_v1 = plan_initial_strategy(profile)
    yield {
        "status": "plan_v1_ready",
        "strategy": plan_v1,
        "message": f"📋 Strategy 1: Model={plan_v1.get('model_type').upper()} | Scaler={plan_v1.get('scaling')} | ClassWeight={plan_v1.get('handle_imbalance')}\nReason: {plan_v1.get('reasoning')}"
    }

    # 3. Act (Iteration 1 Execution)
    yield {"status": "executing_v1", "message": "⚙️ Executing Pipeline (Iteration 1)..."}
    pipeline_v1, metrics_v1 = execute_pipeline(df, target_column, plan_v1)
    
    execution_history = [
        {"iteration": 1, "strategy": plan_v1, "metrics": metrics_v1}
    ]
    
    yield {
        "status": "iteration_1_complete",
        "metrics": metrics_v1,
        "message": f"📊 Iteration 1 Results: F1={metrics_v1['f1_score']} | Acc={metrics_v1['accuracy']} | Prec={metrics_v1['precision']} | Rec={metrics_v1['recall']}"
    }

    best_pipeline = pipeline_v1
    best_metrics = metrics_v1
    best_iteration = 1

    # 4. Reflect & Self-Correction Trigger
    if metrics_v1["f1_score"] < target_f1_threshold:
        yield {
            "status": "reflecting",
            "message": f"🤕 Reflection: F1 Score ({metrics_v1['f1_score']}) is below threshold ({target_f1_threshold}). Autonomously revising strategy..."
        }
        
        reflection_result = reflect_and_revise(profile, plan_v1, metrics_v1, target_f1_threshold)
        
        plan_v2 = {
            "model_type": reflection_result.get("model_type", "random_forest"),
            "imputation_strategy": reflection_result.get("imputation_strategy", "median"),
            "scaling": reflection_result.get("scaling", True),
            "handle_imbalance": reflection_result.get("handle_imbalance", "balanced"),
            "reasoning": reflection_result.get("reasoning", "Revised via reflection loop")
        }
        
        yield {
            "status": "plan_v2_ready",
            "strategy": plan_v2,
            "critique": reflection_result.get("critique"),
            "message": f"💡 Strategy 2: Model={plan_v2['model_type'].upper()} | ClassWeight={plan_v2['handle_imbalance']}\nAdjustment: {reflection_result.get('critique', plan_v2['reasoning'])}"
        }

        # 5. Act (Iteration 2 Execution)
        yield {"status": "executing_v2", "message": "⚙️ Retraining with revised strategy (Iteration 2)..."}
        pipeline_v2, metrics_v2 = execute_pipeline(df, target_column, plan_v2)
        
        execution_history.append({"iteration": 2, "strategy": plan_v2, "metrics": metrics_v2})
        
        yield {
            "status": "iteration_2_complete",
            "metrics": metrics_v2,
            "message": f"📊 Iteration 2 Results: F1={metrics_v2['f1_score']} | Acc={metrics_v2['accuracy']} | Prec={metrics_v2['precision']} | Rec={metrics_v2['recall']}"
        }

        if metrics_v2["f1_score"] >= metrics_v1["f1_score"]:
            diff = round(metrics_v2["f1_score"] - metrics_v1["f1_score"], 4)
            best_pipeline = pipeline_v2
            best_metrics = metrics_v2
            best_iteration = 2
            yield {
                "status": "improvement_confirmed",
                "message": f"✅ Performance improved (+{diff} F1). Retaining Iteration 2 pipeline."
            }
        else:
            yield {
                "status": "retaining_v1",
                "message": "⚠️ Iteration 2 did not improve score. Retaining Iteration 1 pipeline."
            }

    # 6. Finalize & Package Artifacts
    yield {
        "status": "finalized",
        "best_iteration": best_iteration,
        "best_pipeline": best_pipeline,
        "best_metrics": best_metrics,
        "execution_history": execution_history,
        "profile": profile,
        "message": "🏆 Investigation & ML Pipeline Finalized."
    }