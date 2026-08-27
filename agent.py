import json
import os
import pandas as pd
from google import genai
from google.genai import types
from tools import profile_dataset, execute_pipeline

# Initialize Google GenAI client
client = genai.Client()
MODEL_NAME = "gemini-3.6-flash"


def plan_initial_strategy(profile: dict) -> dict:
    """Uses Gemini to plan the initial data treatment strategy."""
    prompt = f"""
You are the Planning Core of ForensiQ, an autonomous data forensics & ML agent.
Analyze this dataset profile:
{json.dumps(profile, indent=2)}

Generate an initial preprocessing and modeling plan.
Respond with a STRICT JSON object containing these exact keys:
{{
  "num_imputer": "mean" | "median",
  "cat_imputer": "most_frequent",
  "scale": true | false,
  "drop_cols": list of strings,
  "model_type": "logistic" | "rf",
  "class_weight": null | "balanced",
  "reasoning": "1-2 sentences explaining why this strategy fits the data profile"
}}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    return json.loads(response.text)


def reflect_and_revise_strategy(profile: dict, previous_strategy: dict, previous_metrics: dict) -> dict:
    """Critiques the previous run and autonomously formulates an improved strategy."""
    prompt = f"""
You are the Reflection & Self-Correction Engine of ForensiQ.
The previous iteration underperformed.

Dataset Profile:
{json.dumps(profile, indent=2)}

Previous Strategy:
{json.dumps(previous_strategy, indent=2)}

Previous Metrics:
{json.dumps(previous_metrics, indent=2)}

Decide on corrective actions (e.g. switch to 'rf', enable class_weight="balanced", change imputation, or drop noisy columns).
Respond with a STRICT JSON object containing:
{{
  "num_imputer": "mean" | "median",
  "cat_imputer": "most_frequent",
  "scale": true | false,
  "drop_cols": list of strings,
  "model_type": "logistic" | "rf",
  "class_weight": null | "balanced",
  "reasoning": "1-2 sentences explaining the concrete corrective adjustment made"
}}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    return json.loads(response.text)


def run_forensiq(df: pd.DataFrame, target_col: str, target_f1_threshold: float = 0.75):
    """
    Autonomous Agent Loop: Investigate -> Plan -> Act -> Observe -> Reflect -> Correct.
    """
    # 1. Investigate
    yield {"status": "investigating", "message": "🔎 Profiling dataset and investigating anomalies..."}
    profile = profile_dataset(df, target_col)
    
    anomalies = []
    if profile.get("missing_counts"):
        anomalies.append(f"Missing values across {len(profile['missing_counts'])} column(s)")
    if profile.get("duplicate_rows", 0) > 0:
        anomalies.append(f"{profile['duplicate_rows']} duplicate row(s)")
    if profile.get("is_imbalanced"):
        anomalies.append("Severe target class imbalance detected")
    if profile.get("high_cardinality_cols"):
        anomalies.append(f"High cardinality column(s): {', '.join(profile['high_cardinality_cols'])}")

    anomaly_text = " | ".join(anomalies) if anomalies else "Clean distribution"
    yield {
        "status": "profile_ready",
        "message": f"📊 Profile: {profile['shape']['rows']} rows × {profile['shape']['cols']} cols. Issues: {anomaly_text}",
        "data": profile
    }

    # 2. Plan Initial Strategy
    yield {"status": "planning", "message": "🧠 AI Planner: Formulating initial data treatment strategy..."}
    plan_v1 = plan_initial_strategy(profile)
    yield {
        "status": "plan_ready",
        "message": f"📋 Strategy 1: Model={plan_v1['model_type'].upper()} | Scaler={plan_v1['scale']} | ClassWeight={plan_v1['class_weight']}\n   Reason: {plan_v1.get('reasoning')}",
        "strategy": plan_v1
    }

    # 3. Act & Train (Iteration 1)
    yield {"status": "executing", "message": "🛠️ Executing Pipeline (Iteration 1)..."}
    res_v1 = execute_pipeline(df, target_col, plan_v1)
    f1_v1 = res_v1["metrics"]["f1_score"]
    yield {
        "status": "eval_ready",
        "message": f"📊 Iteration 1 Results: F1={f1_v1} | Acc={res_v1['metrics']['accuracy']} | Prec={res_v1['metrics']['precision']} | Rec={res_v1['metrics']['recall']}",
        "metrics": res_v1["metrics"]
    }

    final_result = res_v1
    iteration_history = [res_v1]

    # 4. Reflect & Self-Correct (Iteration 2)
    if f1_v1 < target_f1_threshold:
        yield {
            "status": "reflecting",
            "message": f"🤔 Reflection: F1 Score ({f1_v1}) is below threshold ({target_f1_threshold}). Autonomously revising strategy..."
        }
        
        plan_v2 = reflect_and_revise_strategy(profile, plan_v1, res_v1["metrics"])
        yield {
            "status": "plan_revised",
            "message": f"🔄 Strategy 2: Model={plan_v2['model_type'].upper()} | ClassWeight={plan_v2['class_weight']}\n   Adjustment: {plan_v2.get('reasoning')}",
            "strategy": plan_v2
        }

        yield {"status": "retrying", "message": "🛠️ Retraining with revised strategy (Iteration 2)..."}
        res_v2 = execute_pipeline(df, target_col, plan_v2)
        f1_v2 = res_v2["metrics"]["f1_score"]
        iteration_history.append(res_v2)

        yield {
            "status": "eval_ready_v2",
            "message": f"📊 Iteration 2 Results: F1={f1_v2} | Acc={res_v2['metrics']['accuracy']} | Prec={res_v2['metrics']['precision']} | Rec={res_v2['metrics']['recall']}",
            "metrics": res_v2["metrics"]
        }

        if f1_v2 >= f1_v1:
            yield {"status": "improved", "message": f"✅ Performance improved (+{round(f1_v2 - f1_v1, 4)} F1). Retaining Iteration 2 pipeline."}
            final_result = res_v2
        else:
            yield {"status": "retained", "message": "⚠️ Revised strategy did not yield gain. Retaining Iteration 1 pipeline."}
            final_result = res_v1
    else:
        yield {"status": "threshold_met", "message": f"✅ Initial model met target performance threshold (F1: {f1_v1} ≥ {target_f1_threshold})."}

    yield {
        "status": "complete",
        "message": "🏆 Investigation & ML Pipeline Finalized.",
        "final_result": final_result,
        "history": iteration_history,
        "profile": profile
    }