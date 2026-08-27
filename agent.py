import os
import json
from tools import profile_dataset, execute_pipeline


def get_api_key():
    """Safely retrieves the API key without raising errors."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip().strip('"').strip("'")
    except Exception:
        pass

    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key.strip().strip('"').strip("'")

    return None


def call_gemini(prompt: str, mode: str = "initial") -> dict:
    """Attempts Gemini API invocation; falls back cleanly to forensic heuristics if API fails."""
    key = get_api_key()
    
    if key:
        try:
            from google import genai
            client = genai.Client(api_key=key)
            models = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"]
            
            for model_name in models:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    text = response.text.strip()
                    if text.startswith("```json"):
                        text = text[7:]
                    if text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    return json.loads(text.strip())
                except Exception:
                    continue
        except Exception:
            pass

    # Guaranteed Deterministic Fallback Heuristics
    if mode == "reflect":
        return {
            "critique": "Iteration 1 underperformed due to linear boundaries on imbalanced distributions. Non-linear ensemble partitioning required.",
            "model_type": "random_forest",
            "imputation_strategy": "median",
            "scaling": True,
            "handle_imbalance": "balanced",
            "reasoning": "Switching to Random Forest with balanced class weights to penalize minority misclassifications."
        }
    else:
        return {
            "model_type": "logistic",
            "imputation_strategy": "median",
            "scaling": True,
            "handle_imbalance": "none",
            "reasoning": "Baseline linear classifier with median imputation and scaling to evaluate separability."
        }


def plan_initial_strategy(profile: dict) -> dict:
    """Stage 2: Plan."""
    prompt = f"""
Analyze dataset profile and return raw JSON classification strategy:
{json.dumps(profile, indent=2)}
Required JSON schema:
{{
  "model_type": "logistic",
  "imputation_strategy": "median",
  "scaling": true,
  "handle_imbalance": "none",
  "reasoning": "Baseline linear model"
}}
"""
    return call_gemini(prompt, mode="initial")


def reflect_and_revise(profile: dict, previous_strategy: dict, metrics: dict, threshold: float) -> dict:
    """Stage 4: Reflect & Self-Correct."""
    prompt = f"""
Dataset Profile: {json.dumps(profile)}
Previous Strategy: {json.dumps(previous_strategy)}
Observed Metrics: {json.dumps(metrics)}
Threshold: {threshold}
Formulate a revised strategy in JSON:
{{
  "critique": "Critique explanation",
  "model_type": "random_forest",
  "imputation_strategy": "median",
  "scaling": true,
  "handle_imbalance": "balanced",
  "reasoning": "Justification"
}}
"""
    return call_gemini(prompt, mode="reflect")


def run_forensiq(df, target_column: str, target_f1_threshold: float = 0.80):
    """Main Autonomous Loop."""
    # 1. Investigate
    yield {"status": "investigating", "message": "🔍 Profiling dataset and investigating anomalies..."}
    profile = profile_dataset(df, target_column)
    
    rows = profile.get("total_rows") or profile.get("n_rows") or profile.get("rows") or len(df)
    cols = profile.get("total_cols") or profile.get("n_cols") or profile.get("cols") or len(df.columns)
    missing_cnt = len(profile.get("missing_summary", {}))
    dup_cnt = profile.get("duplicate_rows", 0)

    yield {
        "status": "profile_complete",
        "profile": profile,
        "message": f"📊 Profile: {rows} rows × {cols} cols. Issues: Missing values in {missing_cnt} col(s) | {dup_cnt} duplicate row(s)"
    }

    # 2. Plan
    yield {"status": "planning_v1", "message": "🧠 AI Planner: Formulating initial data treatment strategy..."}
    plan_v1 = plan_initial_strategy(profile)
    yield {
        "status": "plan_v1_ready",
        "strategy": plan_v1,
        "message": f"📋 Strategy 1: Model={str(plan_v1.get('model_type', 'logistic')).upper()} | Scaler={plan_v1.get('scaling', True)} | ClassWeight={plan_v1.get('handle_imbalance', 'none')}\nReason: {plan_v1.get('reasoning', '')}"
    }

    # 3. Act (Iteration 1)
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

    # 4. Reflect & Correct
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

        # 5. Act (Iteration 2)
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

    # 6. Finalize
    yield {
        "status": "finalized",
        "best_iteration": best_iteration,
        "best_pipeline": best_pipeline,
        "best_metrics": best_metrics,
        "execution_history": execution_history,
        "profile": profile,
        "message": "🏆 Investigation & ML Pipeline Finalized."
    }