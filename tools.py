import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def profile_dataset(df: pd.DataFrame, target_col: str) -> dict:
    """Profiles the dataset and flags potential ML data quality issues."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    missing_summary = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
    duplicate_rows = int(df.duplicated().sum())

    target_counts = df[target_col].value_counts(dropna=False).to_dict()
    target_counts_clean = {str(k): int(v) for k, v in target_counts.items()}

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    categorical_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c != target_col]

    return {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "missing_summary": missing_summary,
        "duplicate_rows": duplicate_rows,
        "target_distribution": target_counts_clean,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
    }


def execute_pipeline(df: pd.DataFrame, target_col: str, strategy: dict):
    """Executes preprocessing and training according to the planned strategy."""
    clean_df = df.copy()

    # Drop non-predictive ID columns if present
    id_cols = [c for c in clean_df.columns if "id" in c.lower() and c != target_col]
    if id_cols:
        clean_df = clean_df.drop(columns=id_cols)

    clean_df = clean_df.dropna(subset=[target_col])

    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]

    # Convert binary categorical targets to numeric
    if y.dtype == "object" or str(y.dtype) == "category":
        classes = list(y.unique())
        if len(classes) == 2:
            y = y.map({classes[0]: 0, classes[1]: 1})
        else:
            y = pd.factorize(y)[0]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocessing
    imp_strategy = strategy.get("imputation_strategy", "median")
    if imp_strategy not in ["mean", "median", "most_frequent"]:
        imp_strategy = "median"

    num_steps = [("imputer", SimpleImputer(strategy=imp_strategy))]
    if strategy.get("scaling", True):
        num_steps.append(("scaler", StandardScaler()))

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols:
        transformers.append(("cat", SimpleImputer(strategy="most_frequent"), cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Model definition
    model_type = strategy.get("model_type", "logistic")
    class_weight = strategy.get("handle_imbalance", "none")
    cw_param = "balanced" if class_weight == "balanced" else None

    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=cw_param)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight=cw_param)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    # Train / Test split with safe stratification fallback
    test_size = 0.2 if len(X) >= 50 else 0.3
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
    }

    return pipeline, metrics