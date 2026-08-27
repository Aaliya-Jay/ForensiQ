import io
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def profile_dataset(df: pd.DataFrame, target_col: str) -> dict:
    """Investigates data quality issues, column types, missingness, duplicates, and class imbalance."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # Guard check: Prevent using unique ID columns as target
    if df[target_col].nunique() > 50:
        raise ValueError(f"Column '{target_col}' has too many unique values ({df[target_col].nunique()}) to be used as a classification target. Please select a categorical or binary target (like 'churn').")

    duplicate_rows = int(df.duplicated().sum())
    clean_df = df.dropna(subset=[target_col]).copy()
    features = clean_df.drop(columns=[target_col])

    # Filter out datetime / high-cardinality text columns automatically
    date_cols = [
        c for c in features.columns
        if features[c].dtype == "object"
        and pd.to_datetime(features[c], errors="coerce").notnull().mean() > 0.8
    ]
    features = features.drop(columns=date_cols)

    num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in features.select_dtypes(exclude=[np.number]).columns if features[c].nunique() <= 50]
    high_card_cols = [c for c in features.select_dtypes(exclude=[np.number]).columns if features[c].nunique() > 50]

    target_dist = clean_df[target_col].value_counts(normalize=True).to_dict()
    is_imbalanced = any(r > 0.70 for r in target_dist.values()) if target_dist else False

    return {
        "shape": {"rows": len(clean_df), "cols": features.shape[1] + 1},
        "missing_counts": clean_df.isnull().sum()[lambda x: x > 0].to_dict(),
        "duplicate_rows": duplicate_rows,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "high_cardinality_cols": high_card_cols,
        "dropped_unusable_cols": date_cols + high_card_cols,
        "target_col": target_col,
        "is_imbalanced": is_imbalanced,
        "class_distribution": {str(k): round(v, 4) for k, v in target_dist.items()}
    }


def execute_pipeline(df: pd.DataFrame, target_col: str, strategy: dict) -> dict:
    """Executes preprocessing, model training, and metric evaluation according to agent strategy."""
    working_df = df.drop_duplicates().dropna(subset=[target_col]).copy()

    # 1. Encode Target into clean integers
    le = LabelEncoder()
    y = le.fit_transform(working_df[target_col].astype(str))
    X = working_df.drop(columns=[target_col])

    # 2. Drop high-cardinality / date columns requested by strategy
    drop_cols = strategy.get("drop_cols", [])
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.select_dtypes(exclude=[np.number]).columns if X[c].nunique() <= 50]
    X = X[num_cols + cat_cols]

    # 3. Build dynamic preprocessing transformers
    transformers = []
    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy=strategy.get("num_imputer", "median")))]
        if strategy.get("scale", True):
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), num_cols))

    if cat_cols:
        cat_steps = [
            ("imputer", SimpleImputer(strategy=strategy.get("cat_imputer", "most_frequent"))),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
        transformers.append(("cat", Pipeline(cat_steps), cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # 4. Model Selection
    model_type = strategy.get("model_type", "logistic")
    class_weight = strategy.get("class_weight", None)

    if model_type == "rf":
        clf = RandomForestClassifier(n_estimators=100, class_weight=class_weight, random_state=42)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42)

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

    # Safe Stratification: Only stratify if every class has at least 2 samples
    class_counts = pd.Series(y).value_counts()
    can_stratify = (class_counts >= 2).all() and len(class_counts) > 1
    strat = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strat
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    avg_mode = "binary" if len(np.unique(y)) == 2 else "weighted"
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "metrics": {
            "f1_score": round(float(f1_score(y_test, y_pred, average=avg_mode, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, average=avg_mode, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, average=avg_mode, zero_division=0)), 4)
        },
        "confusion_matrix": cm,
        "strategy": strategy,
        "features_used": {"num": num_cols, "cat": cat_cols},
        "model_pipeline": pipeline
    }