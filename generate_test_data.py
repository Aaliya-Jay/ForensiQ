"""
Generates 3 synthetic datasets representing distinct real-world data issues:
1. messy_missing.csv: Missing numeric and categorical data + duplicate rows.
2. imbalanced_churn.csv: High class imbalance (90/10) with categorical features.
3. noisy_ecommerce.csv: High cardinality IDs, outliers, and mixed features.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1000

# --- Dataset 1: Missing values & Duplicates ---
df1 = pd.DataFrame({
    "age": np.random.choice([np.nan, 22, 35, 48, 60, 29, 41], size=n_samples),
    "income": np.random.choice([np.nan, 30000, 55000, 80000, 120000, 45000], size=n_samples),
    "city": np.random.choice([np.nan, "New York", "London", "Tokyo", "Berlin"], size=n_samples),
    "credit_score": np.random.normal(650, 50, size=n_samples),
    "approved": np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
})
# Inject 50 duplicate rows
df1 = pd.concat([df1, df1.iloc[:50]], ignore_index=True)
df1.to_csv("test_dataset_missing.csv", index=False)
print("✅ Created test_dataset_missing.csv")

# --- Dataset 2: Imbalanced Churn ---
df2 = pd.DataFrame({
    "tenure_months": np.random.randint(1, 72, size=n_samples),
    "monthly_charges": np.random.uniform(20.0, 120.0, size=n_samples),
    "contract_type": np.random.choice(["Month-to-Month", "One Year", "Two Year"], size=n_samples),
    "payment_method": np.random.choice(["Electronic", "Mailed Check", "Bank Transfer"], size=n_samples),
    "churn": np.random.choice([0, 1], size=n_samples, p=[0.88, 0.12])  # 12% minority
})
df2.to_csv("test_dataset_imbalanced.csv", index=False)
print("✅ Created test_dataset_imbalanced.csv")

# --- Dataset 3: Noisy Ecommerce with High Cardinality ID ---
df3 = pd.DataFrame({
    "user_id_hash": [f"USR_{np.random.randint(100000, 999999)}" for _ in range(n_samples)],
    "device": np.random.choice(["Mobile", "Desktop", "Tablet"], size=n_samples),
    "page_views": np.random.poisson(lam=5, size=n_samples),
    "session_duration_sec": np.random.exponential(scale=180, size=n_samples),
    "converted": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
})
df3.to_csv("test_dataset_noisy.csv", index=False)
print("✅ Created test_dataset_noisy.csv")