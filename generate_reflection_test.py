import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 300

# 1. Generate Non-Linear Features
feature_x = np.random.uniform(-3, 3, n_samples)
feature_y = np.random.uniform(-3, 3, n_samples)
spending = np.random.exponential(scale=50, size=n_samples)
account_age_months = np.random.randint(1, 72, size=n_samples)

latent_score = (feature_x * feature_y) + (spending > 60).astype(int) * 1.5
probability = 1 / (1 + np.exp(-latent_score))
target = (probability > 0.72).astype(int)

# 2. Add realistic anomalies & noise
data = {
    "transaction_id": [f"TXN_{10000 + i}" for i in range(n_samples)],
    "feature_x": feature_x,
    "feature_y": feature_y,
    "monthly_spend": spending,
    "account_age_months": account_age_months,
    "device_type": np.random.choice(["iOS", "Android", "Web", "Unknown"], size=n_samples, p=[0.4, 0.4, 0.15, 0.05]),
    "risk_region": np.random.choice(["North", "South", "East", "West"], size=n_samples),
    "fraud_detected": target
}

df = pd.DataFrame(data)

# Inject missing values
df.loc[df.sample(frac=0.10, random_state=42).index, "monthly_spend"] = np.nan
df.loc[df.sample(frac=0.08, random_state=42).index, "feature_x"] = np.nan

# Inject duplicate records
duplicates = df.sample(n=10, random_state=42)
df = pd.concat([df, duplicates], ignore_index=True)

df.to_csv("test_dataset_reflection_demo.csv", index=False)
print("Successfully created 'test_dataset_reflection_demo.csv'")