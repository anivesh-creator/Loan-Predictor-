import pandas as pd
import numpy as np

print("🏦 Making fake loan applications...")

np.random.seed(42)

count = 500

df = pd.DataFrame({
    'salary_monthly': np.random.randint(30000, 200000, count),
    'fico': np.random.randint(300, 850, count),
    'loan_requested': np.random.randint(100000, 5000000, count),
    'years_employed': np.random.randint(0, 30, count),
    'current_debt': np.random.randint(0, 100000, count),
})

print(f"✅ Created {count} fake people")

df['loan_approved'] = (
    (df['fico'] >= 650) &
    (df['salary_monthly'] * 3 > df['loan_requested'] / 100) &
    (df['years_employed'] >= 1) &
    (df['current_debt'] < df['salary_monthly'] * 0.4)
).astype(int)

print(f"✅ {df['loan_approved'].sum()} people approved")
print(f"✅ {len(df) - df['loan_approved'].sum()} people rejected")

df.to_csv('loan_data.csv', index=False)
print("✅ saved to: loan_data.csv")

print("\nFirst 5 people:")
print(df.head())
