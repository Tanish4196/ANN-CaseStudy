import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 10000

data = {
    'RowNumber': np.arange(1, n_samples + 1),
    'CustomerId': np.random.randint(15600000, 15800000, n_samples),
    'Surname': [f'User_{i}' for i in range(n_samples)],
    'CreditScore': np.random.randint(350, 850, n_samples),
    'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples, p=[0.5, 0.25, 0.25]),
    'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45]),
    'Age': np.clip(np.random.normal(38, 10, n_samples), 18, 92).astype(int),
    'Tenure': np.random.randint(0, 11, n_samples),
    'NumOfProducts': np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.45, 0.04, 0.01]),
    'HasCrCard': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'IsActiveMember': np.random.choice([0, 1], n_samples, p=[0.48, 0.52]),
    'EstimatedSalary': np.random.uniform(10, 200000, n_samples),
}

df = pd.DataFrame(data)

nonzero_balances = np.clip(np.random.normal(100000, 30000, n_samples), 10000, 250000)
df['Balance'] = np.where(df['Geography'] == 'Germany', 
                         nonzero_balances, 
                         np.where(np.random.rand(n_samples) > 0.36, nonzero_balances, 0))

score = (
    (df['Age'] > 50) * 1.5 +
    (df['Age'] <= 30) * -0.5 +
    (df['IsActiveMember'] == 0) * 1.2 +
    (df['Geography'] == 'Germany') * 0.8 +
    (df['Balance'] > 0) * 0.3 + 
    (df['Gender'] == 'Female') * 0.4 +
    (df['NumOfProducts'] == 3) * 1.5 +
    (df['NumOfProducts'] == 4) * 2.0
)

prob = 1 / (1 + np.exp(-(score - 3.2))) 
df['Exited'] = (np.random.rand(n_samples) < prob).astype(int)

df.to_csv('Artificial_Neural_Network_Case_Study_data.csv', index=False)
print("Data generated successfully: Artificial_Neural_Network_Case_Study_data.csv")
