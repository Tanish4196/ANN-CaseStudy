import nbformat as nbf
import os
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- 1. Train Model and Create Artifacts ---
print("Training Model and Creating Artifacts...")
os.makedirs("artifacts", exist_ok=True)

df = pd.read_csv("Artificial_Neural_Network_Case_Study_data.csv")

# Feature Engineering
X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
y = df['Exited']

# Encoding
le_gender = LabelEncoder()
X['Gender'] = le_gender.fit_transform(X['Gender'])

X = pd.get_dummies(X, columns=['Geography'], drop_first=False)

feature_columns = list(X.columns)
with open("artifacts/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open("artifacts/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build ANN
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

model.save("artifacts/churn_ann_model.keras")
print("Saved artifacts to 'artifacts/'")

# --- 2. Create model.ipynb ---
print("Generating model.ipynb...")
nb = nbf.v4.new_notebook()

code_1 = """\
# Customer Churn Prediction - ANN Model Training
import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout\
"""

code_2 = """\
# Load dataset
df = pd.read_csv("Artificial_Neural_Network_Case_Study_data.csv")
df.head()\
"""

code_3 = """\
# Preprocessing
X = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
y = df['Exited']

# Encoding constraints
le_gender = LabelEncoder()
X['Gender'] = le_gender.fit_transform(X['Gender'])
X = pd.get_dummies(X, columns=['Geography'], drop_first=False)\
"""

code_4 = """\
os.makedirs("artifacts", exist_ok=True)
# Save feature columns safely for the Streamlit app
feature_columns = list(X.columns)
with open("artifacts/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

# Scaling dataset safely
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open("artifacts/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)\
"""

code_5 = """\
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the Network
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)\
"""

code_6 = """\
# Evaluate and Save Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
model.save("artifacts/churn_ann_model.keras")\
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Customer Churn Model Training\nThis notebook demonstrates data cleaning, preprocessing, and model training to build our robust Artificial Neural Network using standard bank churning profiles."),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_markdown_cell("## Deep Learning ANN Architecture"),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_code_cell(code_6)
]

with open('model.ipynb', 'w') as f:
    nbf.write(nb, f)

print("model.ipynb created successfully.")
