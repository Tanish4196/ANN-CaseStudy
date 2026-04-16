📊 Customer Churn Prediction using Artificial Neural Network (ANN)

An interactive Machine Learning web application that predicts whether a customer is likely to churn using an Artificial Neural Network (ANN).

🔗 Built with Streamlit | TensorFlow | Scikit-learn

📊 Features

🔮 Predict customer churn probability
📈 Displays risk percentage (Churn vs Retention)
🎯 Clean and interactive UI dashboard
⚡ Real-time predictions using trained ANN model

🧠 How It Works

The model is trained on customer data and uses the following preprocessing steps:

Gender encoding
Geography one-hot encoding
Feature scaling using StandardScaler
Artificial Neural Network (ANN) for classification
📁 Project Structure
.
├── app.py
├── model.ipynb
├── requirements.txt
├── runtime.txt
├── artifacts/
│   ├── churn_ann_model.keras
│   ├── scaler.pkl
│   └── feature_columns.json
├── Artificial_Neural_Network_Case_Study_data.csv
└── README.md
⚙️ Installation
🔹 Clone the repository
git clone https://github.com/Tanish4196/ANN-CaseStudy.git
cd ANN-CaseStudy
🔹 Install dependencies
pip install -r requirements.txt
🌍 Deployment

This app is deployed using Streamlit Community Cloud

👉 Live App:
https://ann-casestudy-tanishsharma11222648.streamlit.app/

📌 Notes
Model predicts churn probability based on user input
Threshold logic:
≥ 50% → High Risk (Churn)
< 50% → Low Risk (Retention)
Ensure all artifact files exist before running the app
👨‍💻 Author

Tanish Sharma
GitHub: https://github.com/Tanish4196









