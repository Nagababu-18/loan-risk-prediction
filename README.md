Here's a clean and professional **`README.md`** file for your **Loan Risk Prediction System** project, including the 3D pie chart update and full deployment-ready instructions.

---

```markdown
# 🏦 Loan Risk Prediction System

This is a machine learning-powered web application that predicts the likelihood of a loan being approved based on user inputs. It also visually displays the percentage of approved vs rejected predictions using an interactive 3D pie chart.

## 🚀 Features

- 🔮 Predicts loan approval status using historical data.
- 📈 Shows dynamic 3D pie chart for approvals vs rejections.
- 🌐 Web UI built with Tailwind CSS.
- ⚙️ Backend built with Flask + trained XGBoost model.
- 📊 Tracks approval/rejection trends in real-time.

---

## 🖼️ Demo

<img width="1605" height="801" alt="Screenshot 2025-07-20 071539" src="https://github.com/user-attachments/assets/0408eecd-7bb9-4ddb-8b76-829d4dfdbae8" />


---

## 📁 Project Structure

```

LoanApprovalSystem/
│
├── backend/
│   ├── app.py                    # Flask backend
│   ├── loan\_risk\_model.pkl       # Trained ML model (XGBoost)
│   ├── scaler.pkl                # Trained StandardScaler
│   ├── templates/
│   │   └── index.html            # UI HTML (TailwindCSS + Plotly)
│
├── dataset/
│   └── loan\_approval\_dataset.csv # Historical data (for training)
│
├── static/                       # (Optional: for future CSS/JS)
│
└── README.md                     # Project documentation

````

---

## ⚙️ Setup Instructions

### 1. 📦 Install Dependencies

```bash
pip install flask numpy pandas scikit-learn xgboost joblib plotly
````

### 2. 🧠 Train Model (if not already done)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

# Load and preprocess data
df = pd.read_csv("dataset/loan_approval_dataset.csv")
X = df.drop('loan_status', axis=1)
y = df['loan_status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = XGBClassifier()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, 'backend/loan_risk_model.pkl')
joblib.dump(scaler, 'backend/scaler.pkl')
```

### 3. ▶️ Run Flask App

```bash
cd backend
python app.py
```

Open your browser at: `http://127.0.0.1:5000`

---

## 🧪 Sample Input for Prediction

| Field              | Value    |
| ------------------ | -------- |
| Income             | 80000    |
| Loan Amount        | 100      |
| Loan Term          | 360      |
| Credit Score       | 800      |
| Dependents         | 0        |
| Gender             | Male     |
| Married            | Yes      |
| Education          | Graduate |
| Self Employed      | No       |
| Property Area      | Urban    |
| Coapplicant Income | 25000    |

---

## 📊 Visual Output

After prediction, you’ll see:

* ✅ Loan Approved or ❌ Rejected
* 📈 3D Pie Chart showing total approval vs rejection distribution

---

## ✨ Future Enhancements

* ✅ Add login and user authentication
* 🧠 Improve model accuracy with more features
* 📧 Email notification after prediction
* 📈 Admin dashboard to track prediction stats

---

## 📌 Author

**Nagu Chintha**
GitHub: nagababu-18
Email:chinthanagu6@g.mail.com

---

