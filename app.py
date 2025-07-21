from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import joblib
import json
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../instance/users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

# Load model and scaler
model = joblib.load("loan_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

# Decorator for protected route
def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapper

# Route: Login/Register
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return "Username already exists"
    hashed_pw = generate_password_hash(password)
    new_user = User(username=username, password=hashed_pw)
    db.session.add(new_user)
    db.session.commit()
    return redirect(url_for('index'))

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        session['user_id'] = user.id
        session['username'] = user.username
        return redirect(url_for('dashboard'))
    return "Invalid credentials"

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Route: Dashboard + Prediction
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    prediction_text = None
    show_result = False
    rejection_reason = ""
    improvement_tip = ""
    estimated_time = ""
    approval_percent = 0
    rejection_percent = 0

    if request.method == 'POST':
        try:
            features = [
                float(request.form['income']),
                float(request.form['coapplicant_income']),
                float(request.form['loan_amount']),
                float(request.form['loan_term']),
                float(request.form['credit_score']),
                float(request.form['dependents']),
                float(request.form['gender']),
                float(request.form['married']),
                float(request.form['education']),
                float(request.form['self_employed']),
                float(request.form['property_area'])
            ]
            input_array = np.array(features).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]

            prediction_text = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
            show_result = True

            # Read and update stats
            if os.path.exists("prediction_counts.json"):
                with open("prediction_counts.json", "r") as f:
                    counts = json.load(f)
            else:
                counts = {"approved": 0, "rejected": 0}

            if prediction == 1:
                counts["approved"] += 1
            else:
                counts["rejected"] += 1

            with open("prediction_counts.json", "w") as f:
                json.dump(counts, f)

            # Use lowercase keys to avoid KeyError
            approval_percent = counts.get("approved", 0)
            rejection_percent = counts.get("rejected", 0)

            # Dynamic rejection reasoning
            if prediction == 0:
                if features[4] < 600:
                    rejection_reason = "Credit score is too low."
                    improvement_tip = "Pay bills on time, reduce credit card usage, and maintain older accounts."
                    estimated_time = "3-6 months"
                elif features[2] > 500000:
                    rejection_reason = "Loan amount requested is too high."
                    improvement_tip = "Consider reducing the loan amount or improving your income profile."
                    estimated_time = "1-3 months"
                elif features[0] < 2000:
                    rejection_reason = "Income is below acceptable threshold."
                    improvement_tip = "Increase stable income or add co-applicant with income."
                    estimated_time = "2-4 months"
                else:
                    rejection_reason = "One or more inputs do not meet the bank's requirements."
                    improvement_tip = "Review inputs or consult with a financial advisor."
                    estimated_time = "Varies"

        except Exception as e:
            return f"Prediction error: {str(e)}"

    return render_template(
        'dashboard.html',
        username=session.get('username'),
        prediction_text=prediction_text,
        show_result=show_result,
        approval_percent=approval_percent,
        rejection_percent=rejection_percent,
        rejection_reason=rejection_reason,
        improvement_tip=improvement_tip,
        estimated_time=estimated_time
    )

if __name__ == '__main__':
    app.run(debug=True)
