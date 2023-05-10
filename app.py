from flask import Flask, render_template, request
import joblib
import numpy as np
from keras.utils import pad_sequences
import tensorflow as tf
import os
import json
import pandas as pd
from keras.models import load_model
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
import sqlite3
from flask_security import Security, SQLAlchemyUserDatastore, UserMixin, RoleMixin

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)
    role = db.Column(db.String(50), nullable=False)

with app.app_context():
    db.create_all()


# Load your saved models here
dt_model = joblib.load('dt_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
lstm_model = load_model('lstm_model.h5')
rnn_model = load_model('rnn_model.h5')

# Load your tokenizer
tokenizer = joblib.load('tokenizer.pkl')

# Set max_len according to your preprocessed data
max_len = 100

# Add any required preprocessing steps here
def preprocess(text):
    # Implement the same preprocessing steps as before
    input_seq = tokenizer.texts_to_sequences([text])
    input_pad = pad_sequences(input_seq, maxlen=max_len)
    return input_pad

app.secret_key = os.urandom(24)
# User authentication (FR-01)
# You can implement user authentication using a database like SQLite, MySQL, or PostgreSQL
# For demonstration purposes, we'll use an in-memory dictionary
users = {
    "admin": "password",
}


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            session['user'] = username
            session['role'] = user.role
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials', 'error')
    return render_template('login.html')



@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


@app.route('/predict-sentiment', methods=['GET', 'POST'])
def predict_sentiment():
    if "user" not in session:
        return redirect(url_for("login"))
    if request.method == 'POST':
        text = request.form['text']
        model_name = request.form['model']
        # Preprocess the text
        preprocessed_text = preprocess(text)
        
        # Select the appropriate model
        if model_name == 'decision_tree':
            model = dt_model
        elif model_name == 'xgboost':
            model = xgb_model
        elif model_name == 'lstm':
            model = lstm_model
        elif model_name == 'rnn':
            model = rnn_model
        
        # Make a prediction
        if model_name in ['decision_tree', 'xgboost']:
            prediction = model.predict(preprocessed_text)
        else:
            prediction = (model.predict(preprocessed_text) > 0.5).astype(int)

        sentiment = 'positive' if prediction[0] == 1 else 'negative'
        
        return render_template('predict_sentiment.html', sentiment=sentiment)
    else:
        return render_template('predict_sentiment.html')
    

# Data Upload (FR-02)
@app.route("/data-upload", methods=["GET", "POST"])
def data_upload():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        data_file = request.files["datafile"]
        if data_file:
            global data
            data = pd.read_csv(data_file)
            return redirect(url_for("data_visualization"))
    return render_template("data_upload.html")

# Data Visualization (FR-03)
@app.route("/data-visualization")
def data_visualization():
    if "user" not in session:
        return redirect(url_for("login"))

    # Implement data visualization using the 'data' DataFrame
    return render_template("data_visualization.html")

# Customer Segmentation (FR-04)
@app.route("/customer-segmentation")
def customer_segmentation():
    if "user" not in session:
        return redirect(url_for("login"))

    # Implement customer segmentation using the 'data' DataFrame
    return render_template("customer_segmentation.html")

# Purchase Prediction (FR-05)
@app.route("/purchase-prediction")
def purchase_prediction():
    if "user" not in session:
        return redirect(url_for("login"))

    # Implement purchase prediction using the 'data' DataFrame and machine learning models
    return render_template("purchase_prediction.html")

# Customer Lifetime Value Calculation (FR-06)
@app.route("/customer-lifetime-value")
def customer_lifetime_value():
    if "user" not in session:
        return redirect(url_for("login"))

    # Implement Customer Lifetime Value calculation using the 'data' DataFrame
    return render_template("customer_lifetime_value.html")

# User Management (FR-07)
@app.route("/user-management")
def user_management():
    if 'user' not in session or session['role'] != 'admin':
        return redirect(url_for("login"))

    # Implement user management (add, remove, assign roles, set permissions)
    return render_template("user_management.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        new_user = User(username=username, password=password, role=role)
        db.session.add(new_user)
        db.session.commit()
        flash('User created successfully', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


if __name__ == "__main__":
    for rule in app.url_map.iter_rules():
        print(rule.endpoint)

    app.run(debug=True)