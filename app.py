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

app = Flask(__name__)

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

@app.route('/', methods=['GET', 'POST'])
def index():
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
        
        return render_template('index.html', sentiment=sentiment)
    else:
        return render_template('index.html')
    

app.secret_key = os.urandom(24)
# User authentication (FR-01)
# You can implement user authentication using a database like SQLite, MySQL, or PostgreSQL
# For demonstration purposes, we'll use an in-memory dictionary
users = {
    "admin": "password",
}

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username] == password:
            session["user"] = username
            return redirect(url_for("data_upload"))
        else:
            flash("Invalid credentials", "error")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

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
    if "user" not in session:
        return redirect(url_for("login"))

    # Implement user management (add, remove, assign roles, set permissions)
    return render_template("user_management.html")

if __name__ == "__main__":
    app.run(debug=True)