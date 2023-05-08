from flask import Flask, render_template, request
import joblib
import numpy as np
from keras.utils import pad_sequences
from keras.models import load_model


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

if __name__ == '__main__':
    app.run(debug=True)
