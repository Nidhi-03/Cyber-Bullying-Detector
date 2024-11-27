import os
from flask import Flask, render_template, request
import re
import tensorflow as tf  
from tensorflow.keras.preprocessing.sequence import pad_sequences  
import pickle  

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

app = Flask(__name__)

try:
    model = tf.keras.models.load_model('lstm_model.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

def preprocess_comment(comment):
    comment = comment.lower()  
    comment = re.sub(r'[^\w\s]', '', comment)
    return comment.strip()  

@app.route('/', methods=['GET', 'POST'])
def detect_comment():
    prediction = None  
    user_input = ""  
    if request.method == 'POST':
        if 'detect' in request.form:
            user_input = request.form['comment']  
            processed_comment = preprocess_comment(user_input)  
            sequences = tokenizer.texts_to_sequences([processed_comment])  
            padded_sequences = pad_sequences(sequences, maxlen=100)  
            prediction_label = model.predict(padded_sequences)[0][0]  
            prediction = "Cyberbullying" if prediction_label > 0.5 else "Non-Cyberbullying"  
        elif 'delete' in request.form:
            user_input = ""  
            prediction = None  

    return render_template('index.html', user_input=user_input, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)  

