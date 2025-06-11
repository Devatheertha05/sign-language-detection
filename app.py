from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import mediapipe as mp
from PIL import Image
import io
import os
app = Flask(__name__)

# Load trained model
with open("model/sign_model.pkl", "rb") as f:
    model = pickle.load(f)

current_prediction = "..."

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global current_prediction
    if 'frame' not in request.files:
        return jsonify({'prediction': current_prediction})
    
    file = request.files['frame']
    image = Image.open(file.stream).convert('RGB')
    image_np = np.array(image)

    results = hands.process(image_np)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if len(landmarks) == 63:
                prediction = model.predict([landmarks])
                current_prediction = prediction[0]
                break

    return jsonify({'prediction': current_prediction})

if __name__ == '__main__':
    app.run(debug=True)
app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

