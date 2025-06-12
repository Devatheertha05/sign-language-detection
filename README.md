
# 🤟 Sign Language Detection Web App

A real-time sign language recognition system built with Flask, OpenCV, and machine learning. This application captures sign gestures via webcam, predicts the corresponding alphabet using a trained ML model, and speaks the result using browser-based speech synthesis.

🚀 **[Live Demo](https://signlanguage-ecj0.onrender.com)**

## 🌟 Features

- 🎥 Real-time webcam-based sign detection
- 🧠 Trained machine learning model (Random Forest)
- 💬 Automatic speech output of predicted sign
- 💻 Glassmorphic UI with Bootstrap 5
- 🧪 Speech synthesis integration (browser-based)
- 🛠️ Flask backend for model integration and video streaming

## 📁 Project Structure

```
sign-language-flask-app/
│
├── static/                    # Custom JS/CSS (optional)
├── templates/
│   └── index.html             # Frontend HTML file
│
├── model/
│   └── model.pkl              # Trained ML model
│
├── sign_data_combined.csv     # Dataset used for training
├── app.py                     # Flask app with camera logic
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Devatheertha05/sign_language_detection.git
cd sign_language_detection
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the App Locally

```bash
python app.py
```

Open your browser at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 🧠 Model Training (Optional)

To retrain the model using `sign_data_combined.csv`:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd, pickle

df = pd.read_csv("sign_data_combined.csv")
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
```

## 🧪 Tech Stack

- **Frontend:** HTML, Bootstrap 5, JavaScript (SpeechSynthesis API)
- **Backend:** Python, Flask
- **ML Model:** RandomForest (scikit-learn)
- **Libraries:** OpenCV, Pandas, Pickle, Mediapipe

## 📦 Dependencies

Install all required Python packages using:

```bash
pip install -r requirements.txt
```

## 🔊 Demo Features

- 📷 Real-time webcam feed
- ✋ Detects hand sign gestures
- 🗣️ Speaks out the predicted result

## 📜 License

This project is open-source and free to use.

## 🤝 Contributions

Feel free to fork this repo and open a pull request!

## 🧑‍💻 Author

**Devatheertha**  
[GitHub](https://github.com/Devatheertha05)
