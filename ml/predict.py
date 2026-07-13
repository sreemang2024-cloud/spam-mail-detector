"""Loads the trained model + vectorizer and serves predictions."""
import os
import joblib

from ml.preprocess import basic_clean, extract_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')


class Predictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.load()

    def load(self) -> bool:
        model_path = os.path.join(MODEL_DIR, 'model.pkl')
        vec_path = os.path.join(MODEL_DIR, 'vectorizer.pkl')
        if os.path.exists(model_path) and os.path.exists(vec_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vec_path)
            return True
        self.model = None
        self.vectorizer = None
        return False

    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def predict(self, text: str) -> dict:
        if not self.is_ready():
            raise RuntimeError("Model not trained yet. Train the model first.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Email text is empty.")

        clean = basic_clean(text)
        vec = self.vectorizer.transform([clean])
        pred = int(self.model.predict(vec)[0])

        try:
            proba = self.model.predict_proba(vec)[0]
            confidence = float(max(proba))
        except AttributeError:
            confidence = 0.85

        return {
            "prediction": "spam" if pred == 1 else "legitimate",
            "is_spam": bool(pred),
            "confidence": round(confidence * 100, 2),
            "features": extract_features(text),
        }
