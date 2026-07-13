"""
Spam Email Detector — Flask backend.
Serves the UI (templates/static) and a small JSON API for:
dataset upload, training, single prediction, batch prediction, history, model status.
"""
import os
import json
import uuid
from datetime import datetime, timezone

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import pandas as pd

from ml.train import train_model
from ml.predict import Predictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
HISTORY_FILE = os.path.join(BASE_DIR, 'history.json')

for d in (UPLOAD_DIR, DATA_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 25 MB upload cap

predictor = Predictor()
ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def save_history_entry(entry: dict) -> None:
    history = load_history()
    history.insert(0, entry)
    history = history[:100]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


# ---------- page ----------

@app.route('/')
def index():
    return render_template('index.html')


# ---------- api ----------

@app.route('/api/model-status', methods=['GET'])
def model_status():
    ready = predictor.is_ready()
    metrics = None
    metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return jsonify({"ready": ready, "metrics": metrics})


@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file was selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only .csv files are supported."}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:10]}_{filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    file.save(save_path)

    try:
        preview_df = pd.read_csv(save_path, encoding='latin-1', nrows=5)
        full_df = pd.read_csv(save_path, encoding='latin-1')
    except Exception as e:
        os.remove(save_path)
        return jsonify({"error": f"Could not read this CSV: {str(e)}"}), 400

    if len(full_df) < 20:
        os.remove(save_path)
        return jsonify({"error": "Dataset too small — provide at least 20 rows."}), 400

    return jsonify({
        "message": "Dataset uploaded.",
        "filename": unique_name,
        "columns": list(preview_df.columns),
        "row_count": int(len(full_df)),
        "preview": json.loads(preview_df.head(5).to_json(orient='records'))
    })


@app.route('/api/train', methods=['POST'])
def train():
    data = request.get_json(silent=True) or {}
    filename = data.get('filename')
    algorithm = data.get('algorithm', 'logreg')
    if algorithm not in ('logreg', 'nb'):
        algorithm = 'logreg'

    csv_path = os.path.join(UPLOAD_DIR, filename) if filename else os.path.join(DATA_DIR, 'sample_dataset.csv')

    if filename and not os.path.exists(csv_path):
        return jsonify({"error": "Uploaded dataset not found. Please upload it again."}), 404

    try:
        metrics = train_model(csv_path, algorithm=algorithm)
        predictor.load()
        return jsonify({"message": "Model trained successfully.", "metrics": metrics})
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({"error": "Email text is required."}), 400
    if len(text) > 20000:
        return jsonify({"error": "Email text is too long (max 20,000 characters)."}), 400

    try:
        result = predictor.predict(text)
        save_history_entry({
            "id": uuid.uuid4().hex[:8],
            "text": text[:200],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    if 'file' not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({"error": "Only .csv files are supported."}), 400
    if not predictor.is_ready():
        return jsonify({"error": "Model not trained yet. Train the model first."}), 409

    try:
        df = pd.read_csv(file, encoding='latin-1')
    except Exception as e:
        return jsonify({"error": f"Could not read this CSV: {str(e)}"}), 400

    text_col = None
    for cand in ['text', 'v2', 'message', 'email', 'body', 'content', 'sms']:
        for c in df.columns:
            if c.lower().strip() == cand:
                text_col = c
                break
        if text_col:
            break
    if text_col is None:
        text_col = df.columns[0]

    results = []
    limit = 500
    for _, row in df.head(limit).iterrows():
        text = str(row[text_col])
        try:
            r = predictor.predict(text)
            results.append({
                "text": text[:150],
                "prediction": r["prediction"],
                "confidence": r["confidence"]
            })
        except ValueError:
            continue

    spam_count = sum(1 for r in results if r["prediction"] == "spam")
    return jsonify({
        "results": results,
        "total": len(results),
        "truncated": len(df) > limit,
        "spam_count": spam_count,
        "legitimate_count": len(results) - spam_count
    })


@app.route('/api/history', methods=['GET'])
def history():
    return jsonify(load_history())


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f)
    return jsonify({"message": "History cleared."})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
