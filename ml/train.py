"""
Training pipeline: TF-IDF + (Logistic Regression | Multinomial Naive Bayes).
Auto-detects label/text columns so common Kaggle spam datasets
(SMS Spam Collection, Enron spam, spam_ham_dataset, etc.) work without edits.
"""
import os
import json
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from ml.preprocess import basic_clean

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

LABEL_CANDIDATES = ['label', 'v1', 'target', 'class', 'spam', 'category', 'is_spam', 'label_text']
TEXT_CANDIDATES = ['text', 'v2', 'message', 'email', 'body', 'content', 'sms', 'text_combined']

LABEL_VALUE_MAP = {
    'spam': 1, 'ham': 0, '1': 1, '0': 0, 1: 1, 0: 0,
    'yes': 1, 'no': 0, 'true': 1, 'false': 0, 'phishing': 1, 'legit': 0, 'legitimate': 0
}


def detect_columns(df: pd.DataFrame):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    label_col = next((cols_lower[c] for c in LABEL_CANDIDATES if c in cols_lower), None)
    text_col = next((cols_lower[c] for c in TEXT_CANDIDATES if c in cols_lower), None)

    if label_col is None or text_col is None:
        if len(df.columns) >= 2:
            # Heuristic fallback: shorter avg text length column = label, longer = text
            candidates = list(df.columns[:3])
            lengths = {c: df[c].astype(str).str.len().mean() for c in candidates}
            sorted_cols = sorted(lengths, key=lengths.get)
            label_col = label_col or sorted_cols[0]
            text_col = text_col or sorted_cols[-1]
        else:
            raise ValueError(
                "Could not detect label/text columns. Rename them to 'label' and 'text' "
                "(or common variants like 'v1'/'v2', 'category'/'message')."
            )
    return label_col, text_col


def normalize_labels(series: pd.Series) -> pd.Series:
    def conv(v):
        if isinstance(v, (int, float)) and not pd.isna(v):
            return int(v)
        key = str(v).strip().lower()
        if key in LABEL_VALUE_MAP:
            return LABEL_VALUE_MAP[key]
        raise ValueError(f"Unrecognized label value: '{v}'. Use spam/ham or 1/0.")
    return series.apply(conv)


def train_model(csv_path: str, algorithm: str = 'logreg') -> dict:
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Dataset file not found.")

    try:
        df = pd.read_csv(csv_path, encoding='latin-1')
    except Exception:
        df = pd.read_csv(csv_path, encoding='utf-8', engine='python', on_bad_lines='skip')

    df = df.dropna(how='all')
    if len(df) < 20:
        raise ValueError("Dataset too small. Provide at least 20 labeled rows.")

    label_col, text_col = detect_columns(df)
    df = df[[label_col, text_col]].dropna()
    df.columns = ['label', 'text']
    df['label'] = normalize_labels(df['label'])
    df['clean_text'] = df['text'].astype(str).apply(basic_clean)
    df = df[df['clean_text'].str.len() > 0]

    if df['label'].nunique() < 2:
        raise ValueError("Dataset must contain both spam and legitimate examples.")

    min_class_count = df['label'].value_counts().min()
    if min_class_count < 5:
        raise ValueError("Each class needs at least 5 examples to train reliably.")

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    if algorithm == 'nb':
        model = MultinomialNB()
    else:
        algorithm = 'logreg'
        model = LogisticRegression(max_iter=1000, class_weight='balanced')

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "total_size": int(len(df)),
        "spam_count": int(df['label'].sum()),
        "ham_count": int(len(df) - df['label'].sum()),
        "algorithm": algorithm,
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
    }

    joblib.dump(model, os.path.join(MODEL_DIR, 'model.pkl'))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'vectorizer.pkl'))
    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics
