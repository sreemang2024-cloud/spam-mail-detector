# Sentinel — Spam & Phishing Email Detector

## Folder structure
```
spam-detector/
├── app.py                        # Flask app (serves UI + API)
├── requirements.txt
├── ml/
│   ├── __init__.py
│   ├── preprocess.py              # text cleaning + feature extraction
│   ├── train.py                   # TF-IDF + LogisticRegression/NaiveBayes training
│   └── predict.py                 # inference wrapper
├── data/
│   ├── generate_sample_dataset.py # regenerates bundled sample data
│   └── sample_dataset.csv         # bundled sample (works out of the box)
├── models/                        # model.pkl, vectorizer.pkl, metrics.json (created after training)
├── uploads/                       # your uploaded Kaggle CSVs land here
├── templates/index.html
└── static/
    ├── css/style.css
    └── js/main.js
```

## Setup
```bash
cd spam-detector
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python app.py
```
Open **http://127.0.0.1:5000**

## Using your own Kaggle dataset
1. Download any labeled spam dataset from Kaggle, e.g.:
   - "SMS Spam Collection Dataset"
   - "Spam Email Dataset"
   - "Email Spam Classification Dataset"
2. On the site, go to the **Dataset** section and drag/drop the `.csv`.
3. Columns are auto-detected. Supported label columns: `label`, `v1`, `target`, `class`, `spam`, `category`, `is_spam`.
   Supported text columns: `text`, `v2`, `message`, `email`, `body`, `content`, `sms`.
   Label values recognized: `spam`/`ham`, `1`/`0`, `yes`/`no`, `true`/`false`.
4. Go to **Train**, pick an algorithm, click **Train model**.
5. Use **Detect** for single emails, or **Batch** to score a whole CSV at once.

## Notes
- The app ships with a bundled 300-row sample dataset so the pipeline works immediately without any upload.
- All data, models, and history stay local to your machine (`models/`, `uploads/`, `history.json`) — nothing is sent anywhere except your own Flask server.
- To regenerate the bundled sample: `python data/generate_sample_dataset.py`
