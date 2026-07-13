"""
Text cleaning and hand-crafted feature extraction for the spam classifier.
Kept dependency-free (no nltk download step) for zero-friction setup.
"""
import re
import string

URL_PATTERN = re.compile(
    r'(https?://\S+|www\.\S+|\b[a-z0-9-]+\.(?:com|net|org|io|ly|co|info|biz|xyz)\b\S*)',
    re.IGNORECASE
)
EMAIL_PATTERN = re.compile(r'\S+@\S+')
NUMBER_PATTERN = re.compile(r'\b\d+\b')

SPAM_KEYWORDS = [
    'free', 'winner', 'win', 'cash', 'prize', 'urgent', 'click here', 'act now',
    'limited time', 'offer', 'congratulations', 'claim', 'credit', 'loan',
    'lottery', 'guarantee', 'risk-free', 'subscribe', 'unsubscribe',
    'bonus', 'discount', 'buy now', 'order now', 'call now', 'earn money',
    'work from home', 'no cost', '100% free', 'apply now', 'password', 'verify account',
    'account suspended', 'confirm your identity', 'wire transfer', 'gift card'
]

_PUNCT_TABLE = str.maketrans('', '', string.punctuation.replace('!', ''))


def basic_clean(text: str) -> str:
    """Lowercase, mask URLs/emails, strip punctuation (keep '!'), collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_PATTERN.sub(' urltoken ', text)
    text = EMAIL_PATTERN.sub(' emailtoken ', text)
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_features(text: str) -> dict:
    """Interpretable signals shown in the UI alongside the model's verdict."""
    if not isinstance(text, str):
        text = ""
    urls = len(URL_PATTERN.findall(text))
    exclamations = text.count('!')
    caps_words = sum(1 for w in text.split() if len(w) > 1 and w.isupper())
    length = len(text)
    words = text.split()
    word_count = len(words)
    avg_word_len = round(sum(len(w) for w in words) / word_count, 2) if word_count else 0.0
    lower = text.lower()
    keyword_hits = sorted({k for k in SPAM_KEYWORDS if k in lower})
    digit_count = len(NUMBER_PATTERN.findall(text))

    return {
        "url_count": urls,
        "exclamation_count": exclamations,
        "caps_word_count": caps_words,
        "length": length,
        "word_count": word_count,
        "avg_word_length": avg_word_len,
        "keyword_hits": keyword_hits,
        "keyword_hit_count": len(keyword_hits),
        "digit_count": digit_count,
    }
