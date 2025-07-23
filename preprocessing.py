import re
import nltk
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"\S*@\S*\s?", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 3]
    return tokens

def load_and_clean_data():
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    docs = data.data[:1000]  # use a subset for speed
    cleaned_texts = [clean_text(doc) for doc in docs]
    return docs, cleaned_texts