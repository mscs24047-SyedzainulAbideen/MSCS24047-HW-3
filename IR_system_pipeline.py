# ============================================
# INFORMATION RETRIEVAL SYSTEM (Boolean + TF-IDF + BM25)
# ============================================

import pandas as pd
import numpy as np
import re
import nltk
import warnings

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# =====================================================
# 0. Suppress warnings and NLTK messages
# =====================================================
warnings.filterwarnings("ignore")
try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords", quiet=True)

# =====================================================
# 1. LOAD DATASET (handle encoding automatically)
# =====================================================
csv_path = r"D:\MSCS24047\3rd Semester\IR&TM\HW3\Articles.csv"

# Try multiple encodings until successful
for enc in ["utf-8", "latin1", "ISO-8859-1", "cp1252"]:
    try:
        df = pd.read_csv(csv_path, encoding=enc, on_bad_lines="skip")
        break
    except Exception:
        continue

print(f"Dataset loaded successfully! Total documents: {len(df)}")
print("Columns:", df.columns.tolist())

# =====================================================
# 2. SELECT TEXT COLUMN
# =====================================================
text_col = None
for col in df.columns:
    if col.lower() in ["article", "text", "content", "body"]:
        text_col = col
        break
if text_col is None:
    text_col = df.columns[0]

print(f"Using text column: {text_col}")
df[text_col] = df[text_col].astype(str).fillna("")

# =====================================================
# 3. TEXT PREPROCESSING
# =====================================================
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = text.strip()
    return text

def tokenize(text):
    tokens = clean_text(text).split()
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

df["clean_text"] = df[text_col].apply(clean_text)
df["tokens"] = df["clean_text"].apply(tokenize)

# =====================================================
# 4. BOOLEAN RETRIEVAL
# =====================================================
def boolean_search(query):
    q_tokens = tokenize(query)
    results = df[df["tokens"].apply(lambda x: all(t in x for t in q_tokens))]
    return results.head(5)

# =====================================================
# 5. TF-IDF RETRIEVAL
# =====================================================
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

def retrieve_tfidf(query):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = scores.argsort()[-5:][::-1]
    return df.iloc[top_idx]

# =====================================================
# 6. BM25 RETRIEVAL
# =====================================================
bm25 = BM25Okapi(df["tokens"])

def retrieve_bm25(query):
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)
    top_idx = np.argsort(scores)[-5:][::-1]
    return df.iloc[top_idx]

# =====================================================
# 7. RUN A SAMPLE QUERY
# =====================================================
query = "economic policy inflation government"

print("\n===== BOOLEAN RESULTS =====")
for i, row in boolean_search(query).iterrows():
    print(f"\n--- Document {i} ---\n{row[text_col][:400]}")

print("\n===== TF-IDF RESULTS =====")
for i, row in retrieve_tfidf(query).iterrows():
    print(f"\n--- Document {i} ---\n{row[text_col][:400]}")

print("\n===== BM25 RESULTS =====")
for i, row in retrieve_bm25(query).iterrows():
    print(f"\n--- Document {i} ---\n{row[text_col][:400]}")
