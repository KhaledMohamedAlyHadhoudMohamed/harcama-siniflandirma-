import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_score
import numpy as np
import re

def clean(text):
    text = text.upper()
    text = re.sub(r"[^A-ZÇĞİÖŞÜ\s]", "", text)
    return text.split()

# Jaccard için ikili vektörleştirme (çok basit)
def binary_vectorize(words, vocab):
    return [1 if word in words else 0 for word in vocab]

df = pd.read_csv("harcama_aciklamalari_raw.csv")
texts = df["text"].tolist()
cleaned_texts = [" ".join(clean(t)) for t in texts]

# Cosine benzerliği (TF-IDF üzerinden)
print("🔍 TF-IDF + Cosine benzerliği hesaplanıyor...")
vec = TfidfVectorizer()
matrix = vec.fit_transform(cleaned_texts)
cosine_scores = cosine_similarity(matrix[0:1], matrix[1:6])
print("\n📌 Cosine En Benzer 5:")
for i, score in enumerate(cosine_scores[0]):
    print(f"{texts[i+1]} → {score:.4f}")

# Jaccard benzerliği
print("\n🔍 Jaccard benzerliği hesaplanıyor...")
input_words = clean(texts[0])
vocab = list(set(" ".join(cleaned_texts).split()))
input_vec = binary_vectorize(input_words, vocab)
for i in range(1, 6):
    target_words = clean(texts[i])
    target_vec = binary_vectorize(target_words, vocab)
    jaccard = jaccard_score(input_vec, target_vec)
    print(f"{texts[i]} → {jaccard:.4f}")