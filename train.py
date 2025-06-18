import pandas as pd
import re
import joblib
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(text):
    text = text.upper()
    text = re.sub(r"[^A-ZÇĞİÖŞÜ\s]", "", text)
    return text.strip().split()

def train_word2vec(df):
    print("🔄 Word2Vec modeli eğitiliyor...")
    sentences = df["text"].apply(preprocess).tolist()
    model = Word2Vec(sentences=sentences, vector_size=50, window=5, sg=0, min_count=1, workers=4)
    model.save("word2vec_cbow_50.model")
    print("✅ Word2Vec modeli kaydedildi: word2vec_cbow_50.model")

def train_tfidf(df):
    print("🔄 TF-IDF modeli eğitiliyor...")
    texts = df["text"].astype(str).tolist()
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(texts)
    joblib.dump(tfidf, "tfidf_model.pkl")
    print("✅ TF-IDF modeli kaydedildi: tfidf_model.pkl")

def main():
    df = pd.read_csv("harcama_aciklamalari_raw.csv")
    train_word2vec(df)
    train_tfidf(df)

if __name__ == "__main__":
    main()
