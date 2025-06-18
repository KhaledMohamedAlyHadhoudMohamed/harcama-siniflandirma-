import argparse
import pandas as pd
import joblib
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocess_text, average_word2vec

def classify_word2vec(input_text, model_path, dataset_path):
    model = gensim.models.Word2Vec.load(model_path)
    df = pd.read_csv(dataset_path)
    texts = df["text"].tolist()

    input_words = preprocess_text(input_text)
    input_vec = average_word2vec(input_words, model)
    if input_vec is None:
        print("❌ Giriş kelimeleri modelde bulunamadı.")
        return

    similarities = []
    for text in texts:
        words = preprocess_text(text)
        vec = average_word2vec(words, model)
        if vec is not None:
            sim = cosine_similarity([input_vec], [vec])[0][0]
            similarities.append((text, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    print("\n📌 En benzer 5 açıklama (Word2Vec):")
    for text, score in similarities[:5]:
        print(f"{text} → {score:.4f}")

def classify_tfidf(input_text, model_path, dataset_path):
    df = pd.read_csv(dataset_path)
    tfidf = joblib.load(model_path)
    matrix = tfidf.transform(df["text"].tolist())
    input_vec = tfidf.transform([input_text])
    scores = cosine_similarity(input_vec, matrix)[0]
    top_indices = scores.argsort()[::-1][:5]
    print("\n📌 En benzer 5 açıklama (TF-IDF):")
    for i in top_indices:
        print(f"{df['text'].iloc[i]} → {scores[i]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", default="harcama_aciklamalari_raw.csv")
    parser.add_argument("--mode", choices=["w2v", "tfidf"], default="w2v")
    args = parser.parse_args()

    if args.mode == "w2v":
        classify_word2vec(args.input, args.model, args.dataset)
    else:
        classify_tfidf(args.input, args.model, args.dataset)
