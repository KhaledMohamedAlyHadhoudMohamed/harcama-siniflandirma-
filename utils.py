import re
import numpy as np

def preprocess_text(text):
    """
    Verilen metni büyük harfe çevirir, özel karakterleri temizler ve boşluklardan ayırır.
    """
    text = text.upper()
    text = re.sub(r"[^A-ZÇĞİÖŞÜ\s]", "", text)
    return text.strip().split()

def average_word2vec(words, model):
    """
    Kelimelerin Word2Vec vektörlerinin ortalamasını döndürür.
    Kelime modelde yoksa None döner.
    """
    vectors = [model.wv[word] for word in words if word in model.wv]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)
