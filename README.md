# Harcama Açıklaması Sınıflandırma Projesi

Bu proje, banka işlem açıklamalarını harcama kategorilerine otomatik olarak sınıflandırır. Word2Vec ve TF-IDF modelleriyle açıklamalar vektörleştirilir. Ardından cosine benzerliği, Jaccard benzerliği ve semantik analiz ile en yakın açıklamalar bulunur.

## Gereksinimler
Python 3.8+ gereklidir. Kurulum için:
```
pip install -r requirements.txt
```

## Kullanım

### 1. Model Eğitimi
TF-IDF ve Word2Vec modellerini eğitmek için:
```
python train.py
```

### 2. Açıklama Sınıflandırma
Word2Vec modeliyle benzer açıklamaları bulmak için:
```
python classify.py --model word2vec_cbow_50.model --input "SHELL PETROL İSTANBUL"
```

TF-IDF ile sınıflandırmak için:
```
python classify.py --model tfidf_model.pkl --mode tfidf --input "MİGROS MARKET"
```

### 3. Değerlendirme
Cosine ve Jaccard benzerliği ile ilk 5 benzer açıklamayı görmek için:
```
python evaluate.py
```

## Veri Seti
Veri seti `harcama_aciklamalari_raw.csv` dosyasında yer alır. En az 200.000 işlem açıklaması içerir.

## Geliştiren
GitHub: [KhaledMohamedAlyHadhoudMohamed](https://github.com/KhaledMohamedAlyHadhoudMohamed)
