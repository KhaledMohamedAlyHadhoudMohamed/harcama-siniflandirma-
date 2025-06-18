# Harcama Açıklaması Sınıflandırma Projesi

Bu proje, banka işlem açıklamalarını 5 farklı kategoriye (Yeme-İçme, Ulaşım, Fatura, Alışveriş, Sağlık) göre otomatik olarak sınıflandırmayı amaçlar. Word2Vec ve TF-IDF gibi doğal dil işleme yöntemleri kullanılmıştır.

## 📊 Kullanılan Modeller

- **Word2Vec CBOW (vector_size=50)** – anlamsal benzerlikleri yakalamak için
- **TF-IDF** – frekansa dayalı metin karşılaştırmaları için

## 📁 Dosya Açıklamaları

- `harcama_aciklamalari_raw.csv` : Simüle edilmiş 200K banka açıklama verisi
- `train.py` : Word2Vec ve TF-IDF model eğitim scripti
- `classify.py` : Giriş açıklamasına göre benzer açıklamaları listeler
- `utils.py` : Metin temizleme ve ortalama vektör hesaplama yardımcıları
- `requirements.txt` : Gerekli Python kütüphaneleri

## 🧪 Örnek Kullanımlar

### 1. Model Eğitimi

```bash
python train.py
```

### 2. Açıklama Sınıflandırma

#### Word2Vec ile:
```bash
python classify.py --model word2vec_cbow_50.model --input "SHELL PETROL İSTANBUL" --mode w2v
```

#### TF-IDF ile:
```bash
python classify.py --model tfidf_model.pkl --input "SHELL PETROL İSTANBUL" --mode tfidf
```

## 💡 Notlar

- Word2Vec modeli `word2vec_cbow_50.model` dosyasına, TF-IDF modeli `tfidf_model.pkl` dosyasına kaydedilir.
- Sınıflandırma sonucu benzer açıklamalar ve benzerlik skoru ile birlikte döner.
