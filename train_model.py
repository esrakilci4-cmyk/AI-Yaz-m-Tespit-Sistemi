import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# 1️⃣ Veri Yükleme (UTF-8 hatalarını önlemek için errors="ignore")
# -----------------------------
ai_texts = []
human_texts = []

# AI metinleri
ai_folder = "ai_texts"
for filename in os.listdir(ai_folder):
    path = os.path.join(ai_folder, filename)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            ai_texts.append(f.read())

# İnsan metinleri
human_folder = "human_texts"
for filename in os.listdir(human_folder):
    path = os.path.join(human_folder, filename)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            human_texts.append(f.read())

# -----------------------------
# 2️⃣ Etiketler
# -----------------------------
texts = ai_texts + human_texts
labels = ["AI"] * len(ai_texts) + ["Human"] * len(human_texts)

# -----------------------------
# 3️⃣ TF-IDF Vectorizer
# -----------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(texts)

# -----------------------------
# 4️⃣ Model Eğitimi
# -----------------------------
model = MultinomialNB()
model.fit(X, labels)

# -----------------------------
# 5️⃣ Pickle ile güvenli kaydetme
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# -----------------------------
# 6️⃣ Bilgilendirme
# -----------------------------
print("✅ Model ve vectorizer başarıyla oluşturuldu ve kaydedildi!")
print(f"Toplam AI metin sayısı: {len(ai_texts)}")
print(f"Toplam İnsan metin sayısı: {len(human_texts)}")