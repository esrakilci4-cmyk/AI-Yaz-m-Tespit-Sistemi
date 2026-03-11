from flask import Flask, render_template, request
import pickle, os

app = Flask(__name__)

# -----------------------------
# 1️⃣ Model ve vectorizer'ı yükleme
# -----------------------------
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("model.pkl veya vectorizer.pkl bulunamadı! Önce train_model.py çalıştırın.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

print("✅ Model ve vectorizer başarıyla yüklendi!")

# -----------------------------
# 2️⃣ Ana sayfa route
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -----------------------------
# 3️⃣ Tahmin route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text_input', '').strip()
    
    if not text:
        return render_template('index.html', prediction=None, text_input=text)

    # Tahmin
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    # Basit oran hesaplama
    if prediction == "AI":
        ai_ratio = 80
        human_ratio = 20
        yorum = "Bu metin büyük olasılıkla Yapay Zeka tarafından yazılmış."
    else:
        ai_ratio = 20
        human_ratio = 80
        yorum = "Bu metin büyük olasılıkla İnsan tarafından yazılmış."

    return render_template('index.html',
                           prediction=prediction,
                           text_input=text,
                           human=human_ratio,
                           ai=ai_ratio,
                           yorum=yorum,
                           grafik="https://i.ibb.co/Zm3t0jx/sample-graph.png")  # Örnek grafik URL

# -----------------------------
# 4️⃣ Flask çalıştırma
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)