import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def analyze_text(text):

    X = vectorizer.transform([text])

    probs = model.predict_proba(X)[0]

    human = probs[0] * 100
    ai = probs[1] * 100

    return round(human,2), round(ai,2)