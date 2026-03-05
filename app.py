from flask import Flask, render_template, request
from predict import analyze_text

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():

    human = None
    ai = None
    comment = ""

    if request.method == "POST":

        text = request.form["text"]

        human, ai = analyze_text(text)

        if human > ai:

            comment = f"""
            Bu metin %{human} oranında insan yazımına yakındır.
            Sözcük seçimi ve cümle yapısı insan yazım özellikleri göstermektedir.
            """

        else:

            comment = f"""
            Bu metin %{ai} oranında yapay zeka üretimine yakındır.
            Cümle yapıları ve anlatım biçimi yapay zeka üretimine benzemektedir.
            """

    return render_template(
        "index.html",
        human=human,
        ai=ai,
        comment=comment
    )

if __name__ == "__main__":
    app.run(debug=True)