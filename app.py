from flask import Flask, render_template, request
import joblib
from preprocess import preprocess_text

app = Flask(__name__)


model = joblib.load("model/spam_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    message = ""

    if request.method == "POST":
        message = request.form["message"]

        cleaned_message = preprocess_text(message)
        message_vector = tfidf.transform([cleaned_message])

        prediction = model.predict(message_vector)[0]
        probabilities = model.predict_proba(message_vector)
        confidence = round(max(probabilities[0]) * 100, 2)
    return render_template(
            "index.html",
            prediction=prediction,
            confidence=confidence,
            message=message
)

if __name__ == "__main__":
    app.run()