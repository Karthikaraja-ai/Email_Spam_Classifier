import joblib
from preprocess import preprocess_text
model = joblib.load("model/spam_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")
message = input("Enter an email: ")
cleaned_message = preprocess_text(message)
message_vector = tfidf.transform([cleaned_message])
prediction = model.predict(message_vector)
print("Prediction:", prediction[0])