import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# Step 1: Load the dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

df = df.rename(columns={
    "v1": "label",
    "v2": "message"
})
# Step 5: Apply TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["message"])

# Step 6: Labels
y = df["label"]

print(X.shape)
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Train-Test Split

# Create model
model = MultinomialNB()

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model trained successfully!")
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

import joblib

joblib.dump(model, "model/spam_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("Model saved successfully!")