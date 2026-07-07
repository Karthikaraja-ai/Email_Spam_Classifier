import pandas as pd

# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Remove unnecessary columns
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])


# Rename columns
df = df.rename(columns={
    "v1": "label",
    "v2": "message"
})

# Remove duplicate rows
df = df.drop_duplicates()

# Display information11
print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df["label"].value_counts())