import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
ps = PorterStemmer()


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '',string.punctuation))

    words = word_tokenize(text)

    stop_words = set(stopwords.words("english"))

    filtered_words = []

    for word in words:
        if word not in stop_words:
            filtered_words.append(ps.stem(word))

    return " ".join(filtered_words)