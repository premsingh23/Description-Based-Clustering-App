import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """Basic text preprocessing."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = [t for t in word_tokenize(text) if t not in stop_words]
    return " ".join(tokens)


def preprocess_dataframe(df, text_column: str) -> None:
    df[text_column] = df[text_column].astype(str).apply(clean_text)
