import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Full NLP preprocessing pipeline for a single article string."""
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # 4. Remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    # 5. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Tokenize
    tokens = word_tokenize(text)

    # 7. Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # 8. Lemmatize
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_dataframe(df, text_col="content"):
    """Apply clean_text to an entire DataFrame column."""
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_text)
    return df
