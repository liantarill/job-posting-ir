import re
import string
import nltk

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK resources if not already present."""
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

download_nltk_data()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text: str) -> str:
    """
    Clean and preprocess a raw text string.

    Steps:
        1. Lowercase
        2. Remove punctuation and special characters
        3. Tokenize
        4. Remove stopwords
        5. Stem tokens

    Args:
        text: Raw input string.

    Returns:
        Cleaned and stemmed string ready for retrieval.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove punctuation and digits
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and stem
    tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words and len(token) > 1
    ]

    return ' '.join(tokens)


def tokenize_query(query: str) -> list[str]:
    """
    Tokenize and preprocess a query into a list of tokens.
    Used by BM25 which expects a list instead of a joined string.

    Args:
        query: Raw query string.

    Returns:
        List of preprocessed tokens.
    """
    processed = preprocess_text(query)
    return processed.split() if processed else []
