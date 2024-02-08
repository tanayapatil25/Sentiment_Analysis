import string
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed_texts = []
        for text in X:
            # Convert to lowercase
            text = text.lower()

            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))

            # Tokenize the text
            tokens = word_tokenize(text)

            # Remove stopwords
            tokens = [token for token in tokens if token not in stopwords.words('english')]

            # Rejoin tokens into a string
            processed_text = ' '.join(tokens)
            processed_texts.append(processed_text)

        return processed_texts








