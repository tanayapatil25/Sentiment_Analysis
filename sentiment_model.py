import pandas as pd
from text_preprocessor import TextPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv('Restaurant reviews.csv')

# Drop any rows with missing values in 'Review' column
df = df.dropna(subset=['Review'])

# Define features (X) and target variable (y)
X = df['Review']
positive_keywords = ['good', 'best', 'calm', 'wow', 'great', 'lovely', 'excellent', 'nice', 'tasty', 'amazing']
y = df['Review'].apply(lambda x: 1 if any(keyword in x.lower() for keyword in positive_keywords) else 0)

# Create a pipeline
pipeline = Pipeline([
    ('preprocessor', TextPreprocessor()),
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the pipeline on the entire dataset
pipeline.fit(X, y)

# Save the model to a file using pickle
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)


