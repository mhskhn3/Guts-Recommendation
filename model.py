# model.py - Clean version for deployment
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import joblib
import os
import time

# Download NLTK data at module load
print("Downloading NLTK resources for model...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
print("NLTK resources downloaded for model!")

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        """Advanced text preprocessing with multiple cleaning steps"""
        if not isinstance(text, str) or pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters and digits but keep basic punctuation for context
        text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]

        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return ' '.join(tokens)

    def batch_preprocess(self, texts, batch_size=1000):
        """Preprocess texts in batches to manage memory"""
        processed_texts = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            processed_batch = [self.preprocess_text(text) for text in batch]
            processed_texts.extend(processed_batch)

            if (i // batch_size) % 10 == 0:
                print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

        return processed_texts

class TicketRecommendationSystem:
    def __init__(self):
        self.vectorizer = None
        self.svd = None
        self.nn_index = None
        self.ticket_data = None
        self.embedding_matrix = None
        self.ticket_ids = None
        self.preprocessor = TextPreprocessor()

    def load_model(self, filepath="ticket_recommender_model.joblib"):
        """Load a pre-trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")

        print(f"Loading model from {filepath}...")
        model_data = joblib.load(filepath)

        self.vectorizer = model_data['vectorizer']
        self.svd = model_data['svd']
        self.nn_index = model_data['nn_index']
        self.ticket_ids = model_data['ticket_ids']
        self.ticket_data = model_data['ticket_data']
        self.embedding_matrix = model_data['embedding_matrix']

        print("✓ Model loaded successfully!")
        print(f"✓ Loaded data shape: {self.ticket_data.shape}")
        return self

    def find_similar_tickets(self, query_description, top_n=5, similarity_threshold=0.7):
        """Find similar tickets for a given description"""
        if self.vectorizer is None or self.svd is None:
            raise ValueError("Model not trained. Please call build_model first.")

        # Preprocess query
        processed_query = self.preprocessor.preprocess_text(query_description)

        if not processed_query:
            return {"error": "Query resulted in empty text after preprocessing"}

        # Transform query to same vector space
        query_tfidf = self.vectorizer.transform([processed_query])
        query_embedding = self.svd.transform(query_tfidf)
        query_embedding = normalize(query_embedding)

        # Find nearest neighbors
        distances, indices = self.nn_index.kneighbors(query_embedding, n_neighbors=top_n)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity_score = 1 - dist  # Convert distance to similarity

            if similarity_score >= similarity_threshold:
                results.append({
                    'ticket_id': str(self.ticket_ids[idx]),
                    'similarity_score': float(similarity_score),
                    'original_description': str(self.ticket_data['description'].iloc[idx]),
                    'processed_description': str(self.ticket_data['processed_description'].iloc[idx]) if 'processed_description' in self.ticket_data.columns else ""
                })

        return results
