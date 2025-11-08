# bow_tfidf.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import os

def run_bow_tfidf(input_path, bow_output, tfidf_output, model_dir="models"):
    # Read cleaned data
    df = pd.read_csv(input_path)
    texts = df["cleaned_review"].astype(str).tolist()

    # Create folder for models
    os.makedirs(model_dir, exist_ok=True)

    print("🧠 Generating Bag-of-Words (BoW) vectors...")

    bow_vectorizer = CountVectorizer(max_features=3000)
    bow_matrix = bow_vectorizer.fit_transform(texts)

    # Save BoW vectors and model
    with open(bow_output, "wb") as f:
        pickle.dump((bow_matrix, bow_vectorizer), f)

    print(f"✅ BoW saved to {bow_output}")

    print("\n🧠 Generating TF-IDF vectors...")

    tfidf_vectorizer = TfidfVectorizer(max_features=3000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    # Save TF-IDF vectors and model
    with open(tfidf_output, "wb") as f:
        pickle.dump((tfidf_matrix, tfidf_vectorizer), f)

    print(f"✅ TF-IDF saved to {tfidf_output}")
    print("✅ Feature extraction completed!")


if __name__ == "__main__":
    input_file = "data/processed/reviews_ner.csv"

    bow_file = "data/processed/bow_vectors.pkl"
    tfidf_file = "data/processed/tfidf_vectors.pkl"

    run_bow_tfidf(input_file, bow_file, tfidf_file)
