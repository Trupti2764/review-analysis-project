import pandas as pd
import gensim
from gensim.models import Word2Vec
import spacy
import pickle
import os

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def run_word2vec(input_path, model_output="models/word2vec.model", vector_output="data/processed/word2vec_vectors.pkl"):
    # Load cleaned reviews
    df = pd.read_csv(input_path)
    reviews = df["cleaned_review"].astype(str).tolist()

    print("🧠 Tokenizing text with spaCy...")
    tokenized_reviews = []
    for doc in nlp.pipe(reviews, batch_size=50, disable=["ner", "parser"]):
        tokens = [token.text.lower() for token in doc if token.is_alpha]  # keep only alphabetic words
        tokenized_reviews.append(tokens)

    print("🧠 Training Word2Vec model...")
    w2v_model = Word2Vec(
        sentences=tokenized_reviews,
        vector_size=100,  # size of embeddings
        window=5,         # context window
        min_count=2,      # ignore rare words
        workers=4,        # parallel threads
        sg=1              # skip-gram (better for small dataset)
    )

    # Save Word2Vec model
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    w2v_model.save(model_output)
    print(f"✅ Word2Vec model saved to {model_output}")

    # Create review vectors by averaging word embeddings
    print("🧠 Generating review embeddings (average of word vectors)...")
    review_vectors = []
    for tokens in tokenized_reviews:
        vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if vecs:
            review_vectors.append(sum(vecs)/len(vecs))
        else:
            review_vectors.append([0]*w2v_model.vector_size)

    # Save review vectors
    os.makedirs(os.path.dirname(vector_output), exist_ok=True)
    with open(vector_output, "wb") as f:
        pickle.dump(review_vectors, f)

    print(f"✅ Review embeddings saved to {vector_output}")


if __name__ == "__main__":
    input_file = "data/processed/reviews_ner.csv"
    model_file = "models/word2vec.model"
    review_vectors_file = "data/processed/word2vec_vectors.pkl"

    run_word2vec(input_file, model_file, review_vectors_file)
