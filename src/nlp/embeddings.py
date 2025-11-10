import pandas as pd
import spacy
import pickle
import os
from gensim.models import Word2Vec, KeyedVectors

# Load spaCy english NLP  model
nlp = spacy.load("en_core_web_sm")

# Using  Word2Vec embeddings
def run_word2vec(input_path, model_output="models/word2vec.model", vector_output="data/processed/word2vec_vectors.pkl"):
    # Load cleaned reviews
    df = pd.read_csv(input_path)
    reviews = df["cleaned_review"].astype(str).tolist()

    print("Tokenizing usimg spaCy for Word2Vec")
    tokenized_reviews = []
    for doc in nlp.pipe(reviews, batch_size=50, disable=["ner", "parser"]):
        tokens = [token.text.lower() for token in doc if token.is_alpha]
        tokenized_reviews.append(tokens)

    print(" Training Word2Vec model in progress")
    w2v_model = Word2Vec(
        sentences=tokenized_reviews,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1  # skip-gram
    )

    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    w2v_model.save(model_output)
    print(f"Word2Vec model saved to {model_output}")

    print(" Creating embeddings from word2Vec")
    review_vectors = []
    for tokens in tokenized_reviews:
        vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if vecs:
            review_vectors.append(sum(vecs)/len(vecs))
        else:
            review_vectors.append([0]*w2v_model.vector_size)

    os.makedirs(os.path.dirname(vector_output), exist_ok=True)
    with open(vector_output, "wb") as f:
        pickle.dump(review_vectors, f)

    print(f" Embeddings saved to {vector_output}")


# Using GloVe embeddings
def load_glove(glove_path="models/glove/glove.6B.100d.txt"):
    print("Loading GloVe embeddings")
    glove_file_w2v = glove_path + ".word2vec"
    if not os.path.exists(glove_file_w2v):
        from gensim.scripts.glove2word2vec import glove2word2vec
        glove2word2vec(glove_input_file=glove_path, word2vec_output_file=glove_file_w2v)
    glove_vectors = KeyedVectors.load_word2vec_format(glove_file_w2v, binary=False)
    print(f" GloVe vectors loaded from {glove_path}")
    return glove_vectors

def run_glove(input_path, glove_path="models/glove/glove.6B.100d.txt", vector_output="data/processed/glove_vectors.pkl"):
    df = pd.read_csv(input_path)
    reviews = df["cleaned_review"].astype(str).tolist()

    print("Tokenizing text with spaCy for GloVe...")
    tokenized_reviews = []
    for doc in nlp.pipe(reviews, batch_size=50, disable=["ner", "parser"]):
        tokens = [token.text.lower() for token in doc if token.is_alpha]
        tokenized_reviews.append(tokens)

    glove_vectors = load_glove(glove_path)

    print(" Creating embeddings from GloVe...")
    review_vectors = []
    for tokens in tokenized_reviews:
        vecs = [glove_vectors[word] for word in tokens if word in glove_vectors]
        if vecs:
            review_vectors.append(sum(vecs)/len(vecs))
        else:
            review_vectors.append([0]*glove_vectors.vector_size)

    os.makedirs(os.path.dirname(vector_output), exist_ok=True)
    with open(vector_output, "wb") as f:
        pickle.dump(review_vectors, f)

    print(f"✅ Review embeddings saved to {vector_output}")

if __name__ == "__main__":
    input_file = "data/processed/reviews_ner.csv"


    w2v_model_file = "models/word2vec.model"
    w2v_vectors_file = "output/embeddings/word2vec_vectors.pkl"
    run_word2vec(input_file, w2v_model_file, w2v_vectors_file)

    
    glove_file = "models/glove/glove.6B.100d.txt"
    glove_vectors_file = "output/embeddings/glove_vectors.pkl"
    run_glove(input_file, glove_file, glove_vectors_file)
