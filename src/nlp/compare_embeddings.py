import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os

# Creating directory for visuals if not present 
VISUALS_DIR = "output/visuals/"
os.makedirs(VISUALS_DIR, exist_ok=True)

# Loading cleaned reviews
input_file = "data/processed/reviews_ner.csv"
df = pd.read_csv(input_file)
reviews = df["cleaned_review"].astype(str).tolist()

# Load embeddings ( word2Vec and gloVe)
def load_vectors(vector_path):
    with open(vector_path, "rb") as f:
        return np.array(pickle.load(f))

w2v_vectors = load_vectors("output/embeddings/word2vec_vectors.pkl")
glove_vectors = load_vectors("output/embeddings/glove_vectors.pkl")

# random indices to reduce computation time
sample_size = 500 if len(reviews) > 500 else len(reviews)
random_indices = np.random.choice(len(reviews), sample_size, replace=False)

def compute_similarity_distribution(vectors, name):
    similarity_matrix = cosine_similarity(vectors[random_indices])
    sims = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
    sns.kdeplot(sims, label=name)
    print(f"{name}: Mean similarity = {np.mean(sims):.3f}")

plt.figure(figsize=(12, 7))
compute_similarity_distribution(w2v_vectors, "Word2Vec")
compute_similarity_distribution(glove_vectors, "GloVe")

plt.title("Cosine Similarity Distribution for Different Embeddings")
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.legend()

# Save image to output/visuals
plt.tight_layout()
output_file = os.path.join(VISUALS_DIR, "embedding_similarity_distribution.png")
plt.savefig(output_file)
print(f" Cosine similarity distribution plot saved to {output_file}")

plt.show()

# Compute full similarity lists and visualizing similarity distributions and saving plots
def get_similarity_scores(vectors):
    sim_matrix = cosine_similarity(vectors[random_indices])
    sims = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    return sims

w2v_sims = get_similarity_scores(w2v_vectors)
glove_sims = get_similarity_scores(glove_vectors)

# 1. Print basic statistics
print(" Mean Similarity:")
print(f"Word2Vec: {np.mean(w2v_sims):.4f}")
print(f"GloVe : {np.mean(glove_sims):.4f}")

print("\n Median Similarity:")
print(f" Word2Vec : {np.median(w2v_sims):.4f}")
print(f" GloVe  : {np.median(glove_sims):.4f}")

print("\n Standard Deviation:")
print(f" Word2Vec: {np.std(w2v_sims):.4f}")
print(f" GloVe : {np.std(glove_sims):.4f}")

# 2. Boxplot comparison
plt.figure(figsize=(10, 6))
sns.boxplot(data=[w2v_sims, glove_sims])
plt.xticks([0, 1], ["Word2Vec", "GloVe"])
plt.title("Similarity Score Spread Comparison (Boxplot)")
plt.ylabel("Cosine Similarity")
plt.tight_layout()
boxplot_path = os.path.join(VISUALS_DIR, "embedding_similarity_boxplot.png")
plt.savefig(boxplot_path)
plt.close()
print(f"saved comparision of boxplot to {boxplot_path}")

# 3. Heatmap for similarity subset
subset_size = 40 if sample_size > 40 else sample_size

plt.figure(figsize=(10, 8))
sns.heatmap(
    cosine_similarity(w2v_vectors[random_indices[:subset_size]]),
    cmap="coolwarm"
)
plt.title("Word2Vec Similarity Heatmap (Sample 40 Reviews)")
heatmap_path = os.path.join(VISUALS_DIR, "w2v_similarity_heatmap.png")
plt.savefig(heatmap_path)
plt.close()
print(f"Word2Vec heatmap saved to {heatmap_path}")

plt.figure(figsize=(10, 8))
sns.heatmap(
    cosine_similarity(glove_vectors[random_indices[:subset_size]]),
    cmap="coolwarm"
)
plt.title("GloVe Similarity Heatmap (Sample 40 Reviews)")
heatmap_path2 = os.path.join(VISUALS_DIR, "glove_similarity_heatmap.png")
plt.savefig(heatmap_path2)
plt.close()
print(f" GloVe heatmap saved to {heatmap_path2}")

print("\n Embedding comparison completed.\n")
