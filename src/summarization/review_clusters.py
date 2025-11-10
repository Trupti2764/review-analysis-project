import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = "output/processed/"
VISUALS_DIR = "output/visuals/"
REPORTS_DIR = "output/reports/"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Loading cleaned text reviews
print("Loading cleaned reviews")
df = pd.read_csv("data/processed/reviews_ner.csv")
reviews = df["cleaned_review"].astype(str).tolist()
print(f" Total reviews loaded: {len(reviews)}")

# TF-IDF Vectorization 
print("Generating TF-IDF matrix")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
print("TF-IDF matrix generated with shape:", tfidf_matrix.shape)

# Saving TF-IDF matrix 
with open(os.path.join(PROCESSED_DIR, "tfidf_matrix.pkl"), "wb") as f:
    pickle.dump(tfidf_matrix, f)
print("TF-IDF matrix generated and saved to output/processed/")

# Findng Cosine Similarity 
print("Finding cosine similarity matrix")
similarity_matrix = cosine_similarity(tfidf_matrix)

# Clustering 
print(" Performing review clustering")
NUM_CLUSTERS = 5
clustering = AgglomerativeClustering(
    n_clusters  =NUM_CLUSTERS,
    affinity='cosine',
    linkage='average'
)
cluster_labels = clustering.fit_predict(tfidf_matrix.toarray())
df["cluster"] = cluster_labels

# Finding Representative Reviews 
print("   Extracting representative reviews for each cluster...")
representative_reviews = {}
for cluster_id in range(NUM_CLUSTERS):
    cluster_indices = df[df["cluster"] == cluster_id].index.tolist()
    if not cluster_indices:
        continue

    submatrix = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
    avg_sim = submatrix.mean(axis=1)
    top_indices = np.argsort(avg_sim)[-3:]  # top 3 reviews
    representative_reviews[cluster_id] = [
        reviews[cluster_indices[i]] for i in top_indices
    ]

# Saving Representative Reviews 
output_path = os.path.join(REPORTS_DIR, "review_summary.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(" REVIEW SUMMARY USING SIMILARITY INDEX\n\n")
    for cid, reps in representative_reviews.items():
        f.write(f"\n-----------------------------\n")
        f.write(f"Cluster {cid} – Representative Reviews\n")
        f.write("-----------------------------\n\n")
        for r in reps:
            f.write(f"- {r}\n\n")
print(f"Representative reviews saved to {output_path}")

# Plot Cluster Distribution 
plt.figure(figsize=(8,5))
sns.countplot(x=cluster_labels)
plt.title("Review Cluster Distribution")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plot_path = os.path.join(VISUALS_DIR, "review_cluster_distribution.png")
plt.savefig(plot_path)
plt.close()
print(f"Cluster distribution plot saved to {plot_path}")

print("\n Review summarization completed successfully!\n")
