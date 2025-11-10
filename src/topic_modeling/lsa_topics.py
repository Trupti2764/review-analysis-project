import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Output folders
REPORTS_DIR = "output/reports/"
VISUALS_DIR = "output/visuals/"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

# Loading cleaned review text
df = pd.read_csv("data/processed/reviews_ner.csv")
reviews_texts = df["cleaned_review"].astype(str).tolist()

print("Performing LSA Topic Modeling")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
)
tfidf_features = vectorizer.fit_transform(reviews_texts)

# LSA with TruncatedSVD
n_topics = 5   # Extraction of 5 prominent topics
svd_model = TruncatedSVD(n_components=n_topics, random_state=42)
lsa_results = svd_model.fit_transform(tfidf_features)

terms = vectorizer.get_feature_names_out()

# Extracting topic keywords
topic_keywords = {}

for index, component in enumerate(svd_model.components_):
    # finding top 10 keywords for each topic
    top_indices = component.argsort()[-10:][::-1]
    top_terms = [terms[i] for i in top_indices]
    topic_keywords[f"Topic {index+1}"] = top_terms

# Saving topic keywords to a report
output_path = os.path.join(REPORTS_DIR, "topics_lsa.txt")
with open(output_path, "w", encoding="utf-8") as f:
    for topic, keywords in topic_keywords.items():
        f.write(f"{topic}:\n")
        f.write(", ".join(keywords) + "\n\n")

print(f"LSA topics saved to {output_path}")

#  Visualization of plots
plt.figure(figsize=(10, 6))
topic_strength = svd_model.explained_variance_ratio_

plt.bar([f"Topic {i+1}" for i in range(n_topics)], topic_strength)
plt.title("LSA Topic Strength (Explained Variance)")
plt.ylabel("Strength")
plt.xlabel("Topics")
plt.tight_layout()

visual_file = os.path.join(VISUALS_DIR, "lsa_topic_barchart.png")
plt.savefig(visual_file)
plt.close()

print(f"Topic bar chart saved to {visual_file}")
