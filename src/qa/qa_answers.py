import os
import pickle
import pandas as pd
import numpy as np

PROCESSED_DIR = "output/processed/"
REPORTS_DIR = "output/reports/"

os.makedirs(REPORTS_DIR, exist_ok=True)

# Loading Cleaned Review text
print("Loading cleaned reviews")
df = pd.read_csv("data/processed/reviews_ner.csv")
reviews = df["cleaned_review"].astype(str).tolist()

#  Load VADER Sentiment 
print("Loading VADER sentiment scores")
with open(os.path.join(PROCESSED_DIR, "vader_sentiment_scores.pkl"), "rb") as f:
    vader_scores = pickle.load(f)

#  Load LSA Topics 
print("Loading LSA topics...")
with open(os.path.join(REPORTS_DIR, "topics_lsa.txt"), "r", encoding="utf-8") as f:
    topics_text = f.read().split("\n")

# Load Word2Vec for Feature Similarity
print(" Loading Word2Vec embeddings")
with open("output/embeddings/word2vec_vectors.pkl", "rb") as f:
    w2v_embeddings = pickle.load(f)

#  Defining few Questions 
questions = [
    "Is the product quality good?",
    "Does it have any common defects or issues?",
    "Is the product value for money?",
    "How satisfied are customers with performance?",
    "Would customers recommend this product?"
]

def average_sentiment_for_keywords(keywords, reviews, vader_scores):
    """Compute average sentiment for reviews containing at least one keyword."""
    indices = [i for i, r in enumerate(reviews) if any(k.lower() in r.lower() for k in keywords)]
    if not indices:
        return 0  # Neutral if no matching reviews
    return np.mean([vader_scores[i]["compound"] for i in indices])

# Maping questions to relevant keywords or topics
question_keywords = {
    questions[0]: ["good", "excellent", "quality", "satisfactory", "value"],
    questions[1]: ["broken", "defect", "problem", "damage", "issue"],
    questions[2]: ["value", "money", "expensive", "cheap", "worth"],
    questions[3]: ["performance", "speed", "battery", "sound", "functioning"],
    questions[4]: ["recommend", "suggest", "buy", "repurchase", "happy"]
}

#  Generating most suitable Answers 
print(" Generating answers based on sentiment and topics")
qa_pairs = []

for q in questions:
    keywords = question_keywords[q]
    avg_sentiment = average_sentiment_for_keywords(keywords, reviews, vader_scores)
    
    if avg_sentiment > 0.05:
        sentiment_text = "mostly positive"
    elif avg_sentiment < -0.05:
        sentiment_text = "mostly negative"
    else:
        sentiment_text = "neutral/mixed"
    
    # Check if keywords appear in LSA topics for more context
    related_topics = [t for t in topics_text if any(k.lower() in t.lower() for k in keywords)]
    
    topic_text = f" Related topics: {', '.join(related_topics)}." if related_topics else ""
    
    answer = f"The sentiment for this aspect is {sentiment_text}.{topic_text}"
    qa_pairs.append((q, answer))

#  Save QA Output 
output_path = os.path.join(REPORTS_DIR, "simulated_qa.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(" SIMULATED QUESTION-ANSWERING BASED ON REVIEWS\n\n")
    for q, a in qa_pairs:
        f.write(f"Q: {q}\nA: {a}\n\n")

print(f" QA results saved to {output_path}")

# Printing QA
for q, a in qa_pairs:
    print(f"Q: {q}\nA: {a}\n")
