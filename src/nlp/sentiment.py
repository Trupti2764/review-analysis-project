import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# TensorFlow Keras imports
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Lexicon-based sentiment analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Need dictionary loader
def load_embedding_dict(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        print(" Loaded embeddings as dictionary")
        return data

    print(" ERROR: Embeddings are NOT a dictionary")
    return None


# Directories 
PROCESSED_DIR = "output/processed/"
VISUALS_DIR = "output/visuals/"
REPORTS_DIR = "output/reports/"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# Load Reviews 
df = pd.read_csv("data/processed/reviews_ner.csv")
reviews = df["cleaned_review"].astype(str).tolist()


# VADER Sentiment 
print(" Computing VADER sentiment scores...")
analyzer = SentimentIntensityAnalyzer()
vader_scores = [analyzer.polarity_scores(r) for r in reviews]

with open(os.path.join(PROCESSED_DIR, "vader_sentiment_scores.pkl"), "wb") as f:
    pickle.dump(vader_scores, f)

compound_scores = [v['compound'] for v in vader_scores]

plt.figure(figsize=(8, 5))
sns.histplot(compound_scores, bins=20, kde=True)
plt.title("VADER Sentiment Distribution")
plt.xlabel("Compound Score")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "vader_sentiment_distribution.png"))
plt.close()

print(f"VADER sentiment plot saved to {VISUALS_DIR}")

# Save key positive and negative examples
sorted_reviews = sorted(zip(reviews, compound_scores), key=lambda x: x[1])
most_negative = [r for r, s in sorted_reviews[:10]]
most_positive = [r for r, s in sorted_reviews[-10:]]

with open(os.path.join(REPORTS_DIR, "key_phrases_sentiment.txt"), "w", encoding="utf-8") as f:
    f.write("Most Negative Reviews:\n\n")
    for r in most_negative:
        f.write(r + "\n\n")
    f.write("Most Positive Reviews:\n\n")
    for r in most_positive:
        f.write(r + "\n\n")

print(f"Key phrases saved to {REPORTS_DIR}")


# LSTM Sentiment 
print(" Preparing data for LSTM...")

#  FIXED: correct Word2Vec dictionary path
embedding_path = "data/processed/word2vec_dict.pkl"
embeddings = load_embedding_dict(embedding_path)

# Tokenize reviews
tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
word_index = tokenizer.word_index
max_len = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_len)

#  FIXED: derive embedding size from dictionary
embedding_dim = len(next(iter(embeddings.values())))
num_words = len(word_index) + 1

# Build embedding matrix
embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if word in embeddings:
        embedding_matrix[i] = embeddings[word]
    else:
        embedding_matrix[i] = np.random.normal(0, 0.1, embedding_dim)

# Convert VADER compound into labels
y = np.array([1 if s>=0.05 else 0 if s<=-0.05 else 2 for s in compound_scores])
y = np.eye(3)[y]  # one-hot

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM
model = Sequential()
model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix],
                    input_length=max_len, trainable=False))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

# Train model
model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f" LSTM Test Accuracy: {acc:.3f}")

# Save model
model.save("output/processed/lstm_sentiment_model.h5")
print("LSTM sentiment model saved to output/processed/")

sentiment_labels = []
for score in compound_scores:
    if score >= 0.5:
        sentiment_labels.append("Highly Positive")
    elif 0.05 <= score < 0.5:
        sentiment_labels.append("Positive")
    elif -0.05 < score < 0.05:
        sentiment_labels.append("Neutral")
    elif -0.5 <= score <= -0.05:
        sentiment_labels.append("Negative")
    else:
        sentiment_labels.append("Highly Negative")

sentiment_df = pd.DataFrame({
    "review": reviews,
    "compound_score": compound_scores,
    "sentiment_class": sentiment_labels
})

csv_path = os.path.join(PROCESSED_DIR, "review_sentiment_classes.csv")
sentiment_df.to_csv(csv_path, index=False, encoding="utf-8")

print(f"Sentiment classes saved to {csv_path}")
