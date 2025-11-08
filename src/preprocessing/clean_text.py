# clean_text.py

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Run these once
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # remove punctuation and numbers
    text = re.sub(r"\s+", " ", text).strip()

    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]

    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def run_cleaning(input_path, output_path):
    df = pd.read_csv(input_path)

    print("🧹 Cleaning text...")

    df["cleaned_review"] = df["review_translated"].apply(clean_text)

    df.to_csv(output_path, index=False)
    print(f"✅ Cleaning complete.\nSaved to: {output_path}")


if __name__ == "__main__":
    input_file = "data/processed/reviews_translated.csv"
    output_file = "data/processed/reviews_cleaned.csv"

    run_cleaning(input_file, output_file)
