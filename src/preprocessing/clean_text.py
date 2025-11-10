import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Downloads needed for nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_review(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]+", " ", text)  #  keep only letters , remove punctuation and numbers
    text = re.sub(r"\s+", " ", text).strip()

    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]

    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def apply_cleaning(input_path, output_path):
    df = pd.read_csv(input_path)

    print("Cleaning text...")

    df["cleaned_review"] =  df["review_translated"].apply(preprocess_review)

    df.to_csv(output_path,  index=False)
    print(f"Cleaning complete.\nSaved to: {output_path}")


if __name__ == "__main__":
    input_file = "data/processed/reviews_translated.csv"
    output_file = "data/processed/reviews_cleaned.csv"

    apply_cleaning(input_file, output_file)
