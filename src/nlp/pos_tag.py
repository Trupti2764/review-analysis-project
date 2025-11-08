# pos_tag.py

import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize

# ✅ FIX: Download required NLTK models
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

def run_pos_tagging(input_path, output_path):
    df = pd.read_csv(input_path)

    print("🔤 Performing POS tagging...")

    df["pos_tags"] = df["cleaned_review"].apply(lambda x: pos_tag(word_tokenize(str(x))))

    df["adj_count"] = df["pos_tags"].apply(lambda tags: len([t for t in tags if t[1].startswith("JJ")]))
    df["noun_count"] = df["pos_tags"].apply(lambda tags: len([t for t in tags if t[1].startswith("NN")]))
    df["verb_count"] = df["pos_tags"].apply(lambda tags: len([t for t in tags if t[1].startswith("VB")]))

    df.to_csv(output_path, index=False)
    print(f"✅ POS tagging complete.\nSaved to: {output_path}")


if __name__ == "__main__":
    input_file = "data/processed/reviews_cleaned.csv"
    output_file = "data/processed/reviews_pos.csv"

    run_pos_tagging(input_file, output_file)
