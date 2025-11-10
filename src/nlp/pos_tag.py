import pandas as pd
import spacy

# Load English NLP  spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_pos_tagging(input_path, output_path):
    df = pd.read_csv(input_path)

    print("POS tagging using spaCy")

    # to count adjectives, noun, verbs in each review
    pos_tags = []
    adj_counts  = []
    noun_counts = []
    verb_counts = []

    for text in df["cleaned_review"].astype(str):
        doc = nlp(text)

        tags = [(token.text, token.pos_) for token in doc]
        pos_tags.append(tags)

        adj_counts.append(sum(1 for token in doc if token.pos_ == "ADJ"))
        noun_counts.append(sum(1 for token in doc if token.pos_ == "NOUN"))
        verb_counts.append(sum(1 for token in doc if token.pos_ == "VERB"))

    df["pos_tags"] = pos_tags
    df["adj_count"] = adj_counts
    df["noun_count"] = noun_counts
    df["verb_count"] = verb_counts

    df.to_csv(output_path, index=False)
    print(f"POS tagging completed and saved to: {output_path}")


if __name__ == "__main__":
    input_file = "data/processed/reviews_cleaned.csv"
    output_file = "data/processed/reviews_pos.csv"

    extract_pos_tagging(input_file, output_file)
