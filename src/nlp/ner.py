import pandas as pd
import spacy

# Load English NLP spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_entities_spacy(text):
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    return entities


def apply_ner(input_path, output_path):
    df = pd.read_csv(input_path)

    print("Named Entity Recognition (NER) using spaCy")

    df["entities"] = df["cleaned_review"].astype(str).apply(extract_entities_spacy)

    df.to_csv(output_path, index=False)
    print(f" NER Done and saved to: {output_path}")


if __name__ == "__main__":
    input_file = "data/processed/reviews_.csv"
    output_file = "data/processed/reviews_ner.csv"

    apply_ner(input_file, output_file)
