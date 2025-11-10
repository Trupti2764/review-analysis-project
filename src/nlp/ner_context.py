import pandas as pd
import spacy
from collections import defaultdict, Counter
import pickle
import os

# Directories
PROCESSED_DIR = "output/processed/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# For loading cleaned reviews
input_file = "data/processed/reviews_ner.csv"
df = pd.read_csv(input_file)
reviews = df["cleaned_review"].astype(str).tolist()

# Load English NLP spacy model
nlp = spacy.load("en_core_web_sm")

# Parameters for context window
CONTEXT_WINDOW = 3  # number of words before and after entity to include

entity_counts = Counter()
entity_context = defaultdict(list)

print("Extract entities and context from reviews")

for doc in nlp.pipe(reviews, batch_size=50, disable=["parser"]):
    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "ORG", "GPE", "LOC"]:  # focusing on relevant types
            entity_text = ent.text.lower()
            entity_counts[entity_text] += 1

            # to find context words around entity
            start = max(ent.start - CONTEXT_WINDOW, 0)
            end = min(ent.end + CONTEXT_WINDOW, len(doc))
            context_tokens = [token.text.lower() for token in doc[start:end] if token.is_alpha and token.text.lower() != entity_text]
            entity_context[entity_text].append(context_tokens)

# Save entity frequencies and context
entities_data = {"entity_counter": entity_counts, "entity_context": entity_context}

output_file = os.path.join(PROCESSED_DIR, "entities.pkl")
with open(output_file, "wb") as f:
    pickle.dump(entities_data, f)

print(f" Entity context saved to {output_file}")
