# translate.py

import pandas as pd
from googletrans import Translator

translator = Translator()

def translate_text(text):
    try:
        translated = translator.translate(text).text
        return translated
    except:
        return text  # If translation fails, keep original


def run_translation(input_path, output_path):
    df = pd.read_csv(input_path)

    print("🌍 Translating non-English reviews...")

    df["review_translated"] = df.apply(
        lambda row: translate_text(row["review"]) if row["language"] != "en" else row["review"],
        axis=1
    )

    df.to_csv(output_path, index=False)
    print(f"✅ Translation complete.\nSaved to: {output_path}")


if __name__ == "__main__":
    input_file = "data/processed/reviews_language.csv"
    output_file = "data/processed/reviews_translated.csv"

    run_translation(input_file, output_file)
