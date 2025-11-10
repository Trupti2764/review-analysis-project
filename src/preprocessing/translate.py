import pandas as pd
from googletrans import Translator

text_translator = Translator()

def translate_review(text):
    try:
        translated = text_translator.translate(text).text
        return translated
    except:
        return text  # returns original text if translation fails


def run_translation(input_path, output_path):
    df =pd.read_csv(input_path)

    print("  Translating reviews ")

    df['review_translated'] = df.apply(
        lambda row: translate_review(row['review']) if row['language'] != "en" else row['review'],
        axis=1
    )

    df.to_csv(output_path, index=False)
    print(f" file saved to: {output_path}")


if __name__ == "__main__":
    input_file = "data/processed/reviews_language.csv"
    output_file = "data/processed/reviews_translated.csv"

    run_translation(
        input_file, 
        output_file
    )
