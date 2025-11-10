import pandas as pd
from langdetect import detect, DetectorFactory
import os

DetectorFactory.seed =  0  # Keep langdetect output same each run

def get_lang(text):
    try:
        return detect(text)
    except:
        return "unknown"

def add_language_column(input_path, output_path):
    # reading input data
    df  = pd.read_csv(input_path)

    print(" Language detection in progress")
    # adding detected langugae to each row
    df['language'] = df['review'].apply(get_lang)

    df.to_csv(output_path, index=False)
    print(f" Saved at : {output_path}")


if __name__ == "__main__":
    input_file = "data/raw/reviews.csv"
    output_file = "data/processed/reviews_language.csv"

    add_language_column(
        input_file, 
        output_file
    )
