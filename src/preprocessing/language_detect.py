# language_detect.py

import pandas as pd
from langdetect import detect, DetectorFactory
import os

DetectorFactory.seed = 0  # For consistent language predictions

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def run_language_detection(input_path, output_path):
    df = pd.read_csv(input_path)

    print("🔍 Detecting languages...")
    df["language"] = df["review"].apply(detect_language)

    df.to_csv(output_path, index=False)
    print(f"✅ Language detection complete.\nSaved to: {output_path}")


if __name__ == "__main__":
    input_file = "data/raw/reviews.csv"
    output_file = "data/processed/reviews_language.csv"

    run_language_detection(input_file, output_file)
