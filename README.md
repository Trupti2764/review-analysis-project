📝 Comprehensive Product Review Analysis (Classical NLP Approach)

This project performs an end-to-end analysis of customer reviews for an e-commerce product using classical Natural Language Processing techniques.
The aim is to transform raw customer feedback into structured insights through sentiment analysis, topic modeling, semantic similarity, clustering, and evidence-based QA.

✅ No Transformer models
✅ No generative AI
✅ Uses only classical NLP methods: TF-IDF, Word2Vec, LSA, VADER, POS/NER, and a lightweight LSTM.

📌 1. Project Overview

The project simulates the workflow of an NLP engineer analyzing product reviews for an e-commerce platform (Amazon/Flipkart).
Starting from web scraping, reviews are cleaned, processed, analyzed, and summarized to answer key customer questions and highlight major product themes.

The full pipeline covers:

✅ Web scraping

✅ Language detection & translation

✅ Text cleaning & normalization

✅ Syntactic & semantic analysis

✅ Topic extraction

✅ Sentiment classification

✅ Semantic similarity

✅ Clustering & review summarization

✅ Evidence-supported QA

📌 2. Features Implemented
🔹 Data Acquisition

Scraping performed using Selenium.

Handles pagination and dynamic loading.

Raw data stored in data/raw/.

Cleaned and processed reviews stored in data/processed/.

🔹 Language Detection & Translation

Detects English/Hindi reviews.

Translates Hindi → English using googletrans.

Ensures full sentiment consistency after translation.

🔹 Preprocessing

HTML/emoji cleanup

Lowercasing

Tokenization (spaCy)

Stopword removal

Lemmatization

Duplicate handling

Outputs cleaned files to data/processed/

🔹 Syntactic Analysis

POS tagging (spaCy)

Named Entity Recognition (rule-based/statistical)

Extracts meaningful adjectives, nouns, and verbs used in customer feedback.

🔹 Semantic Analysis

TF-IDF vectorization

Word2Vec training (gensim)

Cosine similarity across review embeddings

Identification of recurring terms and semantic clusters

🔹 Topic Modeling

LSA used to derive 3–5 major topics

Extracts top keywords representing customer discussion themes

Topic report stored in output/reports/topics_lsa.txt

🔹 Sentiment Analysis

VADER lexicon-based sentiment scoring

LSTM model trained on VADER pseudo-labels

Sentiment distribution visualized as plots

Outputs stored in output/processed/

🔹 Review Summarization

Clusters reviews using semantic similarity

Extracts representative reviews per cluster

Produces concise human-readable summaries

Stored in output/reports/review_summary.txt

🔹 Simulated QA

Generates common customer questions such as:

Battery life

Performance

Value for money

Common defects

Recommendation

Answers are evidence-supported using extracted topics, sentiments, and representative reviews.

Outputs:

simulated_qa.txt

qa_evidence_evaluation.txt

qa_topic_match_heatmap.png

📌 3. Project Structure
project/
│
├── data/
│   ├── raw/                 # Raw scraped reviews
│   └── processed/           # Cleaned reviews, vectors, labels
│
├── models/                  # Word2Vec, GloVe files
│
├── output/
│   ├── embeddings/          # Word2Vec & GloVe vectors
│   ├── processed/           # TF-IDF, sentiment outputs, LSTM model
│   ├── reports/             # Topics, summaries, QA results
│   └── visuals/             # Plots & heatmaps
│
└── src/
    ├── preprocessing/       # Cleaning, translation, language detection
    ├── nlp/                 # TF-IDF, embeddings, POS, NER, sentiment
    ├── topic_modeling/      # LSA topic extraction
    ├── summarization/       # Cluster + representative review extraction
    ├── qa/                  # Simulated QA + evaluation
    └── scraping/            # Selenium-based reviewers scraper

📌 4. How to Run
Step 1: Install dependencies
pip install -r requirements.txt

Step 2: Activate virtual environment
./venv/Scripts/Activate.ps1

Step 3: Run scraping
python src/scraping/scrape_reviews_selenium.py

Step 4: Run preprocessing
python src/preprocessing/clean_text.py

Step 5: Run full NLP pipeline modules

(Examples)

python src/nlp/bow_tfidf.py
python src/nlp/embeddings.py
python src/nlp/sentiment.py
python src/topic_modeling/lsa_topics.py
python src/summarization/review_clusters.py
python src/qa/qa_answers.py

📌 5. Outputs

Key reports generated:

✅ Sentiment distribution

✅ Topic keywords

✅ Cluster summaries

✅ Representative reviews

✅ Evidence-based QA

✅ Embedding similarity charts

All outputs are stored in the output/ directory.
