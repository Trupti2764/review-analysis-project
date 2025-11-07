# 📝 Comprehensive Product Review Analysis (Classical NLP Approach)

This project performs a complete end-to-end analysis of customer reviews for a chosen e-commerce product using **traditional NLP techniques** (non-Transformer based).  
The goal is to extract, clean, analyze, and interpret genuine customer feedback and convert it into actionable insights.

✅ **Strict constraint:** No Transformer models, no generative AI models.  
✅ Only classical NLP methods such as TF-IDF, Word2Vec, LSA, lexicon-based sentiment, rule-based POS/NER, and LSTM.

---

## ✅ **Project Overview**

Imagine you are hired as an NLP consultant by an e-commerce platform (Amazon/Flipkart).  
Your task is to analyze customer reviews for a specific product (minimum 100 reviews) and generate:

- ✅ Trends  
- ✅ Sentiment analysis  
- ✅ Topic extraction  
- ✅ Semantic similarity  
- ✅ Summary of common opinions  
- ✅ Answers to common user questions  

This project covers the complete pipeline from **web scraping → preprocessing → syntactic analysis → semantic analysis → ML → insights**.

---

## ✅ **Features Implemented**

### 🔹 **1. Data Acquisition**
- Web scraping using BeautifulSoup/Selenium  
- Handles pagination  
- Stores scraped reviews in **data/raw/**  
- Saves cleaned/translated reviews in **data/processed/**  

### 🔹 **2. Language Detection + Translation**
- Detects review language (English/Hindi)  
- Translates Hindi → English using `googletrans`  
- Ensures sentiment consistency  

### 🔹 **3. NLP Preprocessing**
- HTML tag removal  
- Lowercasing  
- Tokenization  
- Stopword removal  
- Lemmatization / Stemming  
- Duplicate removal  

### 🔹 **4. Syntactic Analysis**
- POS tagging (NLTK / spaCy non-transformer model)  
- Rule-based/statistical NER  
- Adjective/verb extraction  

### 🔹 **5. Semantic Analysis**
- TF-IDF vectorization  
- Word2Vec embeddings (gensim)  
- Cosine similarity across reviews  
- Identify key features & similar terms  

### 🔹 **6. Topic Modeling**
- Latent Semantic Analysis (LSA)  
- Extracts 3–5 major topics  
- Top keywords per topic  

### 🔹 **7. Sentiment Analysis**
- Lexicon-based sentiment (VADER/TextBlob)  
- LSTM-based sentiment classifier  
- Overall sentiment distribution  

### 🔹 **8. Review Summarization**
- Cluster reviews using cosine similarity  
- Identify top representative reviews  
- Provide human-readable summary  

### 🔹 **9. Question Answering (Simulated)**
Based on the analysis:
- Generates 3–5 common questions a new customer may ask  
- Provides data-driven answers  

---

## ✅ **Project Folder Structure**

