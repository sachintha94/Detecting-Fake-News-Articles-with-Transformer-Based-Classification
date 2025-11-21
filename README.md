# Detecting-Fake-News-Articles-with-Transformer-Based-Classification

# Fake News Detection with DistilBERT & Streamlit

A complete end-to-end fake news detection system that combines a fine-tuned DistilBERT text classifier with a lightweight Streamlit web app and an optional K-Means clustering module for topic exploration. :contentReference[oaicite:0]{index=0}  

The app takes a news headline and body text as input and returns calibrated probabilities for **real (0)** vs **fake (1)** along with an adjustable decision threshold. It is designed for journalists, educators, fact-checkers, and general users who want a quick credibility signal in the browser.

---

## Features

- ✅ **Transformer-based classifier** using `distilbert-base-uncased`
- ✅ **Binary classification**: `real (0)` vs `fake (1)`
- ✅ **Calibrated probabilities**: `p_real` and `p_fake` with a tunable threshold
- ✅ **Streamlit UI** – simple, mobile-friendly single-page app
- ✅ **Unsupervised analysis** – K-Means over DistilBERT embeddings to reveal topic clusters
- ✅ **Dockerized** for reproducible deployment
- ✅ **CPU-friendly** – runs on a small AWS EC2 instance

---

## Project Overview

Given a short **headline** and **body text**, the system:

1. Concatenates them into:  
   `Title: <title> [SEP] Text: <text>`
2. Tokenizes with `DistilBertTokenizerFast` (max length 512, padding + truncation).
3. Runs the text through a fine-tuned DistilBERT classifier.
4. Outputs:
   - `p_real`, `p_fake`
   - Predicted label (`REAL` / `FAKE`)
   - Explanation of the decision threshold.

A secondary pipeline encodes the same text into embeddings and applies **K-Means** clustering (K≈5–10) to examine topical structure and how fake vs real news are distributed across clusters. :contentReference[oaicite:1]{index=1}  

---

## Dataset

- **Source:** FakeNewsNet (BuzzFeed & PolitiFact splits) via Kaggle. :contentReference[oaicite:2]{index=2}  
- **Publishers:** BuzzFeed, PolitiFact  
- **Classes:** `fake = 1`, `real = 0`  
- **Raw size:** 432 articles (balanced: 216 fake, 216 real)  
- **Key columns used:**
  - `title` – headline
  - `text` – article body
- **Cleaning steps:**
  - Keep only `title` and `text`
  - Add binary label; concatenate the four CSVs
  - Drop rows with null/empty title or text
  - Remove exact duplicates (same title + text)
  - Filter out ultra-short texts (< 10 tokens)
  - Stratified train/val/test split (by label and source where possible) :contentReference[oaicite:3]{index=3}  

The final cleaned dataset (after filtering) is saved as something like:

```text
data/final_fake_news_dataset_clean.csv
