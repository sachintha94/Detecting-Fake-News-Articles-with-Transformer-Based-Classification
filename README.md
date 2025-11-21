# Detecting-Fake-News-Articles-with-Transformer-Based-Classification

# Fake News Detection with DistilBERT & Streamlit

Fake News Detector-Hugging Face LLM Fine-Tuned (DistilBERT) + Streamlit (AWS EC2, Docker)
Built a lightweight web app that classifies news as real (0) or fake (1) using a Hugging Face fine-tuned LLM (DistilBERT). Trained with the Transformers Trainer on FakeNewsNet (BuzzFeed + PolitiFact), then served via Streamlit with calibrated probabilities and a threshold slider. Added topic clustering (K-Means on embeddings) for interpretability. Deployed as a Docker container on AWS EC2.
 • LLM: DistilBERT, fine-tuned with Hugging Face Transformers
 • Data: 432 labeled articles (FakeNewsNet)
 • Results: ~0.76 accuracy, 0.77 weighted F1 (hold-out)
 • Features: real-time predictions, FAKE/REAL badge, interpretable probabilities, topic clusters
 • Stack: Python, PyTorch/Transformers (HF), scikit-learn, Streamlit, Docker, AWS EC2

Web app link : http://3.27.201.8:8501/
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
<img width="845" height="498" alt="image" src="https://github.com/user-attachments/assets/e4bd4f25-64ce-4062-b254-7dc105e6f776" />

![04](https://github.com/user-attachments/assets/810ff94d-478c-4403-89b1-71cf0a04e486)


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

<img width="639" height="452" alt="image" src="https://github.com/user-attachments/assets/cdbe8c20-923e-428b-bbc5-5d26b68195da" />

<img width="579" height="424" alt="image" src="https://github.com/user-attachments/assets/8a563341-6951-4991-bb68-719ec4cca629" />
