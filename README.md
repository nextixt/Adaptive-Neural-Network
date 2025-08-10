# 🧠 Adaptive Emotion Classifier with Real-Time Learning

---

## 📌 Overview
This project is an **adaptive emotion classification system** that:
- Detects the **emotion** expressed in a text (e.g., *happiness, sadness, anger*).
- **Explains its decision** by highlighting the most important words.
- **Learns from user feedback** in real time — if the prediction is wrong, the user corrects it and the model updates itself.
- Runs entirely in **Python** using **scikit-learn** (no TensorFlow/PyTorch required).

---

## 📂 Dataset
**Name:** `tweet_emotions.csv`  
**Source:** [Kaggle – Tweet Emotion Recognition]([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text))  
**Description:**
- ~40,000 English tweets labeled with 13 different emotions.
- Used for training/testing the classifier.
- Columns:
  - `tweet_id` (removed during preprocessing)
  - `content` – raw tweet text
  - `sentiment` – emotion label (e.g., `joy`, `anger`, `sadness`, `fear`)

---

## 🧠 Model Card

**Model Name:** Adaptive Emotion Classifier (RFC-based)  
**Version:** 1.0  
**Framework:** scikit-learn (RandomForestClassifier)  
**Features:**
- Vectorization via **TF-IDF** (1–2 n-grams, 30,000 max features).
- Class balancing with `class_weight="balanced"`.
- Feedback loop with retraining on corrected samples.
- Word importance explanation from feature importances.

**Limitations:**
- Current model retrains fully on feedback (not fully incremental).
- Accuracy depends on English grammar and spelling.
- Doesn’t handle code-switching or non-English text well.

**Planned Improvements:**
- Replace RFC with `SGDClassifier` for true incremental learning.
- Store feedback in a persistent database.
- Add SHAP/LIME visual explanations.
- Deploy
