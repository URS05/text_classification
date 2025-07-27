"""
# 🧠 Text Classification with Logistic Regression

This project is part of a capstone assignment for the Artificial Intelligence track at Coincent.  
It performs binary sentiment classification (positive/negative) on product titles from a large Toys & Games dataset.

## 📂 Dataset

- **Source**: Amazon product data (provided in `parquet` format)
- **Size**: ~890,000 product records
- **Fields Used**: `title`, `average_rating`

## 🎯 Objective

Build a classification model that predicts sentiment using review-based ratings:
- Ratings ≥ 4.0 → Positive sentiment
- Ratings ≤ 2.0 → Negative sentiment

Neutral-rated data is excluded for binary classification.

## ⚙️ Tech Stack

- Python 3.10+
- pandas, scikit-learn
- Regex for text cleaning
- Logistic Regression model

## 🚀 Project Workflow

1. Load and concatenate `.parquet` files
2. Label data as `positive` or `negative` based on average rating
3. Preprocess product titles (lowercase, punctuation/digit removal)
4. Vectorize using `CountVectorizer` (with English stopword removal)
5. Train/test split (80/20)
6. Fit `LogisticRegression` with class balancing
7. Evaluate using:
   - Accuracy
   - Confusion Matrix
   - Classification Report

## 📈 Results

- **Model Accuracy**: ~84.4%
- Strong precision and recall for positive sentiment
- Confusion matrix and full metrics included in `load_toys_data.py` output

## 📸 Screenshots

![Model Output](screenshots/model_output.png)

## 📄 How to Run

```bash
python src/load_toys_data.py
"""