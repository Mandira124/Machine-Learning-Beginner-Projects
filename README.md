This README explains each and every project included in this repository.
# Project 1: Enron Spam Classifier

## Overview
This project implements an **Email Spam Classifier** using **Logistic Regression** and **TF‑IDF** text features.  
The goal is to classify emails as **Spam** or **Ham (Not Spam)** using real-world email data.

---

## Why Enron Dataset?
- Contains **real corporate emails**
- Clearly labeled as **spam** or **ham**
- Widely used as a **benchmark dataset** for spam detection
- Publicly available and suitable for academic and learning purposes

Dataset used: **SetFit/enron_spam (Hugging Face)**

---

## Tools & Technologies Used
- Python
- Pandas
- Scikit-learn
- Hugging Face `datasets`
- TF-IDF Vectorizer
- Logistic Regression
- Joblib (model persistence)

---

## Installation

```bash
pip install datasets pandas scikit-learn joblib
```
## Steps

### 1. Load Dataset
- Load the Enron spam dataset from Hugging Face  
- Convert it into a Pandas DataFrame  

### 2. Data Preparation
- Use the `text` column as input features  
- Use the `label` column as target  
  - `0` → Ham  
  - `1` → Spam  

### 3. Train-Test Split
- Split the dataset into:
  - 80% Training data  
  - 20% Testing data  

### 4. Text Vectorization
- Convert email text into numerical form using **TF-IDF**  
- Limit vocabulary size to 5000 words  
- Remove English stop words  

### 5. Model Training
- Train a **Logistic Regression** classifier  
- Optimize using **Log Loss**  
- Learn word importance weights for spam detection  

### 6. Evaluation
- Evaluate model performance using:
  - Accuracy  
  - Confusion Matrix  
  - Precision, Recall, F1-score  

### 7. Model Saving
- Save trained Logistic Regression model  
- Save TF-IDF vectorizer using **Joblib**  

### 8. Prediction
- Load saved model and vectorizer  
- Predict whether a new email is spam or not  
- Output both label and spam probability  

---

## Example Predictions
- **"Congratulations! You won a free prize!"** → Spam  
- **"Meeting scheduled for tomorrow at 10 AM"** → Not Spam  

---

## Conclusion
This project demonstrates how **Logistic Regression** combined with **TF‑IDF** can effectively classify emails.  
Despite its simplicity, the model performs well on real-world data and serves as a strong baseline for text classification tasks.

---------------------------------------------------------------------------------------------------------------------------------

# Project 2: Multi-class Sentiment Analysis on IMDb Dataset

## Overview
This project extends basic binary text classification (like spam detection) to **multi-class sentiment analysis**.  
Given a movie review, the model predicts if it is:
- Positive
- Neutral
- Negative

We use **TF-IDF vectorization** and **Logistic Regression** to build the model.

---

## Steps

### 1. Load Dataset
- Load multiclass-sentiment-analysis-dataset from Huggingface
- Convert to pandas DataFrame

### 2. Data Preparation
- Use the review text as input
- Map sentiment labels to numbers
  - negative → 0
  - neutral → 1
  - positive → 2

### 3. Train-Test Split
- 80% training data
- 20% testing data

### 4. Text Vectorization
- TF-IDF converts text into numerical features
- Remove stopwords
- Limit vocabulary to 20000 words

### 5. Model Training
- Train multi-class Logistic Regression
- Use one-vs-rest approach
- Learn word importance for sentiment

### 6. Evaluation
- Accuracy, confusion matrix
- Precision, Recall, F1-score per class

### 7. Prediction
- Load model & vectorizer
- Input new review → predict sentiment and probability

