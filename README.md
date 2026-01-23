# Enron Spam Classifier

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
