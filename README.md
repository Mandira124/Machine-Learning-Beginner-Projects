Enron Spam Classifier


A machine learning project to classify emails as spam or not spam (ham) using the Enron Spam dataset from Hugging Face. The model uses TF-IDF vectorization and Logistic Regression for prediction.

Dataset

Source: Hugging Face - SetFit/enron_spam

Contains email text labeled as spam (1) or ham (0).

Dataset loaded directly using the datasets library.

Requirements

Python 3.x

Libraries:

datasets

pandas

scikit-learn

joblib

Install dependencies:

pip install datasets pandas scikit-learn joblib

Project Structure
.
├── spam_model.pkl            # Trained Logistic Regression model
├── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer
├── spam_classifier.py        # Python script with training & prediction


Steps

Load Dataset

Load SetFit/enron_spam using the datasets library.

Convert to Pandas DataFrame.

Preprocess Data

Combine Subject and Message (if needed).

Use text column as input and label column as target.

Train/Test Split

Split data into 80% training and 20% testing using train_test_split.

Vectorize Text

Use TfidfVectorizer with stop_words='english' and max_features=5000.

Fit on training data, transform both train and test data.

Train Model

Logistic Regression (max_iter=1000) is used.

Fit model on TF-IDF-transformed training data.

Evaluate Model

Metrics used: accuracy, confusion matrix, classification report.

Save Model & Vectorizer

joblib.dump() to save spam_model.pkl and tfidf_vectorizer.pkl.

Predict Function

Function predict_spam(message):

Input: raw email text

Output: "SPAM" or "NOT SPAM" and probability score

Usage
from spam_classifier import predict_spam

result = predict_spam("Congratulations! You've won a free iPhone!")
print(result)

Notes

The model is trained on the Enron dataset; performance depends on email content.

Preprocessing like lowercasing and removing punctuation may improve prediction accuracy.

Threshold for spam classification is 0.5 (can be adjusted).
