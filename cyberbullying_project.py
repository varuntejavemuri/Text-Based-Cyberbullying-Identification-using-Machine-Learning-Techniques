# ==================== IMPORTS & SETUP ====================
import numpy as np
import pandas as pd
import re
import string
import joblib
import warnings
import csv
import os
import uuid
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import fasttext
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
###########################################################

# ==================== FEEDBACK SYSTEM ====================
FEEDBACK_FILE = "feedback_data.csv"

def save_feedback_to_csv(original_text, predicted_label, correct_label, cleaned_text):
    """Save feedback data to CSV file"""
    file_exists = os.path.isfile(FEEDBACK_FILE)
    
    with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'original_text', 'predicted_label', 'correct_label', 'cleaned_text'])
        
        writer.writerow([
            datetime.now().isoformat(),
            original_text,
            predicted_label,
            correct_label,
            cleaned_text
        ])

def load_feedback_from_csv():
    """Load feedback data from CSV"""
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame()

def get_feedback_stats():
    """Get statistics about feedback data"""
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        total_feedback = len(df)
        return total_feedback
    return 0

def enhanced_retrain_with_feedback(model, vectorizer, feedback_texts, feedback_labels):
    """Enhanced retraining that properly uses CSV feedback data"""
    try:
        # Load original training data
        url = 'https://drive.google.com/uc?export=download&id=12fBlhsa5GIdtme1jT3KlPPIgIdjzqhv1'
        df = pd.read_json(url, lines=True, orient='columns')
        df['annotation'] = df['annotation'].apply(lambda x: 1 if x['label'][0] == '1' else 0)
        if 'extras' in df.columns:
            df.drop(['extras'], axis=1, inplace=True)
        df = preprocess_dataframe(df, text_column='content')
        
        # Add current feedback samples
        if feedback_texts:
            feedback_df = pd.DataFrame({'content': feedback_texts, 'annotation': feedback_labels})
            feedback_df = preprocess_dataframe(feedback_df, text_column='content')
            df = pd.concat([df, feedback_df], ignore_index=True)
        
        # Load ALL historical feedback from CSV
        historical_feedback = load_feedback_from_csv()
        if not historical_feedback.empty:
            # Use only the corrected samples (where prediction was wrong)
            corrected_feedback = historical_feedback[
                historical_feedback['predicted_label'] != historical_feedback['correct_label']
            ]
            if not corrected_feedback.empty:
                historical_df = pd.DataFrame({
                    'content': corrected_feedback['original_text'],
                    'annotation': corrected_feedback['correct_label']
                })
                historical_df = preprocess_dataframe(historical_df, text_column='content')
                df = pd.concat([df, historical_df], ignore_index=True)
        
        # CRITICAL: Re-fit the vectorizer on the expanded dataset
        vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
        X = vectorizer.fit_transform(df['clean_text'])
        y = df['annotation'].values
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        
        # Retrain the model
        print("🔄 Retraining model with enhanced dataset...")
        model.fit(X_res, y_res)
        
        print(f"✅ Model retrained with {len(df)} total samples "
              f"({len(historical_feedback)} feedback samples included)")
        return model, vectorizer
        
    except Exception as e:
        print(f"❌ Enhanced retraining failed: {e}")
        return model, vectorizer

# ==================== FASTTEXT PIPELINE ====================
def save_fasttext_format(df, text_col, label_col, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            label = f"__label__{row[label_col]}"
            text = row[text_col].replace('\n', ' ')
            f.write(f"{label} {text}\n")

def train_fasttext_model(train_file):
    model = fasttext.train_supervised(train_file, epoch=10, lr=1.0, wordNgrams=2)
    return model

def evaluate_fasttext_model(model, test_file):
    result = model.test(test_file)
    print(f"\n# ===== FastText Evaluation =====")
    print(f"Test samples: {result[0]}")
    print(f"Precision: {result[1]:.4f} | Recall: {result[2]:.4f}")
    return result[1]  # precision

def predict_fasttext(model, text):
    result = model.predict(text)
    label = result[0][0].replace('__label__', '') if result[0] else 'Unknown'
    prob = result[1][0] if result[1] else 0.0
    print(f"Prediction: {label} (probability: {prob:.2f})")
###########################################################

warnings.filterwarnings("ignore")

# ==================== DATA LOADING & PREPROCESSING ====================
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def clean_text(text, stop_words, stemmer):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and not word.isdigit()]
    return " ".join(tokens)

def preprocess_dataframe(df, text_column='content'):
    download_nltk_resources()
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    df['clean_text'] = df[text_column].astype(str).apply(lambda x: clean_text(x, stop_words, stemmer))
    return df

# ==================== FEATURE ENGINEERING ====================
def vectorize_text(df, max_features=3000):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,1))
    X = vectorizer.fit_transform(df['clean_text'])
    return X, vectorizer

# ==================== MODEL TRAINING ====================
def train_logistic_regression(X_train, y_train):
    params = {'C': [0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(max_iter=200), params, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_score_

def train_random_forest(X_train, y_train):
    params = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    grid = GridSearchCV(RandomForestClassifier(n_jobs=-1, random_state=42), params, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_score_

def train_svm(X_train, y_train):
    params = {'C': [0.5, 1, 10], 'kernel': ['linear', 'rbf']}
    grid = GridSearchCV(SVC(probability=True), params, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_score_

# ==================== MODEL EVALUATION ====================
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n# ===== {model_name} Evaluation =====")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return f1

# ==================== BEST MODEL SELECTION ====================
def select_best_model(results, models):
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    print(f"\n# Best model selected: {best_model_name} (F1-score: {results[best_model_name]:.4f})")
    return best_model, best_model_name

# ==================== REAL-TIME PREDICTION & FEEDBACK LOOP ====================
def predict_and_feedback(model, vectorizer):
    print("\nType a sentence to check for cyberbullying (or type 'exit' to quit):")
    # Store feedback samples
    feedback_texts = []
    feedback_labels = []
    while True:
        user_input = input("Your sentence: ")
        if user_input.lower() == 'exit':
            break
        try:
            download_nltk_resources()
            stop_words = set(stopwords.words('english'))
            stemmer = PorterStemmer()
            clean = clean_text(user_input, stop_words, stemmer)
            X_user = vectorizer.transform([clean])
            pred = model.predict(X_user)[0]
            label = "Cyberbullying" if pred == 1 else "Not Cyberbullying"
            print(f"Prediction: {label}")
            feedback = input("Was this prediction correct? (yes/no): ").strip().lower()
            if feedback == "yes":
                print("Thank you for your feedback!")
                continue
            elif feedback == "no":
                correct_label = int(input("Please enter the correct label (1 for cyberbullying, 0 for not): ").strip())
                feedback_texts.append(user_input)
                feedback_labels.append(correct_label)
                # Save to CSV
                save_feedback_to_csv(user_input, pred, correct_label, clean)
                model, vectorizer = enhanced_retrain_with_feedback(model, vectorizer, feedback_texts, feedback_labels)
                joblib.dump(model, "best_model.joblib")
                joblib.dump(vectorizer, "vectorizer.joblib")
            else:
                print("Feedback not recognized. Skipping online training.")
                continue
        except Exception as e:
            print("Error during prediction or feedback:", e)

# Keep original retrain function for compatibility
def retrain_with_feedback(model, vectorizer, feedback_texts, feedback_labels):
    return enhanced_retrain_with_feedback(model, vectorizer, feedback_texts, feedback_labels)

# ==================== MAIN PIPELINE ====================
def main():
    # Load data
    url = 'https://drive.google.com/uc?export=download&id=12fBlhsa5GIdtme1jT3KlPPIgIdjzqhv1'
    print("# Loading data...")
    df = pd.read_json(url, lines=True, orient='columns')
    df['annotation'] = df['annotation'].apply(lambda x: 1 if x['label'][0] == '1' else 0)
    if 'extras' in df.columns:
        df.drop(['extras'], axis=1, inplace=True)
    df = preprocess_dataframe(df, text_column='content')
    print("# Data loaded and preprocessed.")
    
    # Show sample counts
    num_cyberbullying = (df['annotation'] == 1).sum()
    num_not_cyberbullying = (df['annotation'] == 0).sum()
    print(f"Number of cyberbullying samples: {num_cyberbullying}")
    print(f"Number of not cyberbullying samples: {num_not_cyberbullying}")

    # Feature engineering
    print("# Vectorizing text...")
    X, vectorizer = vectorize_text(df, max_features=3000)
    y = df['annotation'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train models
    models = {}
    results = {}

    lr_model, _ = train_logistic_regression(X_train_res, y_train_res)
    models['Logistic Regression'] = lr_model
    results['Logistic Regression'] = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')

    rf_model, _ = train_random_forest(X_train_res, y_train_res)
    models['Random Forest'] = rf_model
    results['Random Forest'] = evaluate_model(rf_model, X_test, y_test, 'Random Forest')

    svm_model, _ = train_svm(X_train_res, y_train_res)
    models['SVM'] = svm_model
    results['SVM'] = evaluate_model(svm_model, X_test, y_test, 'SVM')

    # Select best model
    best_model, best_model_name = select_best_model(results, models)

    # Save model and vectorizer
    joblib.dump(best_model, "best_model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")
    print("# Model and vectorizer saved.")

    # Interactive prediction and feedback loop
    predict_and_feedback(best_model, vectorizer)

if __name__ == "__main__":
    main()