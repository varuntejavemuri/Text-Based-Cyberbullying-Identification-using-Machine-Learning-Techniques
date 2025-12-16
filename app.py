import os
import sys
from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
import re

# Import your ML module
PROJECT_PATH = Path(__file__).parent
if str(PROJECT_PATH) not in sys.path:
    sys.path.append(str(PROJECT_PATH))

import cyberbullying_project as cb

# ------------------------- Streamlit Config -------------------------
st.set_page_config(page_title="Cyberbullying Detection", page_icon="🚨", layout="centered")

# ------------------------- Constants -------------------------
MODEL_FILE = PROJECT_PATH / "best_model.joblib"
VECT_FILE = PROJECT_PATH / "vectorizer.joblib"

# Cyberbullying keywords for automatic detection
CYBERBULLYING_KEYWORDS = [
    'stupid', 'idiot', 'rascal', 'kill', 'die', 'bitch', 'ugly', 'fat', 
    'worthless', 'loser', 'hate', 'moron', 'retard', 'trash', 'garbage',
    'useless', 'dumb', 'fool', 'bastard', 'whore', 'slut', 'freak'
]

# ------------------------- NLTK Setup -------------------------
@st.cache_resource(show_spinner=False)
def ensure_nltk_resources():
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        return True
    except Exception as e:
        st.warning(f"Error downloading NLTK resources: {e}")
        return False

# ------------------------- Load / Train Model -------------------------
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    ensure_nltk_resources()

    if MODEL_FILE.exists() and VECT_FILE.exists():
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECT_FILE)
        return model, vectorizer, "loaded"
    else:
        st.info("No saved model found. Training a lightweight model now...")

        url = 'https://drive.google.com/uc?export=download&id=12fBlhsa5GIdtme1jT3KlPPIgIdjzqhv1'
        df = pd.read_json(url, lines=True, orient='columns')
        df['annotation'] = df['annotation'].apply(lambda x: 1 if x['label'][0] == '1' else 0)
        if 'extras' in df.columns:
            df.drop(['extras'], axis=1, inplace=True)

        df = cb.preprocess_dataframe(df, text_column='content')
        X, vectorizer = cb.vectorize_text(df, max_features=3000)
        y = df['annotation'].values

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_FILE)
        joblib.dump(vectorizer, VECT_FILE)
        return model, vectorizer, "trained"

# ------------------------- Keyword Detection -------------------------
def detect_cyberbullying_keywords(text):
    """Detect cyberbullying using keyword matching"""
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in CYBERBULLYING_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            found_keywords.append(keyword)
    
    return found_keywords

# ------------------------- Enhanced Prediction Function -------------------------
def predict_text(model, vectorizer, text):
    # First check for cyberbullying keywords
    found_keywords = detect_cyberbullying_keywords(text)
    
    if found_keywords:
        return "Cyberbullying", 0.95, f" {', '.join(found_keywords)}"
    
    # If no keywords found, use ML model
    ensure_nltk_resources()
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    clean_text = cb.clean_text(text, stop_words, stemmer)
    transformed = vectorizer.transform([clean_text])

    pred = model.predict(transformed)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(transformed)[0].max()
        except Exception:
            prob = None

    label = "Cyberbullying" if int(pred) == 1 else "Not Cyberbullying"
    return label, prob, clean_text

# ------------------------- UI Layout -------------------------
st.title("🚨 Cyberbullying Detection System")
st.write("Detect whether a sentence contains cyberbullying content or not.")

st.divider()

model, vectorizer, status = load_or_train_model()
# Loading messages removed - model loads silently

user_input = st.text_area("Enter your text below:", height=150, placeholder="Type something...")

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("Please enter some text before predicting.")
    else:
        label, prob, cleaned = predict_text(model, vectorizer, user_input)
        st.session_state["last_prediction"] = {
            "text": user_input,
            "label": label,
            "prob": prob,
            "cleaned": cleaned
        }

if "last_prediction" in st.session_state:
    last = st.session_state["last_prediction"]
    st.markdown("### 🧠 Prediction Result")
    st.markdown(f"**Prediction:** {last['label']}")
    if last["prob"] is not None:
        st.markdown(f"**Confidence:** {last['prob']:.2f}")
    # st.markdown("**Analysis:**")
    # st.code(last["cleaned"], language="text")

    st.divider()
    st.markdown("### 🙋‍♂️ Feedback")
    feedback = st.radio(
        "Was this prediction correct?",
        ["Select", "Yes", "No"],
        index=0,
        key="feedback_choice"
    )

    if st.button("✅ Submit Feedback"):
        if feedback == "Select":
            st.warning("Please choose Yes or No before submitting feedback.")
        elif feedback == "Yes":
            st.success("Thank you for your feedback! Glad it was correct.")
            st.session_state["feedback_status"] = "positive"
        elif feedback == "No":
            st.info("Please select the correct label below to retrain the model.")
            st.session_state["feedback_status"] = "negative"

    if st.session_state.get("feedback_status") == "negative":
        st.markdown("### 🔁 Correct Label Selection")
        correct_label = st.selectbox(
            "Choose the correct label for this text:",
            ["Not Cyberbullying", "Cyberbullying"]
        )
        correct_val = 1 if correct_label == "Cyberbullying" else 0

        if st.button("🔄 Retrain Model with This Example"):
            try:
                # Save feedback to permanent storage
                predicted_val = 1 if last["label"] == "Cyberbullying" else 0
                cb.save_feedback_to_csv(
                    original_text=last["text"],
                    predicted_label=predicted_val,
                    correct_label=correct_val,
                    cleaned_text=last["cleaned"]
                )
                
                # Use enhanced retraining
                model, vectorizer = cb.enhanced_retrain_with_feedback(
                    model=model,
                    vectorizer=vectorizer,
                    feedback_texts=[last["text"]],
                    feedback_labels=[correct_val]
                )
                
                joblib.dump(model, MODEL_FILE)
                joblib.dump(vectorizer, VECT_FILE)
                st.success("✅ Model retrained successfully with ALL feedback data!")
                st.session_state["feedback_status"] = None
            except Exception as e:
                st.error(f"Retraining failed: {e}")

st.divider()
st.markdown("### ⚙️ Advanced Section")
st.write("You can re-run the entire training pipeline defined in `cyberbullying_project.py`.")

if st.button("🚀 Run Full ML Pipeline"):
    try:
        st.info("Running full ML training pipeline... This may take a few minutes.")
        cb.main()
        if MODEL_FILE.exists() and VECT_FILE.exists():
            model = joblib.load(MODEL_FILE)
            vectorizer = joblib.load(VECT_FILE)
        st.success("🎉 Full pipeline executed successfully. Model updated!")
    except Exception as e:
        st.error(f"Error during full pipeline execution: {e}")

st.divider()
st.markdown("""
**💡 Notes:**
- NLTK resources like `punkt`, `stopwords`, and `wordnet` are automatically downloaded.
- Model is auto-trained if no saved model is found.
- **Automatic keyword detection** for common cyberbullying terms.
- **Feedback system** saves permanently to CSV file.
- All historical feedback is used when retraining the model.
""")