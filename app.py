import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧"
)

# -----------------------------
# Load Model
# -----------------------------
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# -----------------------------
# Text Preprocessing
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'\S+@\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'\d+', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\s+', ' ', text).strip()

    words = []

    for word in text.split():

        if word not in stop_words:

            words.append(lemmatizer.lemmatize(word))

    return " ".join(words)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Project Details")

st.sidebar.write("**Project:** Email Spam Detection")

st.sidebar.write("**Machine Learning Model:** Linear SVM")

st.sidebar.write("**Feature Engineering:** TF-IDF")

st.sidebar.write("**Accuracy:** 98.14%")

st.sidebar.write("**Language:** Python")

st.sidebar.write("**Framework:** Streamlit")

# -----------------------------
# Main Page
# -----------------------------
st.title("📧 Email Spam Detection System")

st.write(
    "Enter the email content below to check whether it is **Spam** or **Not Spam**."
)

email = st.text_area(
    "Email Content",
    height=220
)

if st.button(" Detect"):

    if email.strip() == "":
        st.warning("Please enter an email.")

    else:

        cleaned_email = clean_text(email)

        email_vector = vectorizer.transform([cleaned_email])

        prediction = model.predict(email_vector)[0]

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("🚫 Spam Email")

        else:

            st.success("✅ Not Spam")

        st.info("Best Model: Linear SVM")

        st.success("Model Accuracy: 98.14%")

st.markdown("---")

st.caption("Developed using Python, Scikit-learn, NLTK and Streamlit")