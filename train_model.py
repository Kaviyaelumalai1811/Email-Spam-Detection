import pandas as pd
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# Download NLTK Resources

nltk.download("stopwords")
nltk.download("wordnet")

# Step 1 : Data Loading
df = pd.read_csv("email_text.csv")

df = df[['text', 'label']]
df.dropna(inplace=True)

# Step 2 : Data Preprocessing
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

            word = lemmatizer.lemmatize(word)

            words.append(word)

    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)


# Step 3 : Feature Engineering
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["clean_text"])

y = df["label"]

# Step 4 : Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 5 : Model Selection

models = {

    "Multinomial Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Linear SVM": LinearSVC(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

}

results = []

best_model = None
best_model_name = ""
best_f1 = 0


# Step 6 : Model Training & Evaluation

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)
    results.append([
        name,
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4)
    ])

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name


# Step 7 : Comparison Table

comparison = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison)
# Generate confusion matrix for the best model

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,5))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks([0,1], ["Not Spam","Spam"])

plt.yticks([0,1], ["Not Spam","Spam"])

plt.xlabel("Predicted")

plt.ylabel("Actual")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j],
                 ha="center",
                 va="center",
                 color="black",
                 fontsize=14)

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()

# MODEL COMPARISON GRAPH

plt.figure(figsize=(8,5))

plt.bar(comparison["Model"], comparison["Accuracy"])

plt.title("Model Accuracy Comparison")

plt.xlabel("Machine Learning Models")

plt.ylabel("Accuracy")

plt.ylim(0.90,1.00)

plt.xticks(rotation=15)

for i, value in enumerate(comparison["Accuracy"]):
    plt.text(i, value+0.002, f"{value:.4f}", ha="center")

plt.tight_layout()

plt.savefig("model_comparison.png")

plt.show()

# Step 8 : Save Best Model

with open("spam_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

with open("tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\n" + "=" * 60)
print("BEST MODEL :", best_model_name)
print("BEST F1 SCORE :", round(best_f1, 4))
print("Model Saved Successfully!")
print("=" * 60)