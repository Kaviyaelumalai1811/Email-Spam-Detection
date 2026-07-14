\# Email Spam Detection System

\## Overview

The Email Spam Detection System is a Machine Learning project developed using Python. It classifies email messages as \*\*Spam\*\* or \*\*Not Spam\*\* using Natural Language Processing (NLP) techniques and machine learning algorithms.

\## Objective

The objective of this project is to identify spam emails automatically by analyzing the content of an email.

\## Technologies Used

\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- NLTK

\- Streamlit

\- Matplotlib

\## Machine Learning Workflow

1\. Data Collection

2\. Data Preprocessing

3\. Feature Engineering using TF-IDF

4\. Train-Test Split

5\. Model Training

6\. Model Comparison

7\. Best Model Selection

8\. Model Evaluation

9\. Streamlit Application

\## Machine Learning Models Compared

\- Multinomial Naive Bayes

\- Logistic Regression

\- Linear SVM

\- Random Forest

\## Best Performing Model

\- \*\*Model:\*\* Linear SVM

\- \*\*Accuracy:\*\* 98.14%

\## Features

\- Cleans and preprocesses email text

\- Converts text into TF-IDF features

\- Compares multiple machine learning models

\- Predicts whether an email is Spam or Not Spam

\- User-friendly Streamlit interface

\- Confusion Matrix visualization

\- Model Comparison Graph

\## Project Structure

```

emailspamdetect/

│── app.py

│── train\_model.py

│── email\_text.csv

│── spam\_model.pkl

│── tfidf\_vectorizer.pkl

│── confusion\_matrix.png

│── model\_comparison.png

│── requirements.txt

│── README.md

```

\## How to Run

Install the required libraries:

```bash

pip install -r requirements.txt

```

Run the application:

```bash

python -m streamlit run app.py

```

\## Future Improvements

\- Train the model using a larger email dataset.

\- Improve prediction for unseen real-world emails.

\- Enhance the user interface with additional features.
