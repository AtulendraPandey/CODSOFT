#Atulendra Pandey

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Load the SMS Spam Collection dataset with proper encoding.
spam_data = pd.read_csv('spam.csv', encoding='latin-1')

# Perform preprocessing on the input data
spam_data.drop_duplicates(inplace=True)
spam_data['label'] = spam_data['v1'].map({'ham': 'ham', 'spam': 'spam'})
texts = spam_data['v2']
labels = spam_data['label']

# Split the data into training and testing sets
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer to the training texts
texts_train_tfidf = tfidf_vectorizer.fit_transform(texts_train)

# Initialize a Naive Bayes classifier
spam_classifier = MultinomialNB()

# Train the classifier
spam_classifier.fit(texts_train_tfidf, labels_train)

# Transform the test texts using the same vectorizer
texts_test_tfidf = tfidf_vectorizer.transform(texts_test)

# Make predictions
labels_pred = spam_classifier.predict(texts_test_tfidf)

# Calculate accuracy
model_accuracy = accuracy_score(labels_test, labels_pred)

# Display classification report with labels 'ham' and 'spam'
report_summary = classification_report(labels_test, labels_pred, target_names=['Legitimate SMS', 'Spam SMS'])

# Create a progress bar
progress_bar = tqdm(total=100, position=0, leave=True)

# Simulate progress updates
for progress_percent in range(10, 101, 10):
    progress_bar.update(10)
    progress_bar.set_description(f'Current Progress: {progress_percent}%')

# Close the progress bar
progress_bar.close()

# Display the results
print(f'Model Accuracy: {model_accuracy:.2f}')
print('Classification Report:')
print(report_summary)
