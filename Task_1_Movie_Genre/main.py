import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm  # Added import for tqdm

# Load Training Data
train_data = pd.read_csv('train_data.csv')  # Replace with your actual file path

# Load Testing Data
test_data = pd.read_csv('test_data.csv')    # Replace with your actual file path

# Combine 'title' and 'plot' into a single text column for both training and testing data
train_data['text'] = train_data['title'] + ' ' + train_data['plot']
test_data['text'] = test_data['title'] + ' ' + test_data['plot']

# Assuming you have a 'genre' column in your training data
y_train = train_data['genre']

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit and transform on training data
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['text'])

# Transform testing data
X_test_tfidf = tfidf_vectorizer.transform(test_data['text'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)

# Choose a classifier (e.g., Multinomial Naive Bayes)
nb_classifier = MultinomialNB()

# Train the model with a progress bar
with tqdm(total=100, position=0, leave=True, desc="Training Model") as progress_bar:
    nb_classifier.partial_fit(X_train, y_train, classes=y_train.unique())
    progress_bar.update(100)

# Make predictions on the validation set
val_predictions = nb_classifier.predict(X_val)

# Evaluate the model on the validation set
val_accuracy = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", val_accuracy)
print("\nValidation Classification Report:\n", classification_report(y_val, val_predictions, zero_division=1))

# Once satisfied, use the model to predict genres for the test data
test_predictions = nb_classifier.predict(X_test_tfidf)

# Save the predictions to a CSV file or use them as needed for your project
test_data['Predicted_genre'] = test_predictions
test_data.to_csv('path_to_output_predictions.csv', index=False)
