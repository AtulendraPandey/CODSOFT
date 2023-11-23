# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Load the training and testing data
try:
    train_data = pd.read_csv('fraudTrain.csv')
    test_data = pd.read_csv('fraudTest.csv')
except FileNotFoundError:
    print("File not found. Please check the file paths.")

# Define features and target variable for training data
X_train = train_data.drop('is_fraud', axis=1)
y_train = train_data['is_fraud']

# Define features and target variable for test data
X_test = test_data.drop('is_fraud', axis=1)
y_test = test_data['is_fraud']

# Drop unnecessary columns for both training and test data
drop_columns = ['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'first', 'last', 'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num', 'unix_time']
X_train = X_train.drop(drop_columns, axis=1)
X_test = X_test.drop(drop_columns, axis=1)

# Print the remaining column names
print("Remaining columns:", X_train.columns)

# Define preprocessing for numerical and categorical features
numeric_features = ['lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['gender']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier()
}

# Train and evaluate each model on the training data
for model_name, model in models.items():
    # Create a pipeline with preprocessing and the specified model
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', model)])

    # Fit the model on the training data
    model_pipeline.fit(X_train, y_train)
    
    # Predict on the test data
    test_predictions = model_pipeline.predict(X_test)

    # Evaluate the model on the test data
    accuracy = accuracy_score(y_test, test_predictions)
    print(f'{model_name} - Test Accuracy: {accuracy:.4f}')
    
    # Evaluate the model on the test data
    print(f'{model_name} - Classification Report:')
    print(classification_report(y_test, test_predictions, zero_division=1))

    
    # Perform cross-validation
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f'{model_name} - Cross-Validation Accuracy: {cv_scores.mean():.4f} (std: {cv_scores.std():.4f})')
