import polars as pl
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DEBUG=True
random_state=42
# schema=["path", "train_test", "pos_neg", "rating", "content"]
path_to_data = Path("data/imdb.parquet")
data_df = pl.read_parquet(path_to_data)
train = data_df.filter(pl.col('train_test')=='train')
if DEBUG==True: train = train.sample(2000)
text_column = 'content'
label_column = 'pos_neg'
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    train[text_column], train[label_column], test_size=test_size, random_state=random_state
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=random_state))
])

# Train the pipeline
pipeline.fit(X_train, y_train)
# Make predictions
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=pipeline.classes_)


print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', report)


def print_feature_importance_from_pipeline(pipeline, top_n=10):
    # Extract the TfidfVectorizer and Random Forest classifier from the pipeline
    vectorizer = pipeline.named_steps['tfidf']
    rf_classifier = pipeline.named_steps['classifier']
    # Get feature names from the TfidfVectorizer
    feature_names = vectorizer.get_feature_names_out()
    # Get feature importance scores from the Random Forest classifier
    feature_importance = rf_classifier.feature_importances_
    # Create a DataFrame with feature names and their importance scores
    feature_importance_df = pl.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    # Sort the DataFrame by importance in descending order
    feature_importance_df = feature_importance_df.sort(by='Importance', descending=True)
    # Print the top N features
    print(f'Top {top_n} Features:')
    print(feature_importance_df.head(top_n))

print_feature_importance_from_pipeline(pipeline=pipeline, top_n=8)