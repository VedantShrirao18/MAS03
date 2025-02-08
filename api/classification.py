import joblib
import random

# Load Trained Model
model = joblib.load("../models/jira_classification_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

def classify_issue_ml(query):
    query_vectorized = vectorizer.transform([query])
    classification = model.predict(query_vectorized)[0]
    issue_id = f"JIRA-{random.randint(1000, 9999)}"
    return issue_id, classification
