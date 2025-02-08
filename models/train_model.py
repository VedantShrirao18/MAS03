import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

CSV_FILE = "../data/your_data.csv"

# Load Data
df = pd.read_csv(CSV_FILE)

# Fill missing values in text columns
df["Summary"] = df["Summary"].fillna("")
df["Description"] = df["Description"].fillna("")

# Remove rows where Classification is NaN
df = df.dropna(subset=["Classification"])

# Combine Summary and Description for training
X = df["Summary"] + " " + df["Description"]
y = df["Classification"]

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save Model & Vectorizer
joblib.dump(model, "jira_classification_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model training complete!")
