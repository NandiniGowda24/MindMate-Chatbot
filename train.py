# train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import pickle
import os

# 1️⃣ Load dataset safely
data_path = os.path.join(os.path.dirname(__file__), "nlp/data.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

data = pd.read_csv(data_path)

# Normalize column names (remove spaces, lowercase)
data.columns = [col.strip().lower() for col in data.columns]

# Ensure required columns exist
if "statement" not in data.columns or "status" not in data.columns:
    raise ValueError("❌ Dataset must contain 'statement' and 'status' columns!")

# Drop missing values
data = data.dropna(subset=["statement", "status"])

# Ensure all statements are strings
data["statement"] = data["statement"].astype(str)

# Remove empty statements
data = data[data["statement"].str.strip() != ""]

texts = data["statement"]
labels = data["status"]

# 2️⃣ Train/test split (with stratify to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3️⃣ TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4️⃣ Train Logistic Regression classifier
clf = LogisticRegression(
    max_iter=500, solver="liblinear", class_weight="balanced"
)
clf.fit(X_train_vec, y_train)

# 5️⃣ Evaluate
y_pred = clf.predict(X_test_vec)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# 6️⃣ Save trained model and vectorizer
os.makedirs("models", exist_ok=True)

joblib.dump(clf, "models/intent_clf.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")

# 🔑 Also save a pickle file (so Flask app.py can load it directly)
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("\n✅ Model training completed and saved as:")
print("- models/intent_clf.joblib")
print("- models/vectorizer.joblib")
print("- model.pkl (for Flask)")
