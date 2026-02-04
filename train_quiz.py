import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# 1️⃣ Load dataset
data = pd.read_csv("dataset.csv")

# Make sure column names are clean
data.columns = [col.strip().lower() for col in data.columns]

# Features (all symptoms except the label column 'disease')
X = data.drop(columns=["disease"])
y = data["disease"]

# 2️⃣ Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Train model
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

# 4️⃣ Evaluate
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# 5️⃣ Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/symptom_model.joblib")

print("✅ Symptom model saved as models/symptom_model.joblib")
