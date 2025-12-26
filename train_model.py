import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("all_handcrafted_data_tess.csv")

# 🔍 Drop non-numeric columns
df = df.drop(columns=["path", "source"])

# Separate features and labels
X = df.drop(columns=["class"])   # all numeric features
y = df["class"]                  # emotion label

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Training Accuracy:", accuracy_score(y_test, y_pred))

# Save trained model
joblib.dump(model, "model.joblib")
print("Model saved as model.joblib")
