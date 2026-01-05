import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("📥 Loading CREMA-D dataset...")
df = pd.read_csv("cremad_handcrafted.csv")

print("📊 Dataset shape:", df.shape)
print("📊 Class distribution:\n", df["class"].value_counts())

X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

print("🚀 Training model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(model, "model.joblib")
print("💾 Model saved as model.joblib")
