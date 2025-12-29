import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# LOAD HANDCRAFTED FEATURE DATA
# ===============================
CSV_PATH = "all_handcrafted_data_tess.csv"

print("📥 Loading dataset...")
df = pd.read_csv(CSV_PATH)

print("📊 Dataset Shape:", df.shape)
print("📊 Columns:", df.columns.tolist())

# ===============================
# CLEAN DATA
# ===============================
# Drop non-feature columns if present
drop_cols = ["path", "source"]
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# Target column
TARGET_COL = "class"

if TARGET_COL not in df.columns:
    raise ValueError("❌ 'class' column not found in dataset")

# ===============================
# SPLIT FEATURES & LABELS
# ===============================
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# TRAIN RANDOM FOREST
# ===============================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

print("🚀 Training model...")
model.fit(X_train, y_train)

# ===============================
# EVALUATION
# ===============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Test Accuracy:", round(accuracy, 4))
print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(model, "model.joblib")
print("\n💾 Model saved as model.joblib")
