import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset
path = "/kaggle/input/lung-cancer-dataset"
df = pd.read_csv(f"{path}/dataset.csv")

# 2. Encode categorical columns
df["GENDER"] = df["GENDER"].map({"M": 0, "F": 1})
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"NO": 0, "YES": 1})

# 3. Prepare features and target
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# 4. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate on the test set
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved to model.pkl")
