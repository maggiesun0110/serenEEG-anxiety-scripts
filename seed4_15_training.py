import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit

# Load SEED IV 15-feature dataset
data_path = "../results/seediv_anxiety_15fts_20s_overlap.npz"
data = np.load(data_path)
X, y, subject_ids = data["features"], data["labels"], data["subject_ids"]

print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Unique subjects: {len(np.unique(subject_ids))}")
print("Overall label distribution:", Counter(y))

# Group-aware split to avoid subject leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=subject_ids))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
subj_train, subj_test = subject_ids[train_idx], subject_ids[test_idx]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
print("Train label distribution:", Counter(y_train))
print("Test label distribution:", Counter(y_test))

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train and evaluate
for name, model in models.items():
    print(f"\n=== Training and testing {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))