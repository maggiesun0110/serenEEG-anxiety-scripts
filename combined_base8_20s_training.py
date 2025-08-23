import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# Load combined dataset
data = np.load("../results/combined_sam40_seed4_base8.npz")
X, y, ids = data["features"], data["labels"], data["subject_ids"]

print(f"Total samples: {X.shape[0]}, Feature dims: {X.shape[1]}")
print(f"Total unique subjects: {len(np.unique(ids))}")
print(f"Label distribution overall: {Counter(y)}")

# Split data with group shuffle split to prevent subject leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=ids))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print(f"Train label distribution: {Counter(y_train)}")
print(f"Test label distribution: {Counter(y_test)}")

# Compute class weights for balancing
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: w for cls, w in zip(np.unique(y_train), class_weights)}
print(f"Class weights: {class_weight_dict}")

# Initialize models with class weights
rf = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42,
                    scale_pos_weight=class_weight_dict.get(1, 1))  # For binary class 1 weight

models = {"Random Forest": rf, "XGBoost": xgb}

for name, model in models.items():
    print(f"\n=== Training and testing {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))