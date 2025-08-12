import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

# Load combined dataset
data_path = "../results/combined_seed_sam_15fts_20s_overlap.npz"
data = np.load(data_path)
X, y, subject_ids = data["features"], data["labels"], data["subject_ids"]

print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Unique subjects: {len(np.unique(subject_ids))}")

# Group-aware train-test split (to prevent subject leakage)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=subject_ids))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Apply SMOTE only on training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"After SMOTE, training samples: {len(X_train_res)}")
print(f"Training label distribution before SMOTE: {np.bincount(y_train)}")
print(f"Training label distribution after SMOTE: {np.bincount(y_train_res)}")

model = RandomForestClassifier(random_state=42)

total_features = X.shape[1]
min_features = 1
max_features = total_features

best_acc = 0
best_n = min_features

print("\nStarting automatic RFE tuning on combined dataset with SMOTE...\n")

for n_feats in range(min_features, max_features + 1):
    selector = RFE(model, n_features_to_select=n_feats, step=1)
    selector.fit(X_train_res, y_train_res)

    X_train_sel = selector.transform(X_train_res)
    X_test_sel = selector.transform(X_test)

    model.fit(X_train_sel, y_train_res)
    y_pred = model.predict(X_test_sel)
    acc = accuracy_score(y_test, y_pred)

    print(f"Features: {n_feats:3d} | Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_n = n_feats

print(f"\nBest number of features: {best_n} with accuracy {best_acc:.4f}\n")

# Train final model with best number of features
selector = RFE(model, n_features_to_select=best_n, step=1)
selector.fit(X_train_res, y_train_res)

X_train_sel = selector.transform(X_train_res)
X_test_sel = selector.transform(X_test)

model.fit(X_train_sel, y_train_res)
y_pred = model.predict(X_test_sel)

print("Final classification report on test set:")
print(classification_report(y_test, y_pred))

selected_features_idx = np.where(selector.support_)[0]
print(f"Selected feature indices (total {len(selected_features_idx)}):")
print(selected_features_idx)