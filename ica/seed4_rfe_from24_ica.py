import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import RFE

# Load data
data = np.load("../results/seediv_anxiety_24_ica_cleaned.npz")
X, y, subject_ids = data["features"], data["labels"], data["subject_ids"]

# Optional: load names if present
if "feature_names" in data:
    feature_names = data["feature_names"]
else:
    feature_names = np.array([f"feat_{i}" for i in range(X.shape[1])])

# Split data by subject to avoid leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=subject_ids))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# XGBoost model for RFE
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# RFE to select top 10 features
selector = RFE(xgb_model, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)

# Boolean mask
feature_mask = selector.support_

# Selected feature indices and names
selected_indices = np.where(feature_mask)[0]
selected_names = feature_names[feature_mask]

print("Selected feature indices:", selected_indices.tolist())
print("Selected feature names:", selected_names.tolist())
print("Boolean mask:", feature_mask.tolist())

# Evaluate using only top 10 features
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)
xgb_model.fit(X_train_sel, y_train)
acc = xgb_model.score(X_test_sel, y_test)
print(f"Accuracy with top 10 features: {acc*100:.2f}%")

# Save mask for later reuse
np.save("top10_feature_mask_seediv_xgb.npy", feature_mask)