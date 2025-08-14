import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, classification_report

# === Load combined dataset ===
data_file = "../results/seediv_sam40_combined.npz"
data = np.load(data_file)
X, y, subject_ids = data["features"], data["labels"], data["subject_ids"]

# === Standardize features globally ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split data by subject to avoid leakage ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_scaled, y, groups=subject_ids))
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# === Automatic feature selection with RFECV (XGBoost) ===
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Stratified 3-fold CV inside RFECV
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
rfecv = RFECV(estimator=xgb_model, step=5, min_features_to_select=5, cv=cv, scoring='accuracy', n_jobs=-1)
rfecv.fit(X_train, y_train)

selected_mask = rfecv.support_
print(f"Optimal number of features selected: {rfecv.n_features_}")
print(f"Selected feature indices: {np.where(selected_mask)[0]}")

# Transform datasets
X_train_sel = rfecv.transform(X_train)
X_test_sel = rfecv.transform(X_test)

# === Train models on selected features ===
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42)
}

for name, model in models.items():
    print(f"\n=== Training {name} on selected features ===")
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import GroupShuffleSplit

# # === Load combined dataset ===
# data_file = "../results/seediv_sam40_combined.npz"
# data = np.load(data_file)
# X, y, subject_ids = data["features"], data["labels"], data["subject_ids"]

# # === Split data by subject to avoid leakage ===
# gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# train_idx, test_idx = next(gss.split(X, y, groups=subject_ids))

# X_train, X_test = X[train_idx], X[test_idx]
# y_train, y_test = y[train_idx], y[test_idx]

# # === Models ===
# models = {
#     "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42),
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
#     "SVM": SVC(kernel='rbf', probability=True, random_state=42)
# }

# for name, model in models.items():
#     print(f"\n=== Training {name} ===")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     acc = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {acc:.4f}")
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred))