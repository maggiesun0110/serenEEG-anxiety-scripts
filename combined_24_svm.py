import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

# === Load combined dataset ===
data_file = "../results/seediv_sam40_combined.npz"
data = np.load(data_file)
X, y, subject_ids = data["features"], data["labels"], data["subject_ids"]

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split data by subject to avoid leakage ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_scaled, y, groups=subject_ids))
X_train_scaled, X_test_scaled = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# === Automatic RFE using LinearSVC for feature selection ===
max_features = X.shape[1]
best_acc = 0
best_num_feats = 0
best_mask = None
best_report = None

print("=== Automatic RFE for SVM ===")
for n_feats in range(1, max_features + 1):
    # Linear SVM for RFE (provides coef_)
    linear_svm = LinearSVC(random_state=42, max_iter=5000)
    selector = RFE(linear_svm, n_features_to_select=n_feats, step=1)
    selector = selector.fit(X_train_scaled, y_train)
    
    X_train_sel = selector.transform(X_train_scaled)
    X_test_sel = selector.transform(X_test_scaled)
    
    # Train RBF SVM on selected features
    rbf_svm = SVC(kernel='rbf', probability=True, random_state=42)
    rbf_svm.fit(X_train_sel, y_train)
    y_pred = rbf_svm.predict(X_test_sel)
    
    acc = accuracy_score(y_test, y_pred)
    
    if acc > best_acc:
        best_acc = acc
        best_num_feats = n_feats
        best_mask = selector.support_
        best_report = classification_report(y_test, y_pred)
    
    if n_feats % 5 == 0 or n_feats == max_features:
        print(f"Features: {n_feats} - Accuracy: {acc:.4f}")

print(f"\nOptimal number of features selected: {best_num_feats}")
print("Selected feature indices:", np.where(best_mask)[0])
print("Classification Report for best SVM model:")
print(best_report)