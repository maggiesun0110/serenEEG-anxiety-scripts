import numpy as np
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import RFE

# Load data
data = np.load("../results/seediv_anxiety_24_ica_cleaned.npz")
X, y, subject_ids = data["features"], data["labels"], data["subject_ids"]

# Split data by subject to avoid leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=subject_ids))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Models without SVM
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    # "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
}

max_features = X.shape[1]

for name, model in models.items():
    best_acc = 0
    best_num_feats = 0
    best_report = None

    print(f"\n=== {name} with RFE feature selection ===")

    for n_feats in range(1, max_features + 1):
        selector = RFE(model, n_features_to_select=n_feats, step=1)
        selector = selector.fit(X_train, y_train)

        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)

        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)

        acc = accuracy_score(y_test, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_num_feats = n_feats
            best_report = classification_report(y_test, y_pred)

        if n_feats % 5 == 0 or n_feats == max_features:
            print(f"Features: {n_feats} - Accuracy: {acc:.4f}")

    print(f"\nBest accuracy for {name}: {best_acc:.4f} with {best_num_feats} features")
    print("Classification Report for best model:")
    print(best_report)
    print("-" * 60)