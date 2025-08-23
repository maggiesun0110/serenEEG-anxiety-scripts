import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif

# Load features, labels, and subject IDs
data = np.load("../results/seediv_anxiety_24_ica_cleaned.npz")
X, y, subject_ids = data["features"], data["labels"], data["subject_ids"]

# Split data by subject to prevent leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=subject_ids))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Models to evaluate
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM (RBF Kernel)": SVC(probability=True, random_state=42),
}

max_k = min(30, X.shape[1])
ks = list(range(1, max_k+1, 1))

best_results = {name: {"accuracy": 0, "k": None} for name in models.keys()}

for k in ks:
    print(f"\n=== SelectKBest with k={k} features ===")
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    for name, model in models.items():
        print(f"\n-- Model: {name} --")
        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        if acc > best_results[name]["accuracy"]:
            best_results[name]["accuracy"] = acc
            best_results[name]["k"] = k

print("\n=== Best results summary ===")
for name, result in best_results.items():
    print(f"{name}: Best Accuracy = {result['accuracy']:.4f} with k = {result['k']} features")