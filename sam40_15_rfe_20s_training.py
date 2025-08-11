import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

# Load data
data = np.load("../results/sam40_anxiety_15fts_20s_overlap.npz")
X, y = data["features"], data["labels"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Choose model
model = RandomForestClassifier(random_state=42)
# model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

total_features = X.shape[1]
min_features = 1
max_features = total_features

best_acc = 0
best_n = min_features

print("Starting automatic RFE tuning...")

for n_feats in range(min_features, max_features + 1):
    selector = RFE(model, n_features_to_select=n_feats, step=1)
    selector.fit(X_train, y_train)

    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)

    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)
    acc = accuracy_score(y_test, y_pred)

    print(f"Number of features: {n_feats}, Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_n = n_feats

print(f"\nBest number of features: {best_n} with accuracy {best_acc:.4f}")

# Train final model with best number of features
selector = RFE(model, n_features_to_select=best_n, step=1)
selector.fit(X_train, y_train)

X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

model.fit(X_train_sel, y_train)
y_pred = model.predict(X_test_sel)

print("\nFinal classification report on test set:")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

selected_features_idx = np.where(selector.support_)[0]
print(f"Selected feature indices: {selected_features_idx}")