import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load features and labels for 15 features version
data = np.load("../results/sam40_anxiety_15fts_20s_overlap.npz")
X, y = data["features"], data["labels"]

# Split data randomly (80% train, 20% test), stratify to keep class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
}

for name, model in models.items():
    print(f"=== Training and testing {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 40)

    # Feature importance
    if name == "Random Forest":
        importances = model.feature_importances_
    elif name == "XGBoost":
        importances = model.get_booster().get_score(importance_type='weight')
        # Convert dict to list aligned with features
        # XGBoost feature names like 'f0', 'f1', ...
        importances = [importances.get(f"f{i}", 0) for i in range(X.shape[1])]

    # Plot feature importance
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances)
    plt.title(f"{name} Feature Importances")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.show()