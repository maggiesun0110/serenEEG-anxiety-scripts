import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import reciprocal, uniform

# === Config ===
SELECTED_FEATURES = [0, 3, 4, 5, 6, 7, 12, 14, 15, 16, 18, 22, 23, 27, 28, 29,
                     30, 31, 33, 37, 39, 42, 45, 47, 51, 52, 53, 54, 56, 57,
                     63, 66, 69, 70]
RANDOM_STATE = 42

# === Load dataset ===
data_file = "../results/seediv_sam40_combined.npz"
data = np.load(data_file)
X, y, subject_ids = data["features"], data["labels"], data["subject_ids"]

# === Select only chosen features ===
X_selected = X[:, SELECTED_FEATURES]

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# === Subject-based split to avoid leakage ===
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X_scaled, y, groups=subject_ids))
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# === SVM Model ===
svm = SVC(probability=True, random_state=RANDOM_STATE)

# === Randomized Search space ===
random_params = {
    'C': reciprocal(1e-3, 1e3),
    'gamma': reciprocal(1e-4, 1e1),
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4, 5],  # only used if kernel='poly'
    'coef0': uniform(-1, 2)  # for poly/sigmoid
}

random_search = RandomizedSearchCV(
    svm,
    param_distributions=random_params,
    n_iter=50,
    scoring='accuracy',
    cv=5,
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

print("\n===== RandomizedSearchCV Results =====")
print("Best params:", random_search.best_params_)
print("Best CV accuracy:", random_search.best_score_)

# === Refined Grid Search based on best params ===
best_params = random_search.best_params_
grid_params = {
    'C': np.linspace(best_params['C'] * 0.5, best_params['C'] * 1.5, 5),
    'gamma': np.linspace(best_params['gamma'] * 0.5, best_params['gamma'] * 1.5, 5),
    'kernel': [best_params['kernel']],
}

if best_params['kernel'] == 'poly':
    grid_params['degree'] = [max(2, best_params['degree'] - 1),
                              best_params['degree'],
                              best_params['degree'] + 1]
    grid_params['coef0'] = np.linspace(best_params['coef0'] - 0.5, best_params['coef0'] + 0.5, 5)
elif best_params['kernel'] == 'sigmoid':
    grid_params['coef0'] = np.linspace(best_params['coef0'] - 0.5, best_params['coef0'] + 0.5, 5)

grid_search = GridSearchCV(
    svm,
    param_grid=grid_params,
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("\n===== GridSearchCV Results =====")
print("Best params:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)

# === Final evaluation ===
final_model = grid_search.best_estimator_
y_pred = final_model.predict(X_test)

print("\n===== Final Test Results =====")
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))