import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
    f1_score,
)
import joblib
from pathlib import Path

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

OUTPUT = Path('outputs/')

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

## ROC Question 1


log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

y_pred_proba_lr = log_reg.predict_proba(X_test)[:, 1]

auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
print(f"Logistic Regression AUC: {auc_lr:.4f}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred_proba_knn = knn.predict_proba(X_test_scaled)[:, 1]

auc_knn = roc_auc_score(y_test, y_pred_proba_knn)
print(f"K-Nearest Neighbors AUC: {auc_knn:.4f}")


# Logistic Regression has a higher AUC.
# A higher AUC indicates that the model has better capacity to classify between the classes.

## ROC Question 2

fig, ax = plt.subplots(figsize=(8, 6))

RocCurveDisplay.from_estimator(
    log_reg, 
    X_test, 
    y_test, 
    name=f"Logistic Regression (AUC = {auc_lr:.2f})", 
    ax=ax
)

RocCurveDisplay.from_estimator(
    knn, 
    X_test_scaled, 
    y_test, 
    name=f"K-Nearest Neighbors (AUC = {auc_knn:.2f})", 
    ax=ax
)


ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
ax.set_title("ROC Curve Comparison")
ax.legend(loc="lower right")

plt.savefig(OUTPUT / "roc_comparison.png", bbox_inches="tight")
plt.close()

# Logistic Regression has the lower FPR.
# Logistic Regression will produce less false alarms. This means it is more precise and makes fewer mistakes.

## ROC Question 3

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_lr)

best_f1 = -1
best_threshold = 0.5
best_tpr = 0
best_fpr = 0

for i, threshold in enumerate(thresholds):
    y_pred = (y_pred_proba_lr >= threshold).astype(int)
    score = f1_score(y_test, y_pred)
    
    if score > best_f1:
        best_f1 = score
        best_threshold = threshold
        best_tpr = tpr[i]
        best_fpr = fpr[i]

print(f"Optimal Threshold: {best_threshold:.4f}")
print(f"TPR at Optimum: {best_tpr:.4f}")
print(f"FPR at Optimum: {best_fpr:.4f}")
print(f"F1 Score at Optimum: {best_f1:.4f}")

# Optimal threshold for the F1 score is often different than the standard 0.5.
# I would choose a threshold lower than 0.5 when false negative consequences are more important that than false positives.

## GridSearchCV Question 1

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("log_reg", LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {"log_reg__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring="roc_auc")

grid_search.fit(X_train, y_train)

best_c = grid_search.best_params_["log_reg__C"]
best_cv_auc = grid_search.best_score_

best_estimator = grid_search.best_estimator_
y_pred_proba_best = best_estimator.predict_proba(X_test)[:, 1]
test_auc_best = roc_auc_score(y_test, y_pred_proba_best)

print(f"Best C value: {best_c}")
print(f"Best CV AUC score: {best_cv_auc:.4f}")
print(f"Test AUC of the best estimator: {test_auc_best:.4f}")


# grid search choose 0.77, i would have chosen 0.7 so thats pretty close.
# the 0.77 chosen by the AUC is a big change from the defaul 1.0.

## GridSearch Question 2

dt_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("decision_tree", DecisionTreeClassifier(random_state=42))
])

dt_param_grid = {
    "decision_tree__max_depth": [2, 3, 5, 8, None]
}

dt_grid_search = GridSearchCV(
    estimator=dt_pipeline,
    param_grid=dt_param_grid,
    cv=5,
    scoring="roc_auc"
)

dt_grid_search.fit(X_train, y_train)

best_max_depth = dt_grid_search.best_params_["decision_tree__max_depth"]
best_cv_auc_dt = dt_grid_search.best_score_

best_dt_estimator = dt_grid_search.best_estimator_
y_pred_proba_dt = best_dt_estimator.predict_proba(X_test)[:, 1]
test_auc_dt = roc_auc_score(y_test, y_pred_proba_dt)

print(f"Best max_depth: {best_max_depth}")
print(f"Best CV AUC score (Decision Tree): {best_cv_auc_dt:.4f}")
print(f"Test AUC of the best Decision Tree estimator: {test_auc_dt:.4f}")

# Logistic Regression is what i would choose for further development.
# No, other critical factors must be considered.

## Gridsearch Question 3

cv_results = grid_search.cv_results_

mean_scores = cv_results["mean_test_score"]
std_scores = cv_results["std_test_score"]
params = cv_results["params"]

results_list = sorted(
    zip(mean_scores, std_scores, params),
    key=lambda x: x[0],
    reverse=True
)

print(f"{'Mean AUC':<12} {'Std Dev':<12} {'Parameters'}")

for mean_score, std_score, param in results_list:
    print(f"{mean_score:<12.4f} {std_score:<12.4f} {param}")

# If two parameter settings produce nearly identical mean cross-validation AUC scores, 
# we should pick the one with the lower standard deviation, as this indicates better performance
# of the model.

## joblib Question 1

joblib.dump(best_estimator, "models/warmup_model.pkl")

loaded_clf = joblib.load("models/warmup_model.pkl")

original_preds = best_estimator.predict(X_test)
loaded_preds = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# the model would receive raw, unscaled data and because it was trained on normalized data,
# it would cause incorrect predictions.

## joblib Question 2

joblib.dump(best_estimator, "models/warmup_model.pkl")

production_model = joblib.load("models/warmup_model.pkl")

new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

predicted_classes = production_model.predict(new_samples)
predicted_probabilities = production_model.predict_proba(new_samples)

for i, sample in enumerate(new_samples):
    pred_class = predicted_classes[i]
    prob_positive = predicted_probabilities[i][1]
    print(f"Sample {i+1}: Predicted Class = {pred_class}, Probability (Class 1) = {prob_positive:.4f}")

# Comment:
# What do you expect the all-zeros row to predict? Why?
# 
# For the all-zeros row, the model will output a prediction purely from the intercept term,
# If the intercept is negative, it predicts class 0; if positive, class 1.