import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from pathlib import Path

OUTPUT = Path('outputs/')

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---
# Q1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Q2

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print([f"{val:.2f}" for val in X_train_scaled.mean(axis=0)])

## scaler is fit exclusively on X_train to prevent data leakage from the test set.

# --- KNN ---
# Q1

knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)

y_pred_unscaled = knn_unscaled.predict(X_test)

print(f"Accuracy Score: {accuracy_score(y_test, y_pred_unscaled):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred_unscaled))

# Q2

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = knn_scaled.predict(X_test_scaled)

print(f"Accuracy Score: {accuracy_score(y_test, y_pred_scaled):.4f}")

## Scaling makes no difference in accuracy here as all four features are measured in the same units and have comparable scales.

# Q3

cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_train, y_train, cv=5)

for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i} score: {score:.4f}")

print(f"\nMean CV Score: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

## This method is more trustworthy, it evaluates the model across different training and validation folds, reducing the variance and the risk of a single train/test split.

# Q4

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5)
    print(f"k={k}: Mean CV Score = {cv_scores.mean():.4f}")

## I would choose k5 or k7 as they have the more balance between high cross-validation performance and stable generalization.

# --- Classifier Evaluation ---
# Q1

cm = confusion_matrix(y_test, y_pred_unscaled)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues')
plt.savefig(OUTPUT / "knn_confusion_matrix.png", bbox_inches='tight')
plt.close()

# The model predicts all 3 species accurately, it doesnt confuse any two species.

# ---The sklearn API: Decision Trees ---
# Q1

# --- Logistic Regression and Regularization ---
# Q1

c_values = [0.01, 1.0, 100]

for c in c_values:
    model = LogisticRegression(C=c, max_iter=1000, random_state=42)
    #model = LogisticRegression(C=c, max_iter=1000, solver='liblinear')
    model.fit(X_train_scaled, y_train)
    coef_sum = np.abs(model.coef_).sum()
    print(f"C={c}: Total coefficient magnitude = {coef_sum:.4f}")

## As C increases, the total coefficient magnitude also increases.
## regularization is shrinking big values.

# --- PCA ---

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# Q1

print(f"X_digits shape: {X_digits.shape}")
print(f"images shape: {images.shape}")

fig, axes = plt.subplots(1, 10, figsize=(12, 2))
for i in range(10):
    idx = np.where(y_digits == i)[0][0]
    axes[i].imshow(images[idx], cmap="gray_r")
    axes[i].set_title(str(i))
    axes[i].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT / "sample_digits.png", bbox_inches="tight")
plt.close()

# Q2

pca = PCA()
scores = pca.fit_transform(X_digits)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)
plt.colorbar(scatter, label='Digit')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA 2D Projection of Digits Dataset')

plt.savefig(OUTPUT / "pca_2d_projection.png", bbox_inches='tight')
plt.close()

## Images of the same digit tend to form distinct clusters in this 2D space, some classes overlap due to
## shrinking 64 dimensions to just 2.

# Q3

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', markersize=4)
ax.axhline(y=0.80, color='r', linestyle='--', label='80% Variance Threshold')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Explained Variance Ratio')
ax.set_title('PCA Cumulative Explained Variance')
ax.legend()

plt.savefig(OUTPUT / "pca_variance_explained.png", bbox_inches='tight')
plt.close()

# You need approximately 13 components to explain 80% of the variance in the digits dataset.

# Q4

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_values = [2, 5, 15, 40]
n_samples = 5

fig, axes = plt.subplots(len(n_values) + 1, n_samples, figsize=(10, 2 * (len(n_values) + 1)))

# Original images
for j in range(n_samples):
    axes[0, j].imshow(images[j], cmap="gray_r")
    axes[0, j].axis("off")
    if j == 0:
        axes[0, j].set_title("Original")

# Reconstructions for different n values
for row_idx, n_comp in enumerate(n_values, start=1):
    for j in range(n_samples):
        recon = reconstruct_digit(j, scores, pca, n_comp)
        axes[row_idx, j].imshow(recon, cmap="gray_r")
        axes[row_idx, j].axis("off")
        if j == 0:
            axes[row_idx, j].set_title(f"n={n_comp}")

plt.tight_layout()
plt.savefig(OUTPUT / "pca_reconstructions.png", bbox_inches="tight")
plt.close()

# The digits become recognizable around n=15, which matches where the variance curves level off.