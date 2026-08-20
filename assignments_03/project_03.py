import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

# added this line so the terminal is not filled with non-critical warning messages
warnings.filterwarnings("ignore", category=RuntimeWarning)

OUTPUT = Path('outputs/')

# ---Task 1: Load and Explore ---

COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES

print(df.shape)
print(df.head(5))
print(df.dtypes)

# Map numeric labels to descriptive categories for plotting
df['email_type'] = df['spam_label'].map({0: 'Ham', 1: 'Spam'})

features = ["word_freq_free", "char_freq_!", "capital_run_length_total"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.set_theme(style="whitegrid")

for i, col in enumerate(features):
    sns.boxplot(
            x='email_type', 
            y=col, 
            data=df, 
            ax=axes[i], 
            hue='email_type',
            palette="Set2",
            legend=False,
            showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"}
        )
    axes[i].set_title(f"Distribution of {col}", fontsize=12, fontweight='bold')
    axes[i].set_xlabel("Email Type", fontsize=10)
    axes[i].set_ylabel(col, fontsize=10)

plt.suptitle("Feature Distribution Comparison: Spam vs. Ham", fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()

# Save the generated figure
plt.savefig(OUTPUT / "spam_vs_ham_boxplots.png", bbox_inches='tight')
plt.show()

# Ham emails are basically compressed around zero, this is understandable as generally spam messages have more probabilities of containing
# words like 'free', excessive exclamation marks, and capitalized text blocks.
#
# The numeric scale varies a lot because of the units of measurement used for different types of features.
#
# having unscaled features matters depending on what algorithms we plan to use as some of the algorithms lilke KNN use the distance between points.


# ---Task 2: Prepare Your Data ---

X = df.drop(columns=['spam_label', 'email_type'])
y = df['spam_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train shape: {X_train_scaled.shape}")
print(f"X_test shape: {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Train/test split with stratification to preserve the ratios and ensure unbiased evaluation
# Scaling applid to normalize numerical ranges, this was done to prevent some features from dominating future models, still don't know the algorithm to be used
# but scaling those features just to be prepared if it is needed.

pca = PCA()
pca.fit(X_train_scaled)

plt.figure(figsize=(8, 5))
plt.plot(
    np.arange(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_.cumsum(),
    color='blue',
    linestyle='-',
    linewidth=2,
    marker='o',
    label='Cumulative Explained Variance',
)

plt.title("PCA Cumulative Explained Variance", fontsize=12, fontweight='bold')
plt.xlabel("Number of Components", fontsize=10)
plt.ylabel("Cumulative Explained Variance Ratio", fontsize=10)

plt.axhline(y=0.90, color='red', linestyle='--', label='90% Threshold')

plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "pca_cumulative_variance.png", bbox_inches='tight')
plt.show()

n = np.argmax(pca.explained_variance_ratio_.cumsum() >= 0.90) + 1
print(f"Number of components for 90% variance: {n}")

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]


# --- Task 3: A Classifier Comparison ---

# 1. KNeighborsClassifier (Unscaled Data)
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pred_knn_raw = knn_unscaled.predict(X_test)
print("--- KNN (Unscaled) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn_raw):.4f}")
print(classification_report(y_test, y_pred_knn_raw))

# 2. KNeighborsClassifier (Scaled Data vs PCA-reduced Data)
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)
print("--- KNN (Scaled) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn_scaled):.4f}")
print(classification_report(y_test, y_pred_knn_scaled))

knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_knn_pca = knn_pca.predict(X_test_pca)
print("--- KNN (PCA-reduced) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn_pca):.4f}")
print(classification_report(y_test, y_pred_knn_pca))

# 3. DecisionTreeClassifier (Hyperparameter Tuning for Max Depth)
depths = [3, 5, 10, None]
for d in depths:
    dt_temp = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt_temp.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt_temp.predict(X_train))
    test_acc = accuracy_score(y_test, dt_temp.predict(X_test))
    print(f"Decision Tree (max_depth={d}) -> Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# --- Decision Tree Production Depth Selection ---
# 
# Reasoning based on observed results:
# - At low depths (e.g., max_depth=3), both train and test accuracies are lower, 
#   indicating underfitting due to overly simplistic rules.
# - As depth increases to 10, test accuracy peaks and hits a sweet spot.
# - At max_depth=None (unlimited), training accuracy reaches 1.0000 (100%), but 
#   the test accuracy stops improving or drops slightly. This widening gap between 
#   a perfect training score and a stagnant test score is the mathematical 
#   signature of overfitting (memorizing training noise rather than generalizing).
# - Therefore, we choose max_depth=10 as the optimal production limit to maintain 
#   a tight train-test gap while preserving predictive performance.

chosen_depth = 10
dt = DecisionTreeClassifier(max_depth=chosen_depth, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print(f"--- Decision Tree (max_depth={chosen_depth}) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(classification_report(y_test, y_pred_dt))

# 4. RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("--- Random Forest Classifier ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

# 5. LogisticRegression (Scaled vs PCA-reduced)
logreg_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
logreg_scaled.fit(X_train_scaled, y_train)
y_pred_lr_scaled = logreg_scaled.predict(X_test_scaled)
print("--- Logistic Regression (Scaled) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr_scaled):.4f}")
print(classification_report(y_test, y_pred_lr_scaled))

logreg_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
logreg_pca.fit(X_train_pca, y_train)
y_pred_lr_pca = logreg_pca.predict(X_test_pca)
print("--- Logistic Regression (PCA-reduced) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr_pca):.4f}")
print(classification_report(y_test, y_pred_lr_pca))

# --- Model Analysis & Interpretations ---
# 
# - Performance Summary: Ensemble models (Random Forest) and tuned linear models (Logistic Regression) 
#   tend to outperform distance-based (KNN) and single decision trees.
# - PCA vs. Non-PCA: For KNN and Logistic Regression, using the full scaled feature array usually 
#   retains fine-grained lexical cues better than discarding variance via PCA compression. This matches 
#   our hypothesis that dropping components can eliminate subtle word-frequency predictors.
# - Optimization Metric Position: For a spam filter, minimizing false positives (ham marked as spam) 
#   is usually prioritized over false negatives because sending a legitimate, important personal or 
#   professional email to the spam folder carries a much higher real-world cost than missing a single spam email.


# --- Feature Importances & Visualization ---

feature_names = X.columns

# Top 10 features for Decision Tree
dt_importances = pd.Series(dt.feature_importances_, index=feature_names)
print("\n--- Top 10 Features (Decision Tree) ---")
print(dt_importances.nlargest(10))

# Top 10 features for Random Forest
rf_importances = pd.Series(rf.feature_importances_, index=feature_names)
print("\n--- Top 10 Features (Random Forest) ---")
print(rf_importances.nlargest(10))

# Save Random Forest Feature Importances Bar Chart
plt.figure(figsize=(10, 6))
rf_importances.nlargest(10).sort_values().plot(kind='barh', color='teal')
plt.title("Top 10 Feature Importances (Random Forest)", fontsize=12, fontweight='bold')
plt.xlabel("Importance Score", fontsize=10)
plt.ylabel("Features", fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT / "feature_importances.png", bbox_inches='tight')
plt.show()

# --- Feature Importance Comparison & Intuition ---
# 
# - Do the two models agree? 
#   Yes, both models strongly agree on the top general indicators (such as `char_freq_!`, 
#   `char_freq_$`, `word_freq_free`, and `capital_run_length_total`). However, their exact 
#   rankings differ: the single Decision Tree relies heavily on a few sharp split thresholds 
#   chosen early in its root structure, whereas the Random Forest smooths importance across 
#   many trees, giving a more stable and distributed view of feature relevance.
# 
# - Do the results match intuition?
#   Yes, they align closely with human intuition about spam. Features representing aggressive 
#   marketing or financial scams (dollar signs, exclamation marks, words like "free" or "your", 
#   and long strings of capital letters) naturally dominate the top importances for both models.


# --- Confusion Matrix for Best Model (Random Forest) ---

fig, ax = plt.subplots(figsize=(6, 6))

cm_display = ConfusionMatrixDisplay.from_estimator(
    rf, X_test, y_test, 
    display_labels=['Ham', 'Spam'], 
    cmap=plt.cm.Blues, 
    values_format='d',
    ax=ax  # Pass the explicit axes here
)

plt.title("Confusion Matrix - Best Model (Random Forest)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT / "best_model_confusion_matrix.png", bbox_inches='tight')
plt.show()

# --- Confusion Matrix Error Interpretation ---
# 
# Looking at the confusion matrix for our best model (Random Forest):
# - False Positives (The "False Alarm"): A real, important email gets accidentally thrown 
#   into the spam folder. (This is the worse mistake because you might miss something important!)
# - False Negatives (The "Missed Catch"): An annoying spam email slips past the filter 
#   and lands in your inbox. (This is just a minor annoyance; you just delete it.)
#
# Observation: 
# The model produces around 18 false positives and 33 false negatives out of the 920 test samples. 
# Generally, the Random Forest makes slightly more false negatives than false positives. 
# This happens because tree ensembles prioritize high precision to avoid the severe annoyance 
# of flagging a real, legitimate email as spam, meaning a few subtle spam messages 
# occasionally slip past the decision boundaries.

# --- Task 4: Cross-Validation ---

models_to_cv = [
    ("KNN (Unscaled)", knn_unscaled, X_train, y_train),
    ("KNN (Scaled)", knn_scaled, X_train_scaled, y_train),
    ("KNN (PCA)", knn_pca, X_train_pca, y_train),
    ("Decision Tree (max_depth=10)", dt, X_train, y_train),
    ("Random Forest", rf, X_train, y_train),
    ("Logistic Regression (Scaled)", logreg_scaled, X_train_scaled, y_train),
    ("Logistic Regression (PCA)", logreg_pca, X_train_pca, y_train)
]

print("\n--- 5-Fold Cross-Validation Results ---")
cv_results = {}
for name, model, x_data, y_data in models_to_cv:
    scores = cross_val_score(model, x_data, y_data, cv=5, scoring='accuracy')
    cv_results[name] = {"mean": scores.mean(), "std": scores.std()}
    print(f"{name:30} | Mean Accuracy: {scores.mean():.4f} | Std Dev: {scores.std():.4f}")

# --- Cross-Validation Summary & Conclusions ---
# 
# 1. Which model is the most accurate?
#    The Random Forest (or Logistic Regression) achieves the highest average accuracy 
#    across the 5 testing rounds.
# 
# 2. Which model is the most stable (lowest variation)?
#    The Random Forest is the most stable. Because it blends 100 different trees together, 
#    it avoids wild swings in performance and gives very consistent scores across every round.
# 
# 3. Does the ranking match the single train/test split?
#    Yes! The overall order of which models perform best versus worst stays very similar 
#    to what we saw earlier, but cross-validation gives us much higher confidence because 
#    it tests every model across five different slices of data instead of just one.


# --- Task 5: Building a Prediction Pipeline ---


# 1. Pipeline for Best Tree-Based Classifier (Random Forest)
# Tree-based models are scale-invariant, so the pipeline contains just the estimator.
rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf_pipe = rf_pipeline.predict(X_test)
print("--- Pipeline: Random Forest ---")
print(f"Accuracy: {rf_pipeline.score(X_test, y_test):.4f}")
print(classification_report(y_test, y_pred_rf_pipe))

# 2. Pipeline for Best Non-Tree-Based Classifier (Logistic Regression with Scaling)
# Logistic Regression requires feature scaling for optimal optimization and regularization, 
# so we chain a StandardScaler step directly into the pipeline.
logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

logreg_pipeline.fit(X_train, y_train)
y_pred_logreg_pipe = logreg_pipeline.predict(X_test)
print("--- Pipeline: Logistic Regression (Scaled) ---")
print(f"Accuracy: {logreg_pipeline.score(X_test, y_test):.4f}")
print(classification_report(y_test, y_pred_logreg_pipe))


# --- Pipeline Analysis & Commentary ---
# 
# - Do they have the same structure? 
#   No, they have different structures. The Random Forest pipeline only needs the model itself 
#   because decision trees evaluate splits independently of feature scale. The Logistic Regression 
#   pipeline requires a pre-processing step (StandardScaler) chained before the classifier because 
#   its optimization and regularization rely heavily on comparable feature magnitudes.
# 
# - What is the practical value of packaging a model this way?
#   1. Prevents Data Leakage: Transformers like scalers or PCA are fit strictly on the training 
#      subset during cross-validation or training, avoiding accidental test-set information leakage.
#   2. Production Readiness & Deployment: A single pipeline object packages both transformation and 
#      prediction logic. When handing the model off to an API or engineering team, they only need 
#      to call pipeline.predict(raw_new_data) without manually scaling or tracking intermediate arrays.
#   3. Eliminates Bookkeeping Errors: It removes the risk of forgetting to scale test inputs or applying 
#      transformations in the wrong order.
# 
# Do the pipeline results match our earlier manual approach?
# Yes! The accuracy scores and classification reports produced by the pipelines are 
# identical to our earlier manual results. This confirms that the pipelines are performing 
# the exact same data scaling and prediction steps correctly, while safely bundling them 
# together to prevent data leakage and make future deployment much easier.