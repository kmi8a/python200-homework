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



