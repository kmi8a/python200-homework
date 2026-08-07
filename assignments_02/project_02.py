import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# --- Preprocessing ---
# the parameter sep=';' has to be used to load the csv.

OUTPUT = Path('outputs/')

# ---Task 1: Load and Explore ---
df = pd.read_csv('resources/student_performance_math.csv', sep=';')

print(df.shape)
print(df.head(5))
print(df.dtypes)

# Plot distribution of final math grades (G3)

plt.figure(figsize=(8, 6))
plt.hist(df['G3'], bins=21, edgecolor='black')

plt.title("Distribution of Final Math Grades")
plt.xlabel("G3")
plt.ylabel("Students")

plt.savefig(OUTPUT / "g3_distribution.png")


# --- Task 2: Preprocess the Data ---
filtered_df = df[df['G3'] != 0].copy()

print(f'Shape before: {df.shape}')
print(f'Shape after: {filtered_df.shape}')

# keeping these rows skews the final data, affecting the mean and impacting the performance of the prediction model.

binary_cols = ['schoolsup', 'internet', 'higher', 'activities', 'sex']

for col in binary_cols:
    filtered_df[col] = filtered_df[col].map({'yes': 1, 'M': 1, 'no': 0, 'F': 0})

corr_orig = df['absences'].corr(df['G3'])
corr_filtered = filtered_df['absences'].corr(filtered_df['G3'])

print(f"Correlation (Original): {corr_orig}")
print(f"Correlation (Filtered): {corr_filtered}")

# The unfiltered dataset includes a cluster of students with G3=0, when included these skew the data, masking the actual influence of 'absences'.


# --- Task 3: Exploratory Data Analysis ---

numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences', 'freetime', 'goout', 'Walc']

correlations = filtered_df[numeric_cols].corrwith(filtered_df['G3'])
sorted_corrs = correlations.sort_values(ascending=True)

print("Correlations with G3 (sorted from negative to positive):")
print(sorted_corrs)

# The feature with the strongest relationship to G is 'failures', followed by 'absences'.

# Visualization 1 - Absences comparison (Original vs Filtered)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(df['absences'], df['G3'], alpha=0.5)
axes[0].set_title(f'Original Data (Corr: {corr_orig:.3f})')
axes[0].set_xlabel('Absences')
axes[0].set_ylabel('G3')

axes[1].scatter(filtered_df['absences'], filtered_df['G3'], alpha=0.5)
axes[1].set_title(f'Filtered Data (Corr: {corr_filtered:.3f})')
axes[1].set_xlabel('Absences')
axes[1].set_ylabel('G3')

plt.tight_layout()
plt.savefig(OUTPUT / 'comparison_absences_g3.png')
plt.show()
plt.close(fig)

# Visualization 2 - Influence of past failures on G3

fig2, ax2 = plt.subplots(figsize=(8, 6))
filtered_df.boxplot(column='G3', by='failures', ax=ax2, patch_artist=True)

plt.title("Past failures influence on grades")
plt.suptitle("")
plt.xlabel("Number of past failures")
plt.ylabel("Final Grade (G3)")
plt.savefig(OUTPUT / 'past_failures_inluence_g3.png')
plt.show()

# we can see that as the number of failures increases, the grades decrease.

# Visualization 3 - Influence of Mother's and Father's education on G3

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

filtered_df.boxplot(column='G3', by='Medu', ax=axes[0], patch_artist=True, boxprops=dict(facecolor='lightblue'))
axes[0].set_title("G3 vs Mother's Education")
axes[0].set_xlabel("Medu (0:none, 4:higher)")
axes[0].set_ylabel("Final Grade (G3)")

filtered_df.boxplot(column='G3', by='Fedu', ax=axes[1], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
axes[1].set_title("G3 vs Father's Education")
axes[1].set_xlabel("Fedu (0:none, 4:higher)")
axes[1].set_ylabel("Final Grade (G3)")

plt.suptitle("Influence Parents Education on Student Performance")
plt.tight_layout()
plt.savefig(OUTPUT / 'parents_education_influence_g3.png')
plt.show()

# Higher levels of parental education correspond to higher final grades.


# --- Task 4: Baseline Model ---

X = filtered_df[['failures']]
y = filtered_df['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)

slope = baseline_model.coef_[0]
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Slope (Coefficient): {slope:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2 Score: {r2:.3f}")


# The slope indicates how many grade points a student drops on average for each additional past failure.
# The RMSE tells us the typical magnitude of prediction errors our model makes on unseen test data.
# The R2 score is relatively low (typically around 0.10 - 0.20), it matches expectations from EDA.
# While past failures show a noticeable trend, relying on a single feature captures only a fraction of the variance 
# in student performance, highlighting the need for a multi-feature model.


## --- Task 5: Build the Full Model ---

feature_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex"
                ]

df_clean = filtered_df
X_full = df_clean[feature_cols].values
y_full = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

full_model = LinearRegression()
full_model.fit(X_train, y_train)

y_train_pred = full_model.predict(X_train)
y_test_pred = full_model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Train R2: {train_r2:.3f}")
print(f"Test R2: {test_r2:.3f}")
print(f"Test RMSE: {test_rmse:.3f}")

# Adding more features increases the test R2, but academic performance remains complex to predict without intermediate grades.

print("\nFeature Coefficients:")
for name, coef in zip(feature_cols, full_model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# failures being the biggest coefficient really surprised me, and also the difference with the other factor is quite noticeable.
# it shows how important is to have discipline and consistency.
#
# The gap between train and test R2 is small, it indicates that the model is not overfitting, and has reached top performance with the features provided.
#
# For deployment, I would retain features like 'failures', 'absences', and 'higher' , while dropping features with not much impact to prevent unnecessary noise, 
# and improve interpretability.


# --- Task 6: Evaluate and Summarize ---

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(y_test_pred, y_test, alpha=0.6, edgecolors='k')

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal (y = y_pred)')

ax.set_title("Predicted vs Actual (Full Model)")
ax.set_xlabel("Predicted Grade (y_pred)")
ax.set_ylabel("Actual Grade (y)")
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT / 'predicted_vs_actual_g3.png')
plt.show()

# The model tends to struggle more at the extremes.
# A point above the diagonal means the grade was higher than predicted (underestimation).
# A point below the diagonal means the grade was lower than predicted (overestimation).


# Summary:
# The dataset contains 357 student records (after dropping 38 rows where G3 = 0).
# The test set contains 72 samples (20% split).
#
# Test RMSE is 2.664, this means that when the model predicts a student's final grade, its prediction error is about 2.66 points in either direction.
# Test R2 is 0.263, this means that the 15 available features explain around 26.3% of the variance in final math grades.
# 
# The largest positive coefficients are found on the 'higher' or 'studytime', this indicates that big academic ambitions and consistent study habits
# drive predicted final grades upward.
# The largest negative coefficients are found on 'failures', demonstrating that past academic failures weight heavily, pushing down final performance.
#
# The close number between Train R2 and Test R2, this shows that the linear model is not overfitting, but have rather reached a point where
# the available features are not enough to explain student performance.

# --- Neglected Feature: The Power of G1 ---

feature_cols_G1 = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "absences", "freetime", "goout", "Walc", "schoolsup",
    "internet", "higher", "activities", "sex", "G1"
    ]

X_G1 = filtered_df[feature_cols_G1].values
y_G1 = filtered_df["G3"].values

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_G1, y_G1, test_size=0.2, random_state=42)

g1_model = LinearRegression()
g1_model.fit(X_train_b, y_train_b)

y_test_pred_b = g1_model.predict(X_test_b)
test_r2_G1 = r2_score(y_test_b, y_test_pred_b)
test_rmse_G1 = np.sqrt(mean_squared_error(y_test_b, y_test_pred_b))

print(f"Test R2 (with G1): {test_r2_G1:.3f}")
print(f"Test RMSE (with G1): {test_rmse_G1:.3f}")



# Does a high R² mean G1 causes G3?
# No, a high G1 doesn't automatically causes a high G3.
# G1 can be interpreted as an indicator for student habits, attendance, and continuous effort through the school year.
#
# Is this a useful model for identifying students who might struggle?
# No, statistically it is very accurate, but in practice its utility is limited as it tells
# the story after it has occurred, by the time G1 is available, struggling students are already behind.
#
# What might educators need to do if they wanted to intervene early, before G1 is even available?
# this is a complex topic, but if we just focus on the data available, educators can focus on behavioral and demographic 
# indicators such as absences, past failures, promoting the adoption of better study time habits, and being aware of socio-economic and family 
# factors (like parental education or internet access) to flag students to be helped from the beginning of the year.