import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# the parameter sep=';' has to be used to load the csv.

OUTPUT = Path('outputs/')

## Task 1: Load and Explore
df = pd.read_csv('resources/student_performance_math.csv', sep=';')

print(df.shape)
print(df.head(5))
print(df.dtypes)

plt.figure(figsize=(8, 6))
plt.hist(df['G3'], bins=21, edgecolor='black')

plt.title("Distribution of Final Math Grades")
plt.xlabel("G3")
plt.ylabel("Students")

plt.savefig(OUTPUT / "g3_distribution.png")


## Task 2: Preprocess the Data
filtered_df = df[df['G3'] != 0].copy()

print(f'Shape before: {df.shape}')
print(f'Shape after: {filtered_df.shape}')

## keeping these rows skews the final data, affecting the mean and impacting the performance of the prediction model.

binary_cols = ['schoolsup', 'internet', 'higher', 'activities', 'sex']

for col in binary_cols:
    filtered_df[col] = filtered_df[col].map({'yes': 1, 'M': 1, 'no': 0, 'F': 0})

corr_orig = df['absences'].corr(df['G3'])
corr_filtered = filtered_df['absences'].corr(filtered_df['G3'])

print(f"Correlation (Original): {corr_orig}")
print(f"Correlation (Filtered): {corr_filtered}")

# The unfiltered dataset includes a cluster of students with G3=0, when included these skew the data, masking the actual influence of 'absences'.


## Task 3: Exploratory Data Analysis

numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences', 'freetime', 'goout', 'Walc']

correlations = filtered_df[numeric_cols].corrwith(filtered_df['G3'])
sorted_corrs = correlations.sort_values(ascending=True)

print("Correlations with G3 (sorted from negative to positive):")
print(sorted_corrs)

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

## Task 3: Exploratory Data Analysis

numeric_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences', 'freetime', 'goout', 'Walc']

correlations = filtered_df[numeric_cols].corrwith(filtered_df['G3'])
sorted_corrs = correlations.sort_values(ascending=True)

print("Correlations with G3 (sorted from negative to positive):")
print(sorted_corrs)

# The feature with the strongest relationship to G is 'failures', followed by 'absences'.

# plot to explore influence of past failures on G3

fig2, ax2 = plt.subplots(figsize=(8, 6))
filtered_df.boxplot(column='G3', by='failures', ax=ax2, patch_artist=True)

plt.title("Past failures influence on grades")
plt.suptitle("")
plt.xlabel("Number of past failures")
plt.ylabel("Final Grade (G3)")
plt.savefig(OUTPUT / 'pastfailures_inluence_g3.png')
plt.show()

# we can see that as the number of failures increases, the grades decrease.

# plot to explore parent education influence on G3

# Plot influence of Mother's and Father's education on G3
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
plt.savefig(OUTPUT / 'parents_education_inluence_g3.png')
plt.show()

