import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


# Pandas Question 1
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}

df = pd.DataFrame(data)

print(f"First Three Rows:\n{df.head(3)}")
print(f"Shape:\n{df.shape}")
print(f"Data Types:\n{df.dtypes}")


# Pandas Question 2

filtered_df = df[(df['passed'] == True) & (df['grade'] > 80)]
print(filtered_df)

# Pandas Question 3
df['grade_curved'] = df['grade'] + 5
print(df)

# Pandas Question 4
df['name_upper'] = df['name'].str.upper()
print(df[['name', 'name_upper']])

# Pandas Question 5
mean_grades_by_city = df.groupby('city')['grade'].mean()
print(mean_grades_by_city)

# Pandas Question 6
df['city'] = df['city'].replace('Austin', 'Houston')
print(df[['name', 'city']])

# Pandas Question 7
top3 = df.sort_values(by='grade', ascending=False).head(3)
print(top3)

# Numpy Question 1
array = np.array([10, 20, 30, 40, 50])

print(f"shape: {array.shape}")
print(f"dtype: {array.dtype}")
print(f"ndim: {array.ndim}")

# Numpy Question 2
array2 = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print(f"shape: {array2.shape}")
print(f"size: {array2.size}")

# Numpy Question 3
block = array2[:2,:2]
print(block)

# Numpy Question 4
zeros = np.zeros((3, 4))
ones = np.ones((2, 5))

print(zeros)
print(ones)

# Numpy Question 5
q5 = np.arange(0, 50, 5)

q5_shape = q5.shape
q5_mean = np.mean(q5)
q5_sum = np.sum(q5)
q5_std = np.std(q5)

print(q5)
print(q5_shape)
print(q5_mean)
print(q5_sum)
print(f'{q5_std:.2f}')

# Numpy Question 6
data = np.random.normal(loc=0.0, scale=1.0, size=200)

data_mean = np.mean(data)
data_std = np.std(data)

print(f'{data_mean:.4f}')
print(f'{data_std:.4f}')

# Matplot Question 1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.figure()
plt.plot(x, y)
plt.title('Squares')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Matplot Question 2
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

data = sorted(zip(subjects, scores), key=lambda x: x[1])
subjects_sorted, scores_sorted = zip(*data)

plt.figure()
plt.bar(subjects_sorted, scores_sorted)
plt.title('Subject Scores')
plt.xlabel('Subjects')
plt.ylabel('Scores')
plt.show()

# Matplot Question 3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.figure()
plt.scatter(x1, y1, color='blue', label='Dataset 1')
plt.scatter(x2, y2, color='red', label='Dataset 2')

plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Matplot Question 3

fig, (pl1, pl2) = plt.subplots(1, 2, figsize=(10,4))

# left plot
pl1.plot(x, y)
pl1.set_title('Squares')
pl1.set_xlabel('x')
pl1.set_ylabel('y')

#right plot
pl2.bar(subjects_sorted, scores_sorted)
pl2.set_title("Subject Scores")
pl2.set_xlabel("Subjects")
pl2.set_ylabel("Scores")

plt.tight_layout
plt.show()

# Descriptive Stats Question 1
data_q1 = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

mean = np.mean(data_q1)
median = np.median(data_q1)
variance = np.var(data_q1)
std_dev = np.std(data_q1)

print(f'Mean: {mean}')
print(f'Median: {median}')
print(f'Variance: {variance:.3f}')
print(f'Standard Deviation: {std_dev:.3f}')

# Descriptive Stats Question 2
data_q2 = np.random.normal(65, 10, 500)

plt.figure(figsize=(10, 6))
plt.hist(data_q2, bins=20, edgecolor='black', alpha=0.7)
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.show()

# Descriptive Stats Question 3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.figure(figsize=(8, 6))
plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Scores")
plt.show()

# Descriptive Stats Question 4
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.figure(figsize=(8, 6))
plt.boxplot([normal_data, skewed_data], tick_labels=['Normal', 'Exponential'])
plt.title('Distibution Comparison')
plt.ylabel('Value')
plt.show()

## 1. exponential distribution is more skewed in comparison to the normal distribution.
## 2. for the normal distribution: both the mean and median are appropiate, as their distribution is symmetric.
##    for the exponential distribution: the median is more appropiate, because the the mean is pulled towards the long tail.

# Descriptive Stats Question 5
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

mean1 = np.mean(data1)
median1 = np.median(data1)
mode1 = stats.mode(data1, keepdims=True).mode[0]

mean2 = np.mean(data2)
median2 = np.median(data2)
mode2 = stats.mode(data2, keepdims=True).mode[0]

print(f'Data 1:\n   Mean: {mean1}, Median: {median1}, Mode: {mode1}')
print(f'Data 2:\n   Mean: {mean2}, Median: {median2}, Mode: {mode2}')

## The mean for data2 is so different because of the value 150, which is an outlier and heavily skews the data towards it.

# Hypothesis Question 1
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stats, p_value = stats.ttest_ind(group_a, group_b)

print(f't-statistic: {t_stats}')
print(f'p-value: {p_value}')

# Hypothesis Question 2
alpha = 0.05

if p_value < alpha:
    print("The result is statistically significant.")
else:
    print("The result is not statistically significant.")

# Hypothesis Question 3
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stats, p_value = stats.ttest_rel(before, after)

print(f't-statistic: {t_stats}')
print(f'p-value: {p_value}')

# Hypothesis Question 4
scores = [72, 68, 75, 70, 69, 74, 71, 73]
benchmark = 70

t_stats, p_value = stats.ttest_1samp(scores, benchmark)

print(f't-statistic: {t_stats}')
print(f'p-value: {p_value}')

# Hypothesis Question 5
t_stats, p_value = stats.ttest_ind(group_a, group_b, alternative='less')

print(f'p-value: {p_value}')

# Hypothesis Question 6
print("The average score for Group A is lower than the average score for Group B, it's unlikely that this difference occurred due to random chance.")

# Correlation Question 1
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

correlation_matrix = np.corrcoef(x, y)
corr_coef = correlation_matrix[0, 1]

print(correlation_matrix)
print(corr_coef)

## Expected result: corr is expected to be 1.0
## Reasoning: the relationship between X and Y is linear, as X increases Y increases proportionally.

# Correlation Question 2
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [10, 9, 7, 8, 6, 5, 3, 4, 2, 1]

corr, p_value = stats.pearsonr(x, y)

print(f'Correlation Coefficient: {corr}')
print(f'P-value: {p_value}')

# Correlation Question 3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)

correlation_matrix = df.corr()

print(correlation_matrix)

# Correlation Question 4
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.figure(figsize=(8, 6))
plt.scatter(x, y)
plt.title('Negative Correlation')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Correlation Question 5
fig, ax = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, ax=ax)
plt.title("Correlation Heatmap")
plt.show()

# Pipeline Question 1
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr, name='values')

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {'mean': series.mean(), 'median': series.median(), 'std': series.std(), 'mode': series.mode()[0]}

def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    return summarize_data(cleaned_series)

result = data_pipeline(arr)

for key, value in result.items():
    print(f'{key}: {value}')