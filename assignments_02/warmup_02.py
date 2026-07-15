import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT = Path('outputs/')

## scikit-learn Question 1

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000]).reshape(-1, 1)

predict = np.array([4, 8]).reshape(-1, 1)

model = LinearRegression()
model.fit(years, salary)
salary_pred = model.predict(predict)


print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print(f"Prediction at 4 years: {salary_pred[0]}")
print(f"Prediction at 8 years: {salary_pred[1]}")

## scikit-learn Question 2

x = np.array([10, 20, 30, 40, 50])
print(x.shape)

x = x.reshape(-1, 1)
print(x.shape)

# scikit-learn needs x to be 2d in order to perfom the calculations, if the data is 1d
# it wouldn't know if the data represents 6 different types of data or 6 observations of the same type.

## scikit-learn Question 3

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)

print(f"Cluster Centers: {kmeans.cluster_centers_}")
print(f"Points per cluster:: {np.bincount(labels)}")

plt.figure(figsize=(8, 6))
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', s=60, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='x', s=200, label='Centroids')
plt.title("Clusters Found by K-Means")
plt.xlabel("X")
plt.ylabel("Y", rotation=0)
plt.legend()
plt.savefig(OUTPUT / "correlation_heatmap.png")

## Linear Regression Question 1

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

plt.figure(figsize=(8, 6))
plt.scatter(age, cost, c=smoker, cmap='coolwarm', s=60, alpha=0.7)
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.savefig(OUTPUT / "cost_vs_age.png")

## Based on the scatter plot I can see that the data is spread, theres two visible groups suggesting that the smoker variable has an important impact.

## Linear Regression Question 2

x = age.reshape(-1, 1)
y = cost

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print(f'X_train shape: {x_train.shape}')
print(f'X_test shape: {x_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

## Linear Regression Question 3

model = LinearRegression()
model.fit(x_train, y_train)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(x_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(x_test, y_test)

print("RMSE:", rmse)
print("R²:", r2)

# the slope indicates that for every year the costs of medical care increase around 196.5 if all other factors remain constant. 

## Linear Regression Question 4

x_full = np.column_stack([age, smoker])

x_train, x_test, y_train, y_test = train_test_split(
    x_full, y, test_size=0.2, random_state=42
)

full_model = LinearRegression()
full_model.fit(x_train, y_train)

full_y_pred = full_model.predict(x_test)

full_r2 = full_model.score(x_test, y_test)

print("R² (Age only):", r2)
print("R² (Age + Smoker):", full_r2)
print("age coefficient:    ", full_model.coef_[0])
print("smoker coefficient: ", full_model.coef_[1])

# In practical terms, the smoker coefficient tells us that the model got a much better performance when using the smoker column, this is evidenced by
# the increase in the R2 score in comparison to when t=only the Age columnwas used.

## Linear Regression Question 5

plt.figure(figsize=(8, 6))
plt.scatter(full_y_pred, y_test, alpha=0.7)

min_val = min(full_y_pred.min(), y_test.min())
max_val = max(full_y_pred.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

plt.title("Predicted vs. Actual Costs")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.savefig(OUTPUT / "predicted_vs_actual.png")

# Points that falls above the diagonal mean that the actual value is higher that predicted, points that fall below the diagonal mean that the actual
# value is lower tham the predicted one and points on the diagonal mean that the model made a perfect prediction.

