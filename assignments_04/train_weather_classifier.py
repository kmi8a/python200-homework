import json
import os
import platform
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

OUTPUT = Path('outputs/')

url = "https://archive-api.open-meteo.com/v1/archive"
latitude = 35.23
longitude = -80.84

params = {
    "latitude": 35.23,
    "longitude": -80.84,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "temperature_unit": "fahrenheit",
    "timezone": "America/New_York",
}

print("Fetching weather data from Open-Meteo API for Charlotte, NC...")
response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()
df = pd.DataFrame(data["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

df = df[
    [
        "date",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
]

# Print Dataset Summary

print("\n--- Dataset Successfully Loaded ---")
print(f"Total Records: {len(df)}")
print(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Summary Statistics:")
print(df.describe())


# Step 2: Engineer Labels (Fahrenheit Thresholds)

# Threshold definitions for "Good for running" using Fahrenheit:
# temperature_2m_max: 45°F to 85°F
# temperature_2m_min: >= 35°F
# precipitation_sum: < 3.0 mm
# wind_speed_10m_max: < 30 km/h

threshold_description = (
    "Good for running defined as: "
    "temperature_2m_max between 45°F and 85°F, "
    "temperature_2m_min >= 35°F, "
    "precipitation_sum < 3.0 mm, "
    "wind_speed_10m_max < 30 km/h."
)


def is_good_for_running(row):
    cond_max_temp = 45 <= row["temperature_2m_max"] <= 85
    cond_min_temp = row["temperature_2m_min"] >= 35
    cond_precip = row["precipitation_sum"] < 3.0
    cond_wind = row["wind_speed_10m_max"] < 30

    if cond_max_temp and cond_min_temp and cond_precip and cond_wind:
        return 1
    return 0

df["is_good_for_running"] = df.apply(is_good_for_running, axis=1)

class_counts = df["is_good_for_running"].value_counts()
class_proportions = df["is_good_for_running"].value_counts(normalize=True)

print("Label Distribution ('Good for running' = 1, Otherwise = 0)")
print(class_counts)
print("Proportions:")
print(class_proportions)


# Step 3: Train and Tune


feature_cols = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]
X = df[feature_cols]
y = df["is_good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42)),
    ]
)

param_grid = {"classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)

print("\nRunning GridSearchCV to tune hyperparameter C...")
grid_search.fit(X_train, y_train)

print("\nHyperparameter Tuning Results")
print(f"Best C Value: {grid_search.best_params_['classifier__C']}")
print(f"Best CV ROC AUC: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_pred_proba)

print("\nTest Set Evaluation")
print(f"Test ROC AUC: {test_auc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(
    fpr,
    tpr,
    color="blue",
    lw=2,
    label=f"Logistic Regression (AUC = {test_auc:.4f})",
)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random Guess")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title(
    "Receiver Operating Characteristic (ROC) - Weather Classifier", fontsize=14
)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.savefig(OUTPUT / "weather_roc.png", bbox_inches="tight")
plt.close()

# having an AUC of 0.78 means the model is surprisingly good in predicting good running days, we can see on the classification rreport
# that the model has a moderate false positive rate, with it being a more common error than false negatives. i would personaloly rather the app
# over-recommend running instead of under-recommend, as a runner the challenges are always welcome.
# if using this for a real app i would choose a number close to the 0.78 suggested by the model. 

# Step 5: Save the Model and Metadata

os.makedirs("models", exist_ok=True)

model_path = os.path.join("models", "weather_classifier.pkl")
joblib.dump(best_model, model_path)

metadata = {
    "python_version": platform.python_version(),
    "scikit_learn_version": sklearn.__version__,
    "feature_names": feature_cols,
    "best_hyperparameters": grid_search.best_params_,
    "test_auc": float(test_auc),
    "location": {"city": "Charlotte, NC", "latitude": latitude, "longitude": longitude},
    "label_thresholds_description": threshold_description,
}

metadata_path = os.path.join("models", "weather_classifier_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\nModel successfully saved to '{model_path}'")
print(f"Metadata successfully saved to '{metadata_path}'")