import json
import os
import joblib
import pandas as pd

# Task 1: Load and Verify

model_path = os.path.join("models", "weather_classifier.pkl")
metadata_path = os.path.join("models", "weather_classifier_metadata.json")

if not os.path.exists(model_path) or not os.path.exists(metadata_path):
    raise FileNotFoundError(
        "Model or metadata file not found. Please run 'train_weather_classifier.py' first."
    )

model = joblib.load(model_path)

with open(metadata_path, "r") as f:
    metadata = json.load(f)

print("--- Model Metadata Loaded Successfully ---")
print(f"City: {metadata['location']['city']}")
print(f"Latitude / Longitude: {metadata['location']['latitude']}, {metadata['location']['longitude']}")
print(f"Features: {metadata['feature_names']}")
print(f"Test ROC AUC: {metadata['test_auc']:.4f}")
print(f"Python Version: {metadata['python_version']}")
print(f"scikit-learn Version: {metadata['scikit_learn_version']}")
print(f"Thresholds: {metadata['label_thresholds_description']}")

# Task 2: Predict on New Data

feature_cols = metadata["feature_names"]

hypothetical_days = pd.DataFrame(
    [
        [72.0, 52.0, 0.0, 8.5],    # Day 1: Good day (pleasant temp, no rain, light wind)
        [38.0, 25.0, 5.2, 35.0],   # Day 2: Bad day (freezing/cold, heavy rain, high winds)
        [46.0, 33.0, 2.5, 15.0],   # Day 3: Borderline (Cold) (max temp right near the lower bound, light rain)
        [85.0, 68.0, 0.0, 6.2],    # Day 4: Good day (warm, clear skies, low wind)
        [55.0, 42.0, 1.8, 22.0],   # Day 5: Borderline/Bad day (cool, light rain, moderately windy)
    ],
    columns=feature_cols,
)

predictions = model.predict(hypothetical_days)
probabilities = model.predict_proba(hypothetical_days)[:, 1]

hypothetical_days["predicted_label"] = [
    "good" if p == 1 else "skip" for p in predictions
]
hypothetical_days["confidence"] = probabilities

print("\nPredictions for Hypothetical Weather Days")
for i, row in hypothetical_days.iterrows():
    print(f"\nDay {i + 1}:")
    print(f"  Inputs:")
    for col in feature_cols:
        print(f"    - {col}: {row[col]}")
    print(f"  Predicted Label : {row['predicted_label'].upper()}")
    print(f"  Confidence (Good): {row['confidence']:.4f}")


# The probability for the borderline day included was 0.4138, the model classifieds this day as a SKIP, but with low confidence. A day with 0.52 would be qualified as a GOOD day.
# 
# Running `predict_weather.py` before `train_weather_classifier.py`, would throw a `FileNotFoundError` when trying to open `models/weather_classifier.pkl` and `models/weather_classifier_metadata.json`.
# To make the error helpful and prevent this from happening, I wrapped the file-loading logic in a try-except block.
#
# To adapt this script to run daily in production for tomorrow's weather, we would have to:
#   1. Live request to Open-Meteo's Daily Forecast API endpoint, querying tomorrow's weather variables.
#   2. Parse tomorrow's projected values directly into a dataFrame structured with the exact same column order defined in `feature_names`.
#   3. write the prediction and confidence score to a database and then push a notification to the user.
#   4. Run the script automatically every day.