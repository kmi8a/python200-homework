import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
import joblib
import json

clf = joblib.load("models/weather_classifier.pkl")

with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)

FEATURES = metadata["feature_names"]
# ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'wind_speed_10m_max']

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

response = supabase.table("weather_raw").select("*").execute()
raw_rows = response.data
print(f"Fetched {len(raw_rows)} rows from weather_raw")

enriched_response = supabase.table("weather_enriched").select("date").execute()
already_done = {row["date"] for row in enriched_response.data}

to_classify = [row for row in raw_rows if row["date"] not in already_done]
print(f"Records to classify: {len(to_classify)} (skipping {len(already_done)} already enriched)")

df = pd.DataFrame(to_classify)
X = df[FEATURES]  # select only the feature columns, in the right order

predictions  = clf.predict(X)          # array of 0s and 1s
probabilities = clf.predict_proba(X)[:, 1]  # probability of class 1 (good for running)

print(f"Good days predicted: {predictions.sum()} / {len(predictions)}")
print(f"Confidence range: {probabilities.min():.2f} – {probabilities.max():.2f}")

enrichment_records = []
for i, row in enumerate(to_classify):
    enrichment_records.append({
        "date":             row["date"],
        "good_for_running": bool(predictions[i]),
        "confidence":       round(float(probabilities[i]), 4),
        # llm_summary will be added in the next lesson
    })

print("Sample enrichment records:")
for r in enrichment_records[:3]:
    print(r)

good_days = [r for r in enrichment_records if r["good_for_running"]]
skip_days = [r for r in enrichment_records if not r["good_for_running"]]

print(f"Good days: {len(good_days)} ({len(good_days)/len(enrichment_records):.0%})")
print(f"Skip days: {len(skip_days)}")

# Show a few high-confidence and borderline predictions
enrichment_records.sort(key=lambda r: r["confidence"], reverse=True)
print("\nHighest confidence (good for running):")
for r in enrichment_records[:3]:
    print(f"  {r['date']}: {r['confidence']:.3f}")

enrichment_records.sort(key=lambda r: abs(r["confidence"] - 0.5))
print("\nMost borderline (closest to 0.5 confidence):")
for r in enrichment_records[:3]:
    print(f"  {r['date']}: {r['confidence']:.3f}")