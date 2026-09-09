# https://youtu.be/VIsSnb5Ab-I

import os
import json
import pandas as pd
import joblib
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Load model and feature list
clf = joblib.load("models/weather_classifier.pkl")
with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)
FEATURES = metadata["feature_names"]

SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly one sentence — direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)

def make_user_message(row, good_for_running, confidence):
    prediction_text = "good for running" if good_for_running else "not ideal for running"
    return (
        f"Date: {row['date']}\n"
        f"High: {row['temperature_2m_max']}°F, Low: {row['temperature_2m_min']}°F\n"
        f"Precipitation: {row['precipitation_sum']} mm\n"
        f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
        f"Model prediction: {prediction_text} (confidence: {confidence:.0%})"
    )

# --- Step 1: Incremental Read ---
raw_rows = supabase.table("weather_raw").select("*").execute().data
already_done = {r["date"] for r in supabase.table("weather_enriched").select("date").execute().data}
to_classify = [r for r in raw_rows if r["date"] not in already_done]
print(f"Total raw records: {len(raw_rows)}")
print(f"Already enriched: {len(already_done)}")
print(f"Records to process: {len(to_classify)}")

if not to_classify:
    print("Nothing to do — all records already enriched.")
    exit()

# --- Step 2: ML Transform ---
df = pd.DataFrame(to_classify)
X = df[FEATURES]
predictions   = clf.predict(X)
probabilities = clf.predict_proba(X)[:, 1]

enrichment_records = [
    {
        "date":             to_classify[i]["date"],
        "good_for_running": bool(predictions[i]),
        "confidence":       round(float(probabilities[i]), 4),
        "llm_summary":      None,
    }
    for i in range(len(to_classify))
]

good_count = int(predictions.sum())
conf_min = float(probabilities.min())
conf_max = float(probabilities.max())
print(f"Classified good for running: {good_count} of {len(predictions)}")
print(f"Confidence range: {conf_min:.2f} to {conf_max:.2f}")

# --- Step 3: LLM Transform ---
for i, record in enumerate(enrichment_records):
    raw_row = to_classify[i]
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_message(
                    raw_row, record["good_for_running"], record["confidence"]
                )},
            ],
            max_tokens=100,
        )
        summary = response.choices[0].message.content.strip()
        record["llm_summary"] = summary or "Recommendation unavailable."
    except Exception as e:
        print(f"  API error on {record['date']}: {e}")
        record["llm_summary"] = "Recommendation unavailable."

    if (i + 1) % 50 == 0:
        print(f"  Processed {i + 1} / {len(enrichment_records)}")

# --- Step 4: Load ---
db_response = (
    supabase.table("weather_enriched")
    .upsert(enrichment_records, on_conflict="date")
    .execute()
)
print(f"Upserted {len(db_response.data)} rows into weather_enriched")

# --- Step 5: Verify ---

# Total number of rows
all_enriched = supabase.table("weather_enriched").select("date, good_for_running, confidence, llm_summary").execute().data
total_rows = len(all_enriched)
print(f"Total rows in weather_enriched: {total_rows}")

# Number of days classified as good for running
good_count_total = sum(1 for r in all_enriched if r["good_for_running"])
print(f"Total days classified as good for running: {good_count_total}")

# Five sample rows
sample = supabase.table("weather_enriched").select("*").limit(5).execute()
print("\nSample rows:")
for row in sample.data:
    print(f"Date: {row['date']} | Good: {row['good_for_running']} | Conf: {row['confidence']:.2f}")
    print(f"  Summary: {row['llm_summary']}")


# --- Step 6: Reflection ---
# If we load the weather data for a different city, the results would be skewed and the predictive accuracy of
# the model would be compromised because data like temperature, humidity and precipitation learned by the model
# would no longer be valid for the other city.
# On this pipeline the LLM recommendations are purely additive as it only translates the model's classification 
# into natural language, this means that if the model makes a mistake on the classification, the LLM would
# try to rationalize this using natual language, generating coherent but misleading advise.
# When running this pipeline on 50.000 records the main concern would be API latency, a way to solve this would be
# using asynchronous requests and database chunking.