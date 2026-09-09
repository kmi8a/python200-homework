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

# --- Read ---
raw_rows = supabase.table("weather_raw").select("*").execute().data
already_done = {r["date"] for r in supabase.table("weather_enriched").select("date").execute().data}
to_classify = [r for r in raw_rows if r["date"] not in already_done]
print(f"Records to process: {len(to_classify)}")

if not to_classify:
    print("Nothing to do — all records already enriched.")
    exit()

# --- ML Transform ---
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

# --- LLM Transform ---
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

# --- Load ---
db_response = (
    supabase.table("weather_enriched")
    .upsert(enrichment_records, on_conflict="date")
    .execute()
)
print(f"Upserted {len(db_response.data)} rows into weather_enriched")

# --- Spot-check ---
sample = supabase.table("weather_enriched").select("*").limit(3).execute()
for row in sample.data:
    print(f"\n{row['date']} | good={row['good_for_running']} | conf={row['confidence']:.2f}")
    print(f"  {row['llm_summary']}")