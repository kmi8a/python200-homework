import os
import json
import requests
import joblib
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
from prefect import flow, task

LATITUDE  = 35.23
LONGITUDE = -80.84

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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

@task(retries=2, retry_delay_seconds=10)
def extract() -> list:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  LATITUDE,
        "longitude": LONGITUDE,
        "start_date": "2023-01-01",
        "end_date":   "2023-12-31",
        "daily": FEATURES,
        "timezone": "America/New_York",
        "temperature_unit": "fahrenheit",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    daily = response.json()["daily"]

    records = [
        {
            "date":               daily["time"][i],
            "temperature_2m_max": daily["temperature_2m_max"][i],
            "temperature_2m_min": daily["temperature_2m_min"][i],
            "precipitation_sum":  daily["precipitation_sum"][i],
            "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
        }
        for i in range(len(daily["time"]))
    ]

    print(f"Extracted {len(records)} daily records from Open-Meteo")

    return records


@task
def load_raw(records: list) -> None:
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )

    print(f"Upserted {len(response.data)} rows into weather_raw\n")

@task
def transform(raw_records: list) -> list:
    # --- Incremental check ---
    already_done = {r["date"] for r in supabase.table("weather_enriched").select("date").execute().data}
    to_classify = [r for r in raw_records if r["date"] not in already_done]

    print(f"Records to transform: {len(to_classify)} (skipping {len(already_done)} already enriched)")

    if not to_classify:
        print("Nothing to do — all records already enriched.")
        return []

    # --- ML Classify ---
    clf = joblib.load("models/weather_classifier.pkl")
    df = pd.DataFrame(to_classify)
    X = df[FEATURES]

    predictions   = clf.predict(X)
    probabilities = clf.predict_proba(X)[:, 1]

    print(f"ML classification complete. Good days: {int(predictions.sum())} / {len(predictions)}")

    enrichment_records = [
        {
            "date":             to_classify[i]["date"],
            "good_for_running": bool(predictions[i]),
            "confidence":       round(float(probabilities[i]), 4),
            "llm_summary":      None,
        }
        for i in range(len(to_classify))
    ]

    # --- LLM Enrich ---
    for i, record in enumerate(enrichment_records):
        raw_row = to_classify[i]
        prediction_text = "good for running" if record["good_for_running"] else "not ideal for running"
        user_message = (
            f"Date: {raw_row['date']}\n"
            f"High: {raw_row['temperature_2m_max']}°C, Low: {raw_row['temperature_2m_min']}°C\n"
            f"Precipitation: {raw_row['precipitation_sum']} mm\n"
            f"Max wind speed: {raw_row['wind_speed_10m_max']} km/h\n"
            f"Model prediction: {prediction_text} (confidence: {record['confidence']:.0%})"
        )
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=100,
            )
            summary = response.choices[0].message.content.strip()
            record["llm_summary"] = summary or "Recommendation unavailable."
        except Exception as e:
            print(f"  API error on {record['date']}: {e}")
            record["llm_summary"] = "Recommendation unavailable."

        if (i + 1) % 50 == 0:
            print(f"  LLM enriched {i + 1} / {len(enrichment_records)} records")

    print(f"Transform complete: {len(enrichment_records)} records enriched")
    return enrichment_records

@task(retries=2, retry_delay_seconds=5)
def load_enriched(enrichment_records: list) -> None:
    if not enrichment_records:
        print("No new enrichment records to load.")
        return

    response = (
        supabase.table("weather_enriched")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )
    print(f"Upserted {len(response.data)} rows into weather_enriched")

@flow(log_prints=True)
def etl_pipeline():
    raw_records        = extract()
    load_raw(raw_records)
    enrichment_records = transform(raw_records)
    load_enriched(enrichment_records)
    print("Pipeline copmlete.")


if __name__ == "__main__":
    etl_pipeline()