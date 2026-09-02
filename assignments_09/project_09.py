import os
from dotenv import load_dotenv
from supabase import create_client
import requests

load_dotenv()  # reads .env and sets environment variables

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

LATITUDE  = 35.23
LONGITUDE = -80.84

# Step 1: Extract

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude":  LATITUDE,
    "longitude": LONGITUDE,
    "start_date": "2023-01-01",
    "end_date":   "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params)
response.raise_for_status()
data = response.json()

# Step 2: Transform

daily = data["daily"]

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

print("---- Step 2: Transform ----")
print(f"Prepared {len(records)} records")
print("First record:", records[0])
print("Last record:", records[-1])

# Question:
# how many records do you expect for a full year, and how many did you get? 
# If the numbers differ, what might explain the discrepancy?
# Answer:
# For a full year I expect 365 records (one per day), but we got 366 records.
# After investigation, I found out that the extra row was pre-existing in 
# the database from a previous script (warmup_09.py), bringing the total to 366.


# Missing Values

records = [r for r in records if all(v is not None for v in r.values())]
print(f"\nRecords after dropping nulls: {len(records)}\n")

# Step 3: Load

response = (
    supabase.table("weather_raw")
    .upsert(records, on_conflict="date")
    .execute()
)

print("---- Step 3: Load ----")
print(f"Upserted {len(response.data)} rows into weather_raw")

# Question:
# Run the script a second time and confirm the row count in weather_raw does not change. 
# Add a comment: what does this tell you about idempotency?
# Answer:
# This confirms the Supabase .upsert(records, on_conflict="date") operation is idempotent. 
# Because the date column acts as a unique conflict target, repeating the load updates existing 
# rows in-place rather than duplicating them, ensuring safe and repeatable data pipelines.

# Step 4: Verify

count_response = supabase.table("weather_raw").select("date", count="exact").execute()
print("---- Step 4: Verify ----")
print(f"Rows in weather_raw: {count_response.count}")


first = supabase.table("weather_raw").select("*").eq("date", "2023-01-01").execute()
last  = supabase.table("weather_raw").select("*").eq("date", "2023-12-31").execute()
print("First record:", first.data)
print("Last record: ", last.data)

july_fourth = supabase.table("weather_raw").select("*").eq("date", "2023-07-04").execute()

if july_fourth.data:
    print("Row for 2023-07-04:", july_fourth.data[0])
else:
    print("2023-07-04 not found. Finding nearest date...")
    nearest_before = supabase.table("weather_raw").select("*").lte("date", "2023-07-04").order("date", desc=True).limit(1).execute()
    nearest_after = supabase.table("weather_raw").select("*").gte("date", "2023-07-04").order("date", desc=False).limit(1).execute()
    
    nearest_row = nearest_before.data if nearest_before.data else nearest_after.data
    print("Nearest record:", nearest_row[0] if nearest_row else "No records found")

response = (
    supabase.table("weather_raw")
    .select("date")
    .lt("date", "2023-01-01")
    .execute()
)