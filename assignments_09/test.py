import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()  # reads .env and sets environment variables

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# connection test
# response = supabase.table("connection_test").select("*").execute()
# print(response.data)

# Insert a test record
record = {
    "date":               "2023-06-15",
    "temperature_2m_max": 28.4,
    "temperature_2m_min": 17.2,
    "precipitation_sum":  0.0,
    "wind_speed_10m_max": 12.1,
}
supabase.table("weather_raw").insert(record).execute()
print("Inserted.")

# Read it back
response = supabase.table("weather_raw").select("*").eq("date", "2023-06-15").execute()
print("Retrieved:", response.data)

# Upsert (update the max temp — same date, different value)
updated = {**record, "temperature_2m_max": 29.0}
supabase.table("weather_raw").upsert(updated, on_conflict="date").execute()

# Confirm the update
response = supabase.table("weather_raw").select("*").eq("date", "2023-06-15").execute()
print("After upsert:", response.data[0]["temperature_2m_max"])  # 29.0

# Clean up
supabase.table("weather_raw").delete().eq("date", "2023-06-15").execute()
print("Deleted.")