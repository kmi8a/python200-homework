# --- Supabase Connection ---

# Q1
# In a comment block, answer: what are the two pieces of information supabase-py needs to connect to your project?
# Where do you find them in the Supabase dashboard, and why should they never be hardcoded in a Python script?

# Answer
# To connect, supabase-py first needs the endpoint to the database with the corresponding ID which is found 
# in Project Settings under general settings, and the second piece needed is the API key, this one is found 
# under the API keys tab. The API keys should never be hardcoded in a python script for security reasons, 
# for example when uploading the script to github the key would be then visible to anyone, scrapped in seconds,
# and the security of the database compromised.

# Q2
# Write a function get_client() that:
# 1. Loads your credentials from environment variables using python-dotenv
# 2. Creates and returns a Supabase client
# The function should raise a clear error if either environment variable is missing.

import os
from dotenv import load_dotenv
from supabase import create_client

def get_client():
    load_dotenv()
    SUPABASE_URL = os.environ["SUPABASE_URL"]
    SUPABASE_KEY = os.environ["SUPABASE_KEY"]
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase

supabase = get_client()

# Q3
# In a comment block, answer: what is Row Level Security (RLS), and why did you disable it on your tables
# for this course? In what kind of real-world application would you want to keep it enabled?

# Answer:
# Row Level Security is a feature of databases that restricts which rows a user can view, insert, update or 
# delete based on security policies and runtime context, we disabled it on this course so we can use the 
# Anon key to modify the tables withouth he need for authentication, any project that has multiple users 
# would benefit from implementing RLS, a banking app for example is a good example of this.

# --- supabase-py CRUD ---

# Q1

# Answer:

def insert_record(supabase):
    record = {
        "date":               "2026-08-28",
        "temperature_2m_max": 28.4,
        "temperature_2m_min": 17.2,
        "precipitation_sum":  0.0,
        "wind_speed_10m_max": 12.1,
    }
    supabase.table("weather_raw").insert(record).execute()
    return (f"Inserted.")

insert_record(supabase)

# Q2
# Write a function get_records_by_date_range(supabase, start, end) that returns all rows from weather_raw
# where date >= start and date <= end. The function should return the list of row dictionaries.
# Test it with a date range that includes the row you inserted in Q1 and print the result.

# Answer:
def get_records_by_date_range(supabase, start, end):
    response = supabase.table("weather_raw").select("*").gte("date", start).lte("date", end).execute()
    return response.data

filtered_records = get_records_by_date_range(supabase, "2026-08-01", "2026-08-31")
print(filtered_records)

# Q3
# In a comment block, explain the difference between insert and upsert in supabase-py.
# Give a concrete example of when you would choose each. 
# Then write a function safe_upsert(supabase, records) that upserts a list of records into weather_raw 
# using date as the conflict key and prints the number of rows affected.

# Answer:
# Insert will try to save the data to the database  adding a new row, but if there's a conflicting data like 
# a duplicate primary key, then the data won't be saved, the operation will fail raising an error.
# Upsert instead will try to update the data if there's conflicting rows and otherwise write the data by 
# inserting a new row.
# Insert is helpful when logging financial transactions, in this case any duplicate indicates a system bug,
# while Upsert is very useful for a data pipelines like the weather data pipeline we are working on, as by 
# running the script multiple times will fetch records that are already present in the database and update
# the ones that are not.

def safe_upsert(supabase, records):
    response = supabase.table("weather_raw").upsert(records, on_conflict="date").execute()
    rows_affected = len(response.data) if response.data else 0
    print(f'Number of rows affected: {rows_affected}')
    return response.data

to_update = {
    "date":               "2026-08-28",
    "temperature_2m_max": 32.4,
    "temperature_2m_min": 15.2,
    "precipitation_sum":  0.0,
    "wind_speed_10m_max": 5.0,
}

safe_upsert(supabase, to_update)

# --- Idempotency ---

# Q1
# "Idempotency" means that running an operation multiple times produces the same result as running it once.
# In a comment block, explain why idempotency matters for a data pipeline. Give one concrete example of what 
# goes wrong in a non-idempotent pipeline when the script crashes halfway through and is restarted.

# Answer:
# Idempotency is very important in data pipelines as it guarantees that network failures, timeouts, or 
# the script crashing will not corrupt data or create duplicate records when the script is re-run.
# Using as example a weather data pipeline that fetches a month of daily records using standard insert
# instead of upsert, If the script inserts days 1 to 15 but then crashes before finishing the whole month,
# on this case re-running the script will cause a the pipeline to re-insert days 1 through 15 a second time,
# This will then create duplicate rows for every date, corrupting downstream data calculations (like skewing
# average temperature metrics) and forcing manual database cleanup.