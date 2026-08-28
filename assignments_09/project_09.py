import os
from dotenv import load_dotenv
from supabase import create_client
import requests

load_dotenv()  # reads .env and sets environment variables

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)