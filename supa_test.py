from supabase import create_client, Client

SUPABASE_URL = "https://hhfmbbuzuygxvjspnqma.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhoZm1iYnV6dXlneHZqc3BucW1hIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM4NjUzNDksImV4cCI6MjA1OTQ0MTM0OX0.2eH6OSuhunXV29mpQBECkGO6uF8N3JddobStXCmdHgo"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Fetch data from a table
response = supabase.table("job_risk").select("*").execute()
response1 = supabase.table("workforce_reskilling_events").select("*").eq("skill_category", "Data Science").execute()
print(response.data)

