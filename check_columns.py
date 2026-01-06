import pandas as pd

# Analyze the LinkedIn dataset files
try:
    print("Loading postings.csv (first 5 rows)...")
    df_postings = pd.read_csv('data/linkedin_jobs/postings.csv', nrows=5)
    print(f"Postings Columns: {df_postings.columns.tolist()}")

    print("\nLoading jobs/job_skills.csv (first 5 rows)...")
    df_skills = pd.read_csv('data/linkedin_jobs/jobs/job_skills.csv', nrows=5)
    print(f"Skills Columns: {df_skills.columns.tolist()}")
    
except Exception as e:
    print(f"Error: {e}")
