import pandas as pd

try:
    print("Reading postings.csv columns...")
    df_post = pd.read_csv(r"data\linkedin_jobs\postings.csv", nrows=0)
    print(f"POSTINGS COLUMNS: {list(df_post.columns)}")
    
    print("\nReading job_skills.csv columns...")
    df_skills = pd.read_csv(r"data\linkedin_jobs\jobs\job_skills.csv", nrows=0)
    print(f"SKILLS COLUMNS: {list(df_skills.columns)}")
except Exception as e:
    print(f"Error: {e}")
