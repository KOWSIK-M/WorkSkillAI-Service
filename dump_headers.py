try:
    with open(r"data\linkedin_jobs\postings.csv", "r", encoding="utf-8") as f:
        header_post = f.readline().strip()
    
    with open(r"data\linkedin_jobs\jobs\job_skills.csv", "r", encoding="utf-8") as f:
        header_skills = f.readline().strip()
        
    with open("header_dump.txt", "w", encoding="utf-8") as out:
        out.write(f"POSTINGS: {header_post}\n")
        out.write(f"SKILLS: {header_skills}\n")
        
    print("Dumped headers to header_dump.txt")
except Exception as e:
    with open("header_dump.txt", "w") as out:
        out.write(f"Error: {e}")
