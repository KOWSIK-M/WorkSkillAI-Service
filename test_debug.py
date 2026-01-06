import sys
import os
import pandas as pd

print("✅ Script started")

try:
    print("Testing config import...")
    import ml_research.config
    print(f"✅ Config imported. POSTINGS_PATH: {ml_research.config.POSTINGS_PATH}")
    
    path = ml_research.config.POSTINGS_PATH
    if os.path.exists(path):
        print("✅ File exists")
        print(f"File size: {os.path.getsize(path) / (1024*1024):.2f} MB")
    else:
        print(f"❌ File NOT found at {path}")

    print("Testing pandas read (first 5 rows)...")
    df = pd.read_csv(path, nrows=5)
    print("✅ Read success")
    print(df.columns.tolist())

except Exception as e:
    print(f"❌ Error: {e}")
