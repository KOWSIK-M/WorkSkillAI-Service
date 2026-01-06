import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from ..config import *

class DataPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english')
        self.mlb = MultiLabelBinarizer()
        self.common_skills = None

    def load_data(self):
        print("⏳ Loading datasets...")
        # Read postings (only needed columns to save memory)
        # Using encoding='utf-8' and handling mixed types
        # Actual columns: job_id, title, description
        df_jobs = pd.read_csv(POSTINGS_PATH, usecols=['job_id', 'title', 'description'], 
                              dtype=str, on_bad_lines='skip')
        
        # Read skills
        # Actual columns: job_id, skill_abr
        df_skills = pd.read_csv(SKILLS_PATH, dtype={'job_id': str, 'skill_abr': str}, on_bad_lines='skip')
        
        return df_jobs, df_skills

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def preprocess(self):
        df_jobs, df_skills = self.load_data()
        
        print("🔄 Merging and processing...")
        # Merge on job_id
        skill_col = 'skill_abr'
        link_col = 'job_id'
        
        # Group skills by job
        job_skills_grouped = df_skills.groupby(link_col)[skill_col].apply(set).reset_index()
        
        # Merge with jobs
        df = pd.merge(df_jobs, job_skills_grouped, on=link_col, how='inner')
        
        # Rename columns for consistency
        df.rename(columns={'title': 'job_title', skill_col: 'skills'}, inplace=True)

        
        # Drop rows with no skills or no description
        df.dropna(subset=['description', 'skills'], inplace=True)
        
        # Filter top N most common skills
        all_skills = [s for skills in df['skills'] for s in skills]
        from collections import Counter
        skill_counts = Counter(all_skills)
        self.common_skills = set([s for s, _ in skill_counts.most_common(MAX_SKILLS)])
        
        print(f"✅ Filtered to top {MAX_SKILLS} most common skills")
        
        # Filter dataframe to keep only common skills
        df['skills'] = df['skills'].apply(lambda x: [s for s in x if s in self.common_skills])
        df = df[df['skills'].map(len) > 0]
        
        # Text cleaning
        print("🧹 Cleaning text...")
        df['clean_text'] = df['job_title'] + " " + df['description'].apply(self.clean_text)
        
        # Train/Test Split
        print("✂️ Splitting data...")
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            df['clean_text'], df['skills'], test_size=TEST_SIZE, random_state=SEED
        )
        
        # Feature Extraction (TF-IDF)
        print("🔠 TF-IDF Vectorization...")
        X_train = self.vectorizer.fit_transform(X_train_raw)
        X_test = self.vectorizer.transform(X_test_raw)
        
        # Label Binarization
        print("🏷️ Encoding labels...")
        y_train = self.mlb.fit_transform(y_train_raw)
        y_test = self.mlb.transform(y_test_raw)
        
        # Save preprocessors
        os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(RESULTS_DIR, "models", "tfidf.pkl"))
        joblib.dump(self.mlb, os.path.join(RESULTS_DIR, "models", "mlb.pkl"))
        
        print(f"✅ Preprocessing complete. Train shape: {X_train.shape}")
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    dp = DataPreprocessor()
    dp.preprocess()
