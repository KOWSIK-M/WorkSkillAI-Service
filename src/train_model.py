import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Paths
DATA_PATH = "data/IT_Job_Roles_Skills.csv"
MODEL_PATH = os.path.join("models", "skill_gap_model.pkl")
ENCODER_PATH = os.path.join("models", "skills_encoder.pkl")

# Load dataset
df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")

# Ensure Skills column is a list
df['Skills'] = df['Skills'].apply(lambda x: [s.strip() for s in x.split(',')])

# Filter rare skills (optional, helps model focus on common skills)
from collections import Counter
skill_counts = Counter([skill for sublist in df['Skills'] for skill in sublist])
common_skills = [s for s, c in skill_counts.items() if c >= 2]  # appear at least twice
df['Skills'] = df['Skills'].apply(lambda x: [s for s in x if s in common_skills])

# Combine text features
df['features'] = df['Job Title'] + " " + df['Job Description'] + " " + df['Certifications']

# Vectorize features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['features'])

# Encode target skills
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['Skills'])

# Train multi-label classifier
clf = OneVsRestClassifier(LogisticRegression(max_iter=500))
clf.fit(X, y)

# Save model and encoder
os.makedirs("models", exist_ok=True)
joblib.dump({"model": clf, "vectorizer": vectorizer}, MODEL_PATH)
joblib.dump(mlb, ENCODER_PATH)

print("✅ Model trained & saved successfully!")
