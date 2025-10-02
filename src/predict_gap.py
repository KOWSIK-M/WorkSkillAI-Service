from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import os
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Paths
MODEL_PATH = os.path.join("models", "skill_gap_model.pkl")
ENCODER_PATH = os.path.join("models", "skills_encoder.pkl")

# Load model + encoder
saved = joblib.load(MODEL_PATH)
model = saved["model"]
vectorizer = saved["vectorizer"]
mlb = joblib.load(ENCODER_PATH)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudentProfile(BaseModel):
    job_role: str
    skills: List[str]
    certifications: List[str] = []
    description: str = ""  # optional self-description


@app.post("/predict-skill-gap")
def predict_gap(profile: StudentProfile):
    # Combine input into a single feature string
    features = profile.job_role + " " + profile.description + " " + " ".join(profile.certifications)
    X = vectorizer.transform([features])

    # Predict skill probabilities
    probs = model.predict_proba(X)[0]
    required_skills = [mlb.classes_[i] for i, p in enumerate(probs) if p > 0.2]

    # Calculate missing skills
    missing = [s for s in required_skills if s not in profile.skills]

    return {
        "job_role": profile.job_role,
        "required_skills": required_skills,
        "current_skills": profile.skills,
        "missing_skills": missing,
        "gap_score": round(len(missing) / len(required_skills), 2) if required_skills else 0
    }
