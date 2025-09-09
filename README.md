```markdown
# 📌 Skill Gap Prediction Service

This project provides a **REST API** to predict missing skills and skill gaps for different job roles using a dataset of job descriptions, required skills, and certifications.

---

## 📂 Project Structure
```

skill-gap-service/
│── data/
│ └── job_roles.csv # Your dataset (Job Roles → Skills, Certs, Description)
│
│── models/
│ └── skill_gap_model.pkl # Saved ML model (created after training)
│
│── app/
│ ├── main.py # FastAPI app (service layer)
│ ├── train_model.py # Training script
│ ├── predict.py # Prediction logic
│ ├── utils.py # Helper functions (matching, preprocessing, etc.)
│
│── requirements.txt
│── README.md

````

---

## ⚙️ Setup Instructions

### 1️⃣ Clone & Move to Project
```bash
git clone <your_repo_url> skill-gap-service
cd skill-gap-service
````

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Your Dataset

Place your dataset in the `data/` folder as **`job_roles.csv`**.
Format should be:

```csv
job_role,description,skills,certifications
Full Stack Developer,Develops both front-end..., "JavaScript,HTML,CSS,Node.js,...", "Full Stack Web Dev Bootcamp"
Full Stack JAVA Developer/Programmer/Engineer,Designs and develops..., "Java,Spring,Hibernate,SQL,...", "Oracle Java Certification"
Full Stack Python Developer/Programmer/Engineer,Creates full-stack apps..., "Python,Django,Flask,REST APIs,...", "Python Institute Certifications"
```

---

## 🏋️ Training the Model

Run:

```bash
python app/train_model.py
```

👉 This will:

- Preprocess dataset
- Train a multi-label classification model (RandomForest inside OneVsRest)
- Save it as `models/skill_gap_model.pkl`

---

## 🚀 Running the API

Start FastAPI server:

```bash
uvicorn app.main:app --reload
```

API runs at:
👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

Docs available at:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📡 Example API Call

### Request (JSON):

```json
POST /predict-skill-gap
{
  "job_role": "Full Stack Java Developer",
  "skills": ["PowerBI", "Java"],
  "certifications": ["AWS Certified Big Data – Specialty"],
  "description": "I have experience in PowerBI and Data Visualization"
}
```

### Response (JSON):

```json
{
  "job_role": "Full Stack Java Developer",
  "required_skills": ["Java", "Spring", "Hibernate", "SQL", "React", "..."],
  "current_skills": ["PowerBI", "Java"],
  "missing_skills": ["Spring", "Hibernate", "SQL", "React", "..."],
  "gap_score": 0.06
}
```

---

## 🛠️ Requirements

File: `requirements.txt`

```
fastapi
uvicorn
pandas
scikit-learn
joblib
rapidfuzz
```

---

## 🧪 Quick Test with cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict-skill-gap" \
-H "Content-Type: application/json" \
-d '{
  "job_role": "Full Stack Java Developer",
  "skills": ["PowerBI", "Java"],
  "certifications": ["AWS Certified Big Data – Specialty"],
  "description": "I have experience in PowerBI and Data Visualization"
}'
```

---

✅ That’s it! You now have a **Skill Gap Prediction API** powered by FastAPI + ML.
