from fastapi.testclient import TestClient
from src.predict_gap import app

client = TestClient(app)

def test_predict_gap():
    response = client.post("/predict-skill-gap", json={
        "job_role": "Data Scientist",
        "skills": ["Python", "SQL"]
    })
    assert response.status_code == 200
    result = response.json()
    assert "missing_skills" in result
