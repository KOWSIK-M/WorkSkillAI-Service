import os
import torch

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Data Paths
RAW_DATA_PATH = os.path.join(DATA_DIR, "linkedin_jobs")
POSTINGS_PATH = os.path.join(RAW_DATA_PATH, "postings.csv")
SKILLS_PATH = os.path.join(RAW_DATA_PATH, "jobs", "job_skills.csv")

# Experiment Config
SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # from train set
MAX_FEATURES = 5000  # TF-IDF max features
MAX_SKILLS = 100     # Top N most common skills to predict (to keep simple)

# Model Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
