import joblib
import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "source", "model", "merged_random_forest_model.pkl")
JOB_PATH = os.path.join(BASE_DIR, "source", "dataset", "jobs.csv")

model = joblib.load(MODEL_PATH)
jobs_file = pd.read_csv(JOB_PATH)