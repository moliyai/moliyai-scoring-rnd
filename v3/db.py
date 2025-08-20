import joblib
import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "source", "model", "merged_random_forest_model.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "source", "model", "6month_random_forest_model.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "source", "model", "label_encoders.pkl")
JOB_PATH = os.path.join(BASE_DIR, "source", "dataset", "jobs.csv")

label_encoders = joblib.load(LABEL_ENCODER_PATH)
model = joblib.load(MODEL_PATH)
jobs_file = pd.read_csv(JOB_PATH)