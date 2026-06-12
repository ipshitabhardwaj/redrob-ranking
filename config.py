# config.py

import os

# --- Base Paths ---
# Each team member should set their own DATA_DIR
# Either set env variable: set REDROB_DATA=C:\your\path\to\data
# Or just change DATA_DIR below to your local path

DATA_DIR = os.environ.get(
    "REDROB_DATA",
    os.path.join(os.path.dirname(__file__), "data")
)

DATA_PATH = os.path.join(DATA_DIR, "candidates.jsonl")
JD_PATH   = os.path.join(DATA_DIR, "job_description.docx")

# --- Output ---
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "submission", "team_submission.csv"
)

# --- Pipeline Settings ---
RETRIEVAL_K = 500

# --- Scoring Weights ---
WEIGHTS = {
    "semantic":             0.30,
    "response_rate":        0.10,
    "open_to_work":         0.10,
    "recency":              0.10,
    "github_activity":      0.08,
    "profile_completeness": 0.05,
    "notice_period":        0.07,
    "production_ml":        0.10,
    "location":             0.05,
    "startup_fit":          0.05,
}