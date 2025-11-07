"""
PatrolIQ Unified Pipeline
Runs full preprocessing → clustering analysis → dimensionality reduction for Streamlit app.
"""

import os
import subprocess
from datetime import datetime

# ===== CONFIG =====
OUTPUT_DIR = "outputs"
LOG_DIR = OUTPUT_DIR
os.makedirs(LOG_DIR, exist_ok=True)

STEPS = [
    ("Preprocessing", "app/preprocess_fixed.py", "outputs/preprocess_run.log"),
    ("Clustering", "app/clustering_analysis.py", "outputs/clustering_run.log"),
    ("Dimensionality Reduction", "app/dimred_sample.py", "outputs/dimred_run.log"),
]

def run_step(name, script, log_path):
    print(f"\n🟦 STEP: {name}")
    print(f"   → Running: {script}")
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            ["python", "-u", script],
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
        proc.wait()

    # show summary from the log
    print(f"    Completed {name} — log saved: {log_path}")
    print("-" * 60)

def main():
    print(f"\n Starting PatrolIQ pipeline at {datetime.now()}")
    print("=" * 60)
    
    for name, script, log in STEPS:
        run_step(name, script, log)

    print(f"\n Pipeline completed successfully at {datetime.now()}")
    print(f" Check all results in: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
