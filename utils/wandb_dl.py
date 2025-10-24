import wandb
import os

# --- CONFIGURATION ---
ENTITY = "AIML_OC"  # e.g. your username or team name
PROJECT = "LLM-Reward-Shaping"  # e.g. "kangaroo-experiments"
DOWNLOAD_DIR = "./models"  # where to save locally
FILE_SUFFIX = "_final.cleanrl_model"

# -----------------------

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
api = wandb.Api()

print(f"Fetching runs from {ENTITY}/{PROJECT}...")

runs = api.runs(f"{ENTITY}/{PROJECT}")

for run in runs:
    run_name = run.name or run.id
    files = run.files()
    for f in files:
        if f.name.endswith(FILE_SUFFIX):
            local_path = os.path.join(DOWNLOAD_DIR, f.name)
            print(f"Downloading {f.name} from {run_name}...")
            f.download(root=DOWNLOAD_DIR, replace=True)
            break

print("\nAll available models have been downloaded.")
