import os

BASE_PATH = os.environ['SLURM_SUBMIT_DIR']
print(BASE_PATH)

os.makedirs(BASE_PATH, exist_ok=True)

