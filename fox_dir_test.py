import os

BASE_PATH = os.path.join(os.environ['SLURM_SUBMIT_DIR'],
			    os.environ['SLURM_JOB_ID'])
print(BASE_PATH)

os.makedirs(BASE_PATH, exist_ok=True)

