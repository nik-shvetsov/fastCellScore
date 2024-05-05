from pathlib import Path
from dotenv import dotenv_values

from hover_quant.profiles import PROFILES
from hover_quant.augs import ATF

ENV_CONFIG = dotenv_values(Path(Path(__file__).resolve().parent, ".env"))
PROFILE_ID = ENV_CONFIG["PROFILE_ID"]

MODEL_SAVE_DIR = Path("models")
RAW_DATA_DIR = Path("data")
if "monusac" in PROFILE_ID:
    DATA_DIR = Path(RAW_DATA_DIR, "monusac")
elif "pannuke" in PROFILE_ID:
    DATA_DIR = Path(RAW_DATA_DIR, "pannuke")

INPUT_SIZE = PROFILES[PROFILE_ID]["INPUT_SIZE"]
NUM_WORKERS = PROFILES[PROFILE_ID]["NUM_WORKERS"]
NUM_CLASSES = PROFILES[PROFILE_ID]["NUM_CLASSES"]
BATCH_SIZE = PROFILES[PROFILE_ID]["BATCH_SIZE"]
ACCUM_GRAD_BATCHES = PROFILES[PROFILE_ID]["ACCUM_GRAD_BATCHES"]

MAX_NUM_EPOCHS = PROFILES[PROFILE_ID]["MAX_NUM_EPOCHS"]
EARLY_STOP_PATIENCE = PROFILES[PROFILE_ID]["EARLY_STOP_PATIENCE"]

USE_AMP = PROFILES[PROFILE_ID]["USE_AMP"]
USE_GRAD_SCALER = PROFILES[PROFILE_ID]["USE_GRAD_SCALER"]

CRITERION = PROFILES[PROFILE_ID]["CRITERION"]
OPTIMIZER = PROFILES[PROFILE_ID]["OPTIMIZER"]
OPTIMIZER_PARAMS = PROFILES[PROFILE_ID]["OPTIMIZER_PARAMS"]
SCHEDULER = PROFILES[PROFILE_ID]["SCHEDULER"]
SCHEDULER_PARAMS = PROFILES[PROFILE_ID]["SCHEDULER_PARAMS"]
EVAL_BATCH_SIZE = PROFILES[PROFILE_ID]["EVAL_BATCH_SIZE"]
INF_BATCH_SIZE = PROFILES[PROFILE_ID]["INF_BATCH_SIZE"]

MULTI_GPU = PROFILES[PROFILE_ID]["MULTI_GPU"]
ACCELERATOR = PROFILES[PROFILE_ID]["ACCELERATOR"]
LABELS = PROFILES[PROFILE_ID]["LABELS"]