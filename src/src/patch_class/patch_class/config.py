import importlib
from pathlib import Path
from dotenv import dotenv_values

from patch_class.profiles import PROFILES
from patch_class.augs import ATF

ENV_CONFIG = dotenv_values(Path(Path(__file__).resolve().parent, ".env"))
PROFILE_ID = ENV_CONFIG["PROFILE_ID"]

MODEL_NAME = PROFILES[PROFILE_ID]["MODEL_NAME"]
PRETRAINED = PROFILES[PROFILE_ID]["PRETRAINED"]
NUM_CLASSES = PROFILES[PROFILE_ID]["NUM_CLASSES"]
CONF_DROPOUT_CLASSIFIER = PROFILES[PROFILE_ID]["CONF_DROPOUT_CLASSIFIER"]
MAX_NUM_EPOCHS = PROFILES[PROFILE_ID]["MAX_NUM_EPOCHS"]
BATCH_SIZE = PROFILES[PROFILE_ID]["BATCH_SIZE"]
ACCUM_GRAD_BATCHES = PROFILES[PROFILE_ID]["ACCUM_GRAD_BATCHES"]
USE_AMP = PROFILES[PROFILE_ID]["USE_AMP"]
USE_GRAD_SCALER = PROFILES[PROFILE_ID]["USE_GRAD_SCALER"]
EARLY_STOP_PATIENCE = PROFILES[PROFILE_ID]["EARLY_STOP_PATIENCE"]
OPTIMIZER = PROFILES[PROFILE_ID]["OPTIMIZER"]
OPTIMIZER_PARAMS = PROFILES[PROFILE_ID]["OPTIMIZER_PARAMS"]
SCHEDULER = PROFILES[PROFILE_ID]["SCHEDULER"]
SCHEDULER_PARAMS = PROFILES[PROFILE_ID]["SCHEDULER_PARAMS"]
CRITERTION = PROFILES[PROFILE_ID]["CRITERTION"]
CRITERTION_PARAMS = PROFILES[PROFILE_ID]["CRITERTION_PARAMS"]
VAL_METRIC = PROFILES[PROFILE_ID]["VAL_METRIC"]

INPUT_SIZE = PROFILES[PROFILE_ID]["INPUT_SIZE"]
NUM_WORKERS = PROFILES[PROFILE_ID]["NUM_WORKERS"]

TRAIN_DATA_DIR = PROFILES[PROFILE_ID]["TRAIN_DATA_DIR"]
TEST_DATA_DIR = PROFILES[PROFILE_ID]["TEST_DATA_DIR"]
COLOR_AUG_ARGS = PROFILES[PROFILE_ID]["COLOR_AUG_ARGS"]
NORM_ARGS = PROFILES[PROFILE_ID]["NORM_ARGS"]
CLAMP_VALUES = PROFILES[PROFILE_ID]["CLAMP_VALUES"]

ACCELERATOR = PROFILES[PROFILE_ID]["ACCELERATOR"]
MATMUL_PRECISION = PROFILES[PROFILE_ID]["MATMUL_PRECISION"]
CUDA_DETERMINISTIC = PROFILES[PROFILE_ID]["CUDA_DETERMINISTIC"]
CUDA_BENCHMARK = PROFILES[PROFILE_ID]["CUDA_BENCHMARK"]
INF_BATCH_SIZE = PROFILES[PROFILE_ID]["INF_BATCH_SIZE"]
