import pathlib

# General project directories
DIR_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DIR_PROJECT_DATA = DIR_PROJECT_ROOT / "data"
DIR_PROJECT_CACHE = DIR_PROJECT_ROOT / ".cache"

DIR_SAVED_OBJECTS = DIR_PROJECT_CACHE / "saved_objects"

####### Raw data manager module #######

URL_3W_REPO = "https://github.com/petrobras/3W.git"
URL_3W_DATASET_CONFIG_FILE = (
    "https://raw.githubusercontent.com/petrobras/3W/main/dataset/dataset.ini"
)
DIR_DOWNLOADED_REPO = DIR_PROJECT_DATA / "3w_repo"
DIR_RAW_DATASET = DIR_DOWNLOADED_REPO / "dataset"

DIR_CONVERTED_PREFIX = "dataset_converted_v"
PATH_DATA_INSPECTOR_CACHE = DIR_PROJECT_CACHE / "data_inspector_cache.parquet"

DIR_CONVERTED_TEST_PREFIX = "test_dataset_converted_v"

############# Pipeline module #############
ENABLE_CACHE = True

PIPELINE_ROOT = DIR_PROJECT_DATA / "pipeline"
METADATA_ROOT = DIR_PROJECT_DATA / "metadata"
SERVING_ROOT = DIR_PROJECT_DATA / "serving_model"

# Schema validaion pipeline
SCHEMA_PIPELINE_NAME = "pipeline_lemi_3w_schema"
SCHEMA_PIPELINE_ROOT = PIPELINE_ROOT / SCHEMA_PIPELINE_NAME
SCHEMA_METADATA_PATH = METADATA_ROOT / SCHEMA_PIPELINE_NAME / "metadata.db"

# Model generation pipeline
MODEL_PIPELINE_NAME = "pipeline_lemi_3w_model"
MODEL_PIPELINE_ROOT = PIPELINE_ROOT / MODEL_PIPELINE_NAME
MODEL_PIPELINE_METADATA_PATH = METADATA_ROOT / MODEL_PIPELINE_NAME / "metadata.db"
MODEL_PIPELINE_SERVING_DIR = SERVING_ROOT / MODEL_PIPELINE_NAME


############# Exploration module #############
CACHE_NAME_TRAIN_MEAN_STD_DEV = "train-split"
