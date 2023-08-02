import pathlib

# General project directories
DIR_PROJECT_ROOT = pathlib.Path.cwd().parent
DIR_PROJECT_DATA = DIR_PROJECT_ROOT / "data"
DIR_PROJECT_CACHE = DIR_PROJECT_ROOT / ".cache"

####### Raw data manager module #######

URL_3W_REPO = "https://github.com/petrobras/3W.git"
DIR_DOWNLOADED_REPO = DIR_PROJECT_DATA / "3w_repo"
DIR_RAW_DATASET = DIR_DOWNLOADED_REPO / "dataset"

DIR_CONVERTED_DATASET = DIR_PROJECT_DATA / "dataset_converted"
PATH_DATA_INSPECTOR_CACHE = DIR_PROJECT_CACHE / "data_inspector_cache.parquet"

############# Pipeline module #############

PIPELINE_NAME = "pipeline_lemi_3w"
PIPELINE_ROOT = DIR_PROJECT_DATA / "pipeline_lemi_3w_output"
METADATA_PATH = DIR_PROJECT_DATA / "tfx_metadata" / "metadata.db"
ENABLE_CACHE = True
