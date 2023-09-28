import pathlib

############# General project directories #############
DIR_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DIR_PROJECT_DATA = DIR_PROJECT_ROOT / "data"
DIR_PROJECT_CACHE = DIR_PROJECT_ROOT / ".cache"

DIR_SAVED_OBJECTS = DIR_PROJECT_CACHE / "saved_objects"


############# Raw data manager module #############
URL_3W_REPO = "https://github.com/petrobras/3W.git"
URL_3W_DATASET_CONFIG_FILE = (
    "https://raw.githubusercontent.com/petrobras/3W/main/dataset/dataset.ini"
)
DIR_DOWNLOADED_REPO = DIR_PROJECT_DATA / "3w_repo"
DIR_RAW_DATASET = DIR_DOWNLOADED_REPO / "dataset"

DIR_CONVERTED_PREFIX = "dataset_converted_v"
PATH_DATA_INSPECTOR_CACHE = DIR_PROJECT_CACHE / "data_inspector_cache.parquet"
