import pathlib

URL_3W_REPO = "https://github.com/petrobras/3W.git"

DIR_PROJECT_ROOT = pathlib.Path.cwd().parent
DIR_PROJECT_DATA = DIR_PROJECT_ROOT / "data"

DIR_DOWNLOADED_REPO = DIR_PROJECT_DATA / "3w_repo"
DIR_RAW_DATASET = DIR_DOWNLOADED_REPO / "dataset"

DIR_CONVERTED_DATASET = DIR_PROJECT_DATA / "dataset_converted"
