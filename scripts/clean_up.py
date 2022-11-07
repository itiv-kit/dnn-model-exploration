import shutil

def clean() -> None:
    """
    Cleans the output directories by deleting their contents.
    """
    shutil.rmtree(LOG_DIR)
    shutil.rmtree(RESULTS_DIR)