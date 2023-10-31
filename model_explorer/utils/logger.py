import os
import logging


"""Setup logging for the entire module at once. Logging goes both to a file and
to console. The latter can be changed with the verbose flag
"""

LOGGER_NAME = "exploration_logger"
LOG_DIR = "results"

logger = logging.getLogger(LOGGER_NAME)
logger.propagate = False
logger.setLevel(logging.DEBUG)

# log to console
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.propagate = False

formatter_console = logging.Formatter(
    "[%(asctime)s] %(levelname)-8s - %(message)s", "%m-%d %H:%M:%S"
)

ch.setFormatter(formatter_console)
logger.addHandler(ch)


def set_console_logger_level(level: int):
    lg = logging.getLogger(LOGGER_NAME)
    lg.handlers[0].setLevel(level)  # console handler


def set_logger_filename(fn: str = 'exploration_run.log', clear_log: bool = False):
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    fh = logging.FileHandler(os.path.join(LOG_DIR, fn), mode="w" if clear_log else "a")
    fh.setLevel(logging.DEBUG)
    formatter_file = logging.Formatter(
        "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", "%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter_file)

    lg = logging.getLogger(LOGGER_NAME)
    lg.addHandler(fh)
