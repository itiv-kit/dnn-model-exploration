import logging

LOGGER_NAME = "quantization_logger"
LOG_FILE_PATH = "results/workload_run.log"

logger = logging.getLogger(LOGGER_NAME)
logger.propagate = False
logger.setLevel(logging.DEBUG)

# log to console
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.propagate = False

# log to file
fh = logging.FileHandler(LOG_FILE_PATH)
fh.setLevel(logging.DEBUG)

formatter_file = logging.Formatter(
    "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
)
formatter_console = logging.Formatter(
    "[%(asctime)s] %(levelname)-8s - %(message)s"
)

fh.setFormatter(formatter_file)
ch.setFormatter(formatter_console)

logger.addHandler(fh)
logger.addHandler(ch)
