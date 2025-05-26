import sys
import os
import logging.config
from datetime import datetime


def get_log_file_path() -> str:
    """
    Get the file path for the log file.

    Returns:
        str: File path for the log file.
    """
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(
        log_dir,
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")


logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': "[%(asctime)s: %(levelname)s: %(module)s:%(lineno)d: %(message)s]",
        },
    },
    'handlers': {
        'default': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': logging.INFO,
            'stream': sys.stdout,
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'standard',
            'level': logging.INFO,
            'filename': get_log_file_path(),
            'maxBytes': 10485760,
            'backupCount': 5,
        },
    },
    'loggers': {
        'custom-logger': {
            'handlers': [
                'default',
                'file'],
            'level': logging.INFO,
            'propagate': True,
        },
    },
}

log = logging.config.dictConfig(
    logging_config) or logging.getLogger('custom-logger')
