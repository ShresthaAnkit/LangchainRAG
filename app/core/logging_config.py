from logging.config import dictConfig
from datetime import datetime
import logging
import pytz
import sys
from app.core.config import settings


class TimezoneFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt, datefmt)
        self.tz = tz

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        if self.tz:
            dt = dt.astimezone(pytz.timezone(self.tz))
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()


def setup_logging():
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": TimezoneFormatter,
                    "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "tz": "Asia/Kathmandu",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": sys.stdout,
                },
            },
            "root": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console"],
            },
        }
    )


def get_logger(name: str):
    return logging.getLogger(name)
