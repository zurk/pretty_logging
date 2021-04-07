import codecs
import datetime
import io
import logging
import re
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy
from tqdm import tqdm

_log = logging.getLogger(Path(__file__).stem)


def get_datetime_now() -> datetime.datetime:
    """
    Return the current UTC date and time.
    """
    return datetime.datetime.now(datetime.timezone.utc)


def get_timezone() -> Tuple[datetime.tzinfo, str]:
    """Discover the current time zone and it's standard string representation."""
    dt = get_datetime_now().astimezone()
    tzstr = dt.strftime("%z")
    tzstr = tzstr[:-2] + ":" + tzstr[-2:]
    return dt.tzinfo, tzstr


timezone, tzstr = get_timezone()
_now = get_datetime_now()
if _now.month == 12:
    _fest = "🌲"
elif _now.month == 10 and _now.day > (31 - 7):
    _fest = "🎃"
else:
    _fest = ""
del _now


class AwesomeFormatter(logging.Formatter):
    """
    logging.Formatter which adds colors to messages and shortens thread ids.
    """

    GREEN_MARKERS = [
        " ok",
        "ok:",
        "finished",
        "complete",
        "ready",
        "done",
        "running",
        "success",
        "saved",
    ]
    GREEN_RE = re.compile("|".join(GREEN_MARKERS))

    def _get_formatter(self, record: logging.LogRecord) -> Dict[str, str]:
        level_color = "0"
        text_color = "0"
        formatters = dict(record.__dict__)
        short_levelname = {
            "INFO": "INFO",
            "ERROR": " ERR",
            "WARNING": "WARN",
            "DEBUG": " DEB",
            "CRITICAL": "CRIT",
            "NOTSET": "  NA",
        }[record.levelname]

        if record.levelno <= logging.INFO:
            level_color = "1;36"
            lmsg = record.message.lower()
            if self.GREEN_RE.search(lmsg):
                text_color = "1;32"
        elif record.levelno <= logging.WARNING:
            level_color = "1;33"
        elif record.levelno <= logging.CRITICAL:
            level_color = "1;31"

        formatters["short_levelname"] = short_levelname
        formatters["colored_levelname"] = f"\033[{level_color}m{record.levelname}\033[0m"
        formatters["colored_short_levelname"] = f"\033[{level_color}m{short_levelname}\033[0m"
        formatters["colored_message"] = f"\033[{text_color}m{record.message}\033[0m"

        return formatters

    def formatMessage(self, record: logging.LogRecord) -> str:
        """Convert the already filled log record to a string."""

        formatter = self._get_formatter(record)
        if "colored_short_levelname" in self._fmt and record.levelno <= logging.DEBUG:
            return f"{_fest}\033[0;37m{self._fmt}s\033[0m" % formatter
        return f"{_fest}{self._fmt}" % formatter

    def usesTime(self):
        return True


class NumpyLogRecord(logging.LogRecord):
    """
    LogRecord with the special handling of numpy arrays which shortens the long ones.
    """

    @staticmethod
    def array2string(arr: numpy.ndarray) -> str:
        """Format numpy array as a string."""
        shape = str(arr.shape)[1:-1]
        if shape.endswith(","):
            shape = shape[:-1]
        return numpy.array2string(arr, threshold=11) + "%s[%s]" % (arr.dtype, shape)

    def getMessage(self):
        """
        Return the message for this LogRecord.
        Return the message for this LogRecord after merging any user-supplied \
        arguments with the message.
        """
        if isinstance(self.msg, numpy.ndarray):
            msg = self.array2string(self.msg)
        else:
            msg = str(self.msg)
        if self.args:
            a2s = self.array2string
            if isinstance(self.args, Dict):
                args = {
                    k: (a2s(v) if isinstance(v, numpy.ndarray) else v)
                    for (k, v) in self.args.items()
                }
            elif isinstance(self.args, Sequence):
                args = tuple((a2s(a) if isinstance(a, numpy.ndarray) else a) for a in self.args)
            else:
                raise TypeError(
                    "Unexpected input '%s' with type '%s'" % (self.args, type(self.args))
                )
            msg = msg % args
        return msg


class TqdmWithRemaining(tqdm):
    @property
    def remaining(self) -> Optional[float]:
        remaining = None
        if self.last_print_t != self.start_t:
            remaining = (self.last_print_t - self.start_t) / self.n * (self.total - self.n)
        if self.total - 1 == self.n:
            remaining = 0
        return remaining


class TqdmLogger(io.StringIO):
    """
    Redirects tqdm stdout stream to logger.
    """

    def __init__(self, logger: logging.Logger, level: int):
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self, text):
        text = text.replace("\r", "").replace("\x1b[A", "").strip()
        if len(text) != 0:
            self.logger.log(self.level, text)


def tqdm_logger(iterable, logger: logging.Logger, level: int = logging.INFO, *args, **kwargs):
    return TqdmWithRemaining(iterable, *args, file=TqdmLogger(logger, level), **kwargs)


def timeit(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
    def _timeit(func: Callable):
        """
        Decorator to measure execution time for class methods.
        Should be used with @with_logger decorator.
        """

        @wraps(func)
        def wrapped_timeit(*args, **kwargs):
            start_time = time.perf_counter()
            res = func(*args, **kwargs)
            delta = time.perf_counter() - start_time
            nonlocal logger
            if logger is None:
                try:
                    logger = args[0]._log
                except (KeyError, AttributeError):
                    logger = _log
            logger.log(level, f"{func.__name__} took {delta:.3f} sec")
            return res

        return wrapped_timeit

    return _timeit


def traceit(func: Callable):
    """
    Decorator to report function entry and exit for class methods.
    Should be used with @with_logger decorator.
    """

    @wraps(func)
    def wrapped_traceit(cls_or_self, *args, **kwargs):
        cls_or_self._log.debug(f"Call {func.__name__}")
        res = func(cls_or_self, *args, **kwargs)
        cls_or_self._log.debug(f"Finish {func.__name__}")
        return res

    return wrapped_traceit


def with_logger(cls):
    """Add a logger as static attribute to a class."""
    cls._log = logging.getLogger(cls.__name__)
    return cls


def setup(level: Union[str, int], coloring: bool = True, fmt: Optional[str] = None) -> None:
    """
    Make stdout and stderr unicode friendly in case of misconfigured \
    environments, initializes the logging, structured logging and \
    enables colored logs if it is appropriate.
    :param level: The global logging level.
    :param coloring: Use logging coloring or not.
    """

    if not isinstance(level, int):
        level = logging._nameToLevel[level]

    def ensure_utf8_stream(stream):
        if not isinstance(stream, io.StringIO) and hasattr(stream, "buffer"):
            stream = codecs.getwriter("utf-8")(stream.buffer)
            stream.encoding = "utf-8"
        return stream

    sys.stdout, sys.stderr = (ensure_utf8_stream(s) for s in (sys.stdout, sys.stderr))

    # basicConfig is only called to make sure there is at least one handler for the root logger.
    # All the output level setting is down right afterwards.
    logging.basicConfig()
    logging.setLogRecordFactory(NumpyLogRecord)
    root = logging.getLogger()
    root.setLevel(level)

    handler = root.handlers[0]
    datefmt = "%H:%M:%S"
    if not sys.stdin.closed and coloring:
        fmt = (
            "%(colored_short_levelname)s-%(asctime)s-%(name)s-%(colored_message)s"
            if fmt is None
            else fmt
        )
        handler.setFormatter(AwesomeFormatter(fmt, datefmt))
    else:
        fmt = "%(levelname)s-%(asctime)s-%(name)s-%(message)s" if fmt is None else fmt
        handler.setFormatter(logging.Formatter(fmt, datefmt))
