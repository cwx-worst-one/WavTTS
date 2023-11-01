import datetime
import logging
import time

logger = logging.getLogger(__name__)


def elapsed_time(func):
    def inner(*args, **kwargs):
        logger.info(f"Launch {func.__name__}({args=}, {kwargs=})")
        start = time.perf_counter()
        ret = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        elapsed = str(datetime.timedelta(seconds=elapsed))
        logger.info(f"{func.__name__} {elapsed=}")
        return ret

    return inner
