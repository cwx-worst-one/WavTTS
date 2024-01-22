import logging
from collections import defaultdict
from typing import Optional

from webdataset.utils import pytorch_worker_info

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _format_worker_info():
    worker_info = pytorch_worker_info()
    return f"rank={worker_info[0]}/{worker_info[1]}, worker={worker_info[2]}/{worker_info[3]}"


class StatsProxy:
    def __init__(self, log_interval: int = 100):
        self.count = 0
        self.skipped = 0
        self.log_interval = log_interval
        self.messages = defaultdict(int)
        self.worker_info = _format_worker_info()

    def update(self, skipped: bool = True, message: Optional[str] = None):
        self.count += 1
        if skipped:
            self.skipped += 1

        if message is not None:
            self.messages[message] += 1

        if self.count % self.log_interval == 0:
            logger.warning(
                f"[{self.worker_info}] item stats: "
                f"count={self.count}, skipped={self.skipped}, "
                f"messages={dict(self.messages)}"
            )


class UpdateStatsMixin:
    def update_stats(self, skipped: bool = True, message: Optional[str] = None):
        message = f"[{self.__class__.__qualname__}] {message}" if message else None
        self.stats_proxy.update(skipped, message)

    @property
    def log_interval(self):
        return self.stats_proxy.log_interval

    @log_interval.setter
    def log_interval(self, value: int):
        self.stats_proxy.log_interval = value

    @property
    def stats_proxy(self):
        if not hasattr(self, "_stats_proxy"):
            self._stats_proxy = StatsProxy(log_interval=1000)
        return self._stats_proxy
