import logging
import os
import time

from webdataset.utils import pytorch_worker_info

logger = logging.getLogger(__name__)


class Compose:
    """Composes several transforms together."""

    profile_env_name = "DATA_TRANSFORM_PROFILE"

    def __init__(self, transforms=None):
        """init."""

        self.transforms = transforms
        self.profile_num = 0
        self.skip_num = {}
        self.time_record = {}
        self.start = {}
        self.profilling = os.getenv(Compose.profile_env_name) == "1"
        self.skip_warning_freq = int(os.environ.get("SKIP_WARNING_FREQ", "10000"))
        self.log_num = 300
        self.compose_type = "item"
        self.rank, _, self.worker_id, _ = pytorch_worker_info()
        for transform in self.transforms:
            name = transform.__class__.__name__
            self.skip_num[name] = 0
            self.time_record[name] = 0

    def data_check_start(self, name):
        """
        data check start
        Record every 1000 steps
        """
        if self.profilling and self.profile_num > self.log_num:
            self.start[name] = time.time()

    def data_check_end(self, data, name):
        """data check end
        Record every 1000 steps, and the first 100 steps will not be record

        Return Value:
            True : means data is None
            False: means data is OK
        """
        if self.profilling and self.profile_num > self.log_num:
            self.time_record[name] += time.time() - self.start[name]
            if self.profile_num % self.log_num == 0:
                logger.warning(
                    "Dataset Profiler: rank: %d, worker_id: %d,"
                    "%s transform : %s, total used time : %f ms",
                    self.rank,
                    self.worker_id,
                    self.compose_type,
                    name,
                    self.time_record[name] * 1000,
                )
                self.time_record[name] = 0

        if data is None:
            if self.skip_num[name] % self.skip_warning_freq == 100:
                logger.warning(
                    "rank: %d, worker_id: %d, item transform %s has filter %d datas",
                    self.rank,
                    self.worker_id,
                    name,
                    self.skip_num[name],
                )
            self.skip_num[name] += 1
            return True
        return False

    def __call__(self, item):  # sourcery skip: raise-specific-error
        """call func."""
        for trans in self.transforms:
            class_name = trans.__class__.__name__
            self.data_check_start(class_name)
            try:
                item = trans(item)

            except Exception as e:
                raise Exception(
                    f"rank: {self.rank}, worker id: {self.worker_id}, "
                    "{class_name} transform error: {e}"
                ) from e
            if self.data_check_end(item, class_name):
                return None
            if item is None:
                return None
        self.profile_num += 1
        return item

    def __repr__(self):
        """string repr."""
        format_string = f"{self.__class__.__name__}("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class BatchCompose(Compose):
    """Composes draw batch funtions."""

    def __init__(self, transforms=None):
        super().__init__(transforms)
        self.log_num = 30
        self.compose_type = "batch"

    def __call__(self, batch, **kwargs):
        # sourcery skip: raise-specific-error
        """call func."""
        batch_out = {}

        for collate_fn in self.transforms:
            class_name = collate_fn.__class__.__name__
            self.data_check_start(class_name)
            try:
                batch_out = collate_fn(batch)
            except Exception as e:
                raise Exception(
                    f"rank: {self.rank}, {class_name} transform error: {e}"
                ) from e
            if self.data_check_end(batch_out, class_name):
                return None
            if batch_out is None:
                return None
        self.profile_num += 1
        return batch_out
