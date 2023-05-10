# pylint: disable=missing-module-docstring
from .config import Config, ConfigDict, DictAction
from .logging import get_logger, LogLevelChange, all_rank_info
from .math import ceil
from .misc import (
    check_prerequisites,
    concat_list,
    is_list_of,
    is_seq_of,
    is_str,
    is_tuple_of,
    iter_cast,
    list_cast,
    requires_executable,
    requires_package,
    slice_list,
    tuple_cast,
)
from .checkpoint import (
    load_checkpoint,
    save_checkpoint,
    weights_to_cpu,
)
from .path import check_file_exist, fopen, is_filepath, mkdir_or_exist, scandir, symlink
from .progressbar import ProgressBar, track_iter_progress, track_parallel_progress, track_progress
from .registry import Registry, build_from_cfg
from .recall import compute_recall_precision, compute_emotion_metrics, compute_dimemotion_metrics
from .timer import Timer, TimerError, check_time
from .fileio import (
    BaseStorageBackend,
    FileClient,
    BaseFileHandler,
    JsonHandler,
    PickleHandler,
    YamlHandler,
    dump,
    load,
    register_handler,
    dict_from_file,
    list_from_file,
)
from .dist_util import (
    distributed_init,
    get_dist_info,
    master_only,
    dist_allreduce,
    dist_barrier,
    get_rank,
    get_local_rank,
    get_world_size,
    local_master_only,
    dist_broadcast_model,
    ReduceOp,
    dist_broadcast,
    get_communicator,
)
from .hdfs import hdfs_get, hdfs_put, hdfs_mkdir, hdfs_copy, hdfs_test, hdfs_rm, hdfs_ls
from .dist_hdfs import dist_hdfs_get, dist_hdfs_put, dist_file_get
from .confidence import compute_confidence
from .dict import FalconDict
from .se import gen_diffuse
from .wav_util import get_wav_len, get_frames_len, wav_start_pos
from .edit_distance import edit_distance
from .complex import complex_multiply
from .split import split_list, flatten_list
from .data_utils import DataIter, MultiDatasIter


# alignment for Nvidia tensor core
TC_ALIGN = 8


__all__ = [
    'ceil',
    'TC_ALIGN',
    'BaseStorageBackend',
    'FileClient',
    'load',
    'dump',
    'register_handler',
    'BaseFileHandler',
    'JsonHandler',
    'PickleHandler',
    'YamlHandler',
    'list_from_file',
    'dict_from_file',
    'Config',
    'ConfigDict',
    'DictAction',
    'get_logger',
    'is_str',
    'iter_cast',
    'list_cast',
    'tuple_cast',
    'is_seq_of',
    'is_list_of',
    'is_tuple_of',
    'slice_list',
    'concat_list',
    'check_prerequisites',
    'requires_package',
    'requires_executable',
    'is_filepath',
    'fopen',
    'check_file_exist',
    'mkdir_or_exist',
    'symlink',
    'scandir',
    'ProgressBar',
    'track_progress',
    'track_iter_progress',
    'track_parallel_progress',
    'Registry',
    'build_from_cfg',
    'Timer',
    'TimerError',
    'check_time',
    'distributed_init',
    'get_rank',
    'get_local_rank',
    'get_world_size',
    'get_dist_info',
    'dist_allreduce',
    'dist_barrier',
    'master_only',
    'local_master_only',
    'dist_broadcast_model',
    'dist_broadcast',
    'ReduceOp',
    'hdfs_get',
    'hdfs_put',
    'hdfs_mkdir',
    'hdfs_copy',
    'hdfs_test',
    'hdfs_rm',
    'dist_hdfs_get',
    'dist_hdfs_put',
    'compute_confidence',
    'compute_recall_precision',
    'compute_emotion_metrics',
    'compute_dimemotion_metrics',
    'LogLevelChange',
    'all_rank_info',
    'FalconDict',
    'gen_diffuse',
    'get_wav_len',
    'get_frames_len',
    'wav_start_pos',
    'edit_distance',
    'complex_multiply',
    'split_list',
    'flatten_list',
    'DataIter',
    'MultiDatasIter',
    'dist_file_get',
    'load_checkpoint',
    'weights_to_cpu',
    'save_checkpoint',
    'get_communicator',
]
