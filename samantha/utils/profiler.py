import pytorch_lightning
import pytorch_lightning.profilers
import torch
import torch.profiler
from torch.profiler import ProfilerActivity


def make_pytorch_profiler(
    activities=None,
    wait=40,
    warmpup=10,
    active=3,
    repeat=1,
    dir_name="log/profile",
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) -> pytorch_lightning.profilers.PyTorchProfiler:
    """
    Create a PyTorch Profiler instance.

    Args:
        - activities (list, optional): List of activities to be traced.
          Defaults to [ProfilerActivity.CPU, ProfilerActivity.CUDA].
        - wait (int, optional): Number of steps to wait before starting the first profiling cycle. Defaults to 40.
        - warmpup (int, optional): Number of warmup steps before starting to record. Defaults to 10.
        - active (int, optional): Number of steps to record profiling data. Defaults to 3.
        - repeat (int, optional): Number of times to repeat the profiling cycle. Defaults to 1.
        - dir_name (str, optional): Directory to store profiling logs. Defaults to "log/profile".
        - record_shapes (bool, optional): Whether to record tensor shapes. Defaults to True.
        - profile_memory (bool, optional): Whether to track memory allocation/deallocation. Defaults to True.
        - with_stack (bool, optional): Whether to record the call stack information of operations. Defaults to True.

    Returns:
        PyTorchProfiler: Configured PyTorch Profiler instance.
    """
    activities = activities or [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    schedule = torch.profiler.schedule(
        wait=wait, warmup=warmpup, active=active, repeat=repeat
    )
    if dir_name is not None:
        on_trace_ready = torch.profiler.tensorboard_trace_handler(dir_name)
    else:
        on_trace_ready = None
    profiler = pytorch_lightning.profilers.PyTorchProfiler(
        activities=activities,
        schedule=schedule,
        on_trace_ready=on_trace_ready,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    )
    return profiler
