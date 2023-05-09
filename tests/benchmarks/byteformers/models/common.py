from contextlib import suppress
from time import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.autograd.profiler import record_function

from tests.helpers.testing_utils import is_cuda_available


def _get_trace_handler(name: str):
    def trace_handler(prof):
        prof.export_chrome_trace(f"profile_{name}.json")
        prof.export_stacks(f"stacks_{name}.txt", "self_cuda_time_total")

    return trace_handler


def _train_steps(
    model_tester,
    num_steps: int,
    profile: bool = True,
    autocast: bool = True,
    lr: float = 0.01,
    compile_model: bool = False,
):
    inputs = model_tester.get_inputs()
    model = model_tester.create_and_test_model_with_lm_head()
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    if compile_model:
        model = torch.compile(model)

    if is_cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time()
    if profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
            on_trace_ready=_get_trace_handler(
                f"{model_tester.__class__.__name__}-autocast-{autocast}-batch-{model_tester.batch_size}-seq_len-{model_tester.seq_len}-n_embd-{model_tester.n_embd}"  # noqa
            ),
            profile_memory=True,
            with_stack=True,
        )
    else:
        profiler = suppress()

    norm_type = None
    with profiler as p:
        for _ in range(num_steps):
            optim.zero_grad()

            with torch.cuda.amp.autocast(enabled=autocast):
                with record_function("attention_forward"):
                    output = model(inputs)

                with record_function("loss"):
                    output = rearrange(output, "b n c -> b c n")
                    loss = F.cross_entropy(output, inputs)

            with record_function("backward"):
                loss.backward()

            if norm_type is not None:
                clip_norm = 0.3
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, norm_type)
            optim.step()

            if p:
                p.step()

    if is_cuda_available:
        torch.cuda.synchronize()
        max_memory = torch.cuda.max_memory_allocated() / 2**20
    else:
        max_memory = -1

    run_time = time() - start_time
    return run_time, round(max_memory, 1)


def benchmark_model(model_tester, num_warmup: int, num_steps: int, **kwargs):
    warm_up_args = {**kwargs}
    warm_up_args["profile"] = False
    _train_steps(model_tester, num_steps=num_warmup, **warm_up_args)
    return _train_steps(model_tester, num_steps=num_steps, **kwargs)

    num_steps = 50


def benchmark_model_plots(model_tester, num_steps):
    compile_model = False
    batch_sizes = [1]
    summary = []
    for batch_size in batch_sizes:
        model_tester.batch_size = batch_size
        run_time, max_memory = benchmark_model(
            model_tester,
            num_warmup=10,
            num_steps=num_steps,
            compile_model=compile_model,
            autocast=True,
        )
        steps_per_sec = run_time / num_steps
        summary.append(steps_per_sec)

    plt.plot(batch_sizes, summary, marker="o", linestyle="dashed")
    plt.xlabel("Batch size")
    plt.ylabel("Sec/it")
    plt.title(f"{model_tester.__class__.__name__} Benchmark")
    plt.legend()
    plt.savefig("benchmark.png")
    plt.close()
