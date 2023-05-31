import os
import torch
import logging
from tabulate import tabulate

from tests.model_test.report.plot import plot_bars

from s3a.utils.benchmark import benchmark_forward


def metric_time(args, provider, model, model_inputs):
    logging.info(f'run time metric for provider {provider}.')

    res = {}
    for max_seqlen in model_inputs.keys():
        res[max_seqlen] = {}
        for batch_size in model_inputs[max_seqlen].keys():
            res[max_seqlen][batch_size] = {}

            def fn(inputs):
                torch.cuda.synchronize()
                if args.benchmark_params.action == 'generate':
                    output = model.generate(**inputs)
                else:
                    output = model(**inputs)
                torch.cuda.synchronize()
                return output

            try:
                _, m = benchmark_forward(fn, model_inputs[max_seqlen][batch_size]["inputs"],
                                         repeats=args.benchmark_params.repeats, verbose=args.benchmark_params.verbose, desc=provider.upper())
            except Exception as error:
                logging.error(
                    f'({__name__}): an exception occurred for provider `{provider}` in time metric:{type(error).__name__}, {error}')
                raise

            res[max_seqlen][batch_size]["outputs"] = {
                'raw_times': m.raw_times, 'repeat': m.number_per_run, 'num_threads': m.num_threads}
            torch.cuda.empty_cache()

    return res


def report_time(args, metric, res: dict):
    logging.info(f'generate reports for time metric.')
    providers = list(res.keys())
    max_seqlens = list(res[providers[0]].keys())
    batch_sizes = list(res[providers[0]][max_seqlens[0]].keys())

    output_dir = os.path.join(args.final_output_dir, metric)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_csv = open(os.path.join(output_dir, "output.csv"), "w")

    baseline = res[providers[0]]
    for max_seqlen in max_seqlens:
        output_csv.write("max_seqlen=" + str(max_seqlen) + "\n")
        info = {"batch_sizes": batch_sizes}
        plot_datas = []
        for provider in providers:
            info_in_provider = []
            plot_data = []
            for batch_size in batch_sizes:
                cur_raw_time = res[provider][max_seqlen][batch_size]['outputs']['raw_times']
                baseline_time = baseline[max_seqlen][batch_size]['outputs']['raw_times']
                info_in_provider.append(
                    str(cur_raw_time[0]) + "ms(" + str("{:.5f}".format(baseline_time[0] / cur_raw_time[0])) + ")")
                plot_data.append(cur_raw_time[0])
            info[provider] = info_in_provider
            plot_datas.append(plot_data)
        content = tabulate(info, headers="keys", tablefmt="tsv")
        output_csv.write(content)
        output_csv.write("\n")

        plot_bars(args=args, output_dir=output_dir, datas=plot_datas, batch_size=batch_sizes, max_seqlen=max_seqlen, providers=providers,
                  ylabel="Time (ms)")

    output_csv.close()
