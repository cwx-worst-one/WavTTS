import logging
import os
import torch
from tests.model_test.models.model_dict import ModelTesterDcit
from tests.model_test.metrics.metirc_dict import MetricDict, ReportDict


def setup_tester(args):
    logging.info(f"setup tester for {args.model_params.model} model.")
    return ModelTesterDcit[args.model_params.model](args)


def setup_model(tester, provider):
    logging.info(f"setup model for provider {provider}.")
    model, model_size = tester.gen_model(provider)
    if model is not None:
        if tester.action == 'generate':
            model.eval()
        else:
            model.train()

    return model, model_size


def setup_metric_inputs(tester, provider, metric):
    logging.info(f"setup {metric} metric inputs for provider {provider}.")
    return tester.gen_inputs(metric, provider)


def setup_output_dir(args):
    output_dir = os.path.join(os.getcwd(), args.benchmark_params.output_dir,
                              args.model_params.model, args.benchmark_params.exp_dir, args.benchmark_params.action)
    if "model_name" in args.model_params.keys():
        output_dir = os.path.join(output_dir, args.model_params.model_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def run_metrics(args, tester):
    full_info = {'action': args.benchmark_params.action, "metrics": {}}
    for metric in tester.args.benchmark_params.metrics:
        try:
            _ = MetricDict[metric]
        except KeyError:
            logging.warning(f'metric `{metric}` has not been implemented.')
            continue
        full_info["metrics"][metric] = {}

    for provider in tester.args.model_params.providers:
        model, model_size = setup_model(tester, provider)
        if model is None:
            continue
        args.model_size = model_size

        for metric in full_info["metrics"].keys():
            metric_inputs = setup_metric_inputs(tester, provider, metric)
            if metric_inputs is None:
                continue

            metric_res = MetricDict[metric](args, provider, model, metric_inputs)
            if metric_res is None:
                logging.warning(f'metric `{metric}` has no outputs.')
                continue

            full_info["metrics"][metric][provider] = metric_res
        del model
        torch.cuda.empty_cache()
    return full_info


def generate_reports(args, tester, results):
    args.device_type = torch.cuda.get_device_name(torch.cuda.current_device())
    args.final_output_dir = setup_output_dir(args)

    for metric in tester.args.benchmark_params.metrics:
        try:
            report_fn = ReportDict[metric]
        except KeyError:
            logging.warning(f'metric `{metric}` has not been implemented.')
            continue

        report_fn(args, metric, results["metrics"][metric])
