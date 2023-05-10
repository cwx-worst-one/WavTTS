'''
report.py used to generate dolphin weekly test reports.
'''

import os
import re
import json
import requests
from statistics import mean, stdev, median
import dateutil
import matplotlib.pyplot as plt

from config import (
    Agw,
    ARNOLD_PREFIX,
    ARNOLD_URL_PREFIX,
    WEEKLY_TEST_MODEL_PAGE_ID,
    WEEKLY_TEST_LARK_SHEET_TOKEN,
    TRAIN_MODE,
    DECODE_MODE,
    UNUSED_MODELS,
    WEEKLY_MODEL_TRAIN,
    WEEKLY_MODEL_DECODE,
    OBSOLETE_METRICS,
    MODEL_TRAIN_WORKFLOW_UUID,
    MODEL_DECODE_WORKFLOW_UUID,
    SCM_BASE_URL,
    TASK_BASE_URL,
    FALCON_BASE_URL,
    DOLPHIN_TREE_URL,
    WORKFLOW_BASE_URL,
    FALCON_ARTIFACTS_URL,
    REPORT_JSON,
    WTEST_OWNER,
    WTEST_ARNOLD_TOKEN,
    WTEST_SCM_TOKEN
)
from lark_sheet import get_data
from onnx_test_report import generate_onnx_report


def get_trial_gpuv(trial_uuid):
    '''get the trial gpuv'''
    headers = {
        'Authorization': 'Token %s' % WTEST_ARNOLD_TOKEN,
        'Content-Type': 'application/json',
    }
    trial_url = f"{ARNOLD_URL_PREFIX}/api/v2/trial/{trial_uuid}"
    response = requests.get(trial_url, headers=headers)

    gpuv = response.json()['gpuv']
    if len(gpuv) == 0:
        gpuv = 'AnyGPU'
    return gpuv


def get_trial_runtime(trial_uuid):
    '''get the trial latest runtime'''
    headers = {
        'Authorization': 'Token %s' % WTEST_ARNOLD_TOKEN,
        'Content-Type': 'application/json',
    }
    trial_url = f"{ARNOLD_URL_PREFIX}/api/v2/trial/{trial_uuid}"
    response = requests.get(trial_url, headers = headers)
    run_id = response.json()['runs'][0]['id']
    json_url = f"{ARNOLD_PREFIX}/p/arnold/api/v3/instances/?run_id={run_id}&page_size=10&page_num=1"
    headers = {
        'Authorization': 'Token %s' % WTEST_ARNOLD_TOKEN,
        'Content-Type': 'application/json',
    }
    response = requests.get(json_url, headers=headers)
    data = response.json()
    start_time = data["results"][0]["start_time"]
    stop_time = data["results"][0]["stop_time"]

    if start_time is None or len(start_time) == 0:
        return 0
    if stop_time is None or len(stop_time) == 0:
        return 0
    # the format of start_time and stop_time: "2022-08-11T08:02:34Z"
    start_time = start_time.replace('T',' ')
    start_time = start_time.replace('Z','')
    stop_time = stop_time.replace('T',' ')
    stop_time = stop_time.replace('Z','')

    date1 = dateutil.parser.parse(start_time)
    date2 = dateutil.parser.parse(stop_time)
    result = (date2 - date1).total_seconds()/3600.0
    return result


def get_total_gpu_time_for_workflow(workflow_uuid, instance_uuid):
    '''
    get the total gpu time for the workflow's latest run instance
    '''
    workflow_gpu_time = 0
    # get trial_uuids by workflow_uuid and instance_uuid
    response = requests.get(
        f'{FALCON_BASE_URL}/{workflow_uuid}/tasks?',
        params={'instance_uuid': instance_uuid},
        auth=Agw()
    ).json()
    weekly_test_data = response["data"]["data"]
    for trial in weekly_test_data:
        try:
            trial_result = {}
            trial_result['name'] = trial["name"]
            # Empty trial point used to connect all trials into a graph.
            if trial_result["name"] in ["模型解码测试", "生成报告", "start", "end"]:
                continue
            trial_result['create_time'] = re.search(
                r'\d{4}-\d{2}-\d{2}', trial['create_time']).group()
            params = json.loads(trial['params'])
            trial_result['trial_url'] = params['references']['trial_url']
            trial_uuid = trial_result["trial_url"].split('/')[-1]
            trial_runtime = get_trial_runtime(trial_uuid)
            trial_result['trail_runtime'] = trial_runtime
            trial_num = params["quota"]["worker"]["num"]
            trial_gpu = params["quota"]["worker"]["gpu"]
            # compute: trial_gpu_time
            trial_gpu_time = trial_num * trial_gpu * trial_runtime
            trial_result['gpu_time'] = trial_gpu_time
            workflow_gpu_time = workflow_gpu_time + trial_gpu_time
        except:
            print(f"trail {trial} gpu time not available.")
    return  workflow_gpu_time


def generate_history_data():
    train_uuids = []
    with open("report_json/train_workflow_uuid.txt", "r") as f:
        for line in f.readlines():
            train_uuids.append(line.strip())
    for train_uuid in train_uuids:
        results = parse_weekly_test_result(MODEL_TRAIN_WORKFLOW_UUID, train_uuid)
        for trial_name, trial_result in results.items():
            results[trial_name] = clean_trial_result(trial_result)
        datestamp = get_create_time(results)
        print(f"Save to report_json/{datestamp}.train.json...")
        with open(f"report_json/{datestamp}.train.json", "w") as f:
            json.dump(results, f)
    decode_uuids = []
    with open("report_json/decode_workflow_uuid.txt", "r") as f:
        for line in f.readlines():
            decode_uuids.append(line.strip())
    for decode_uuid in decode_uuids:
        results = parse_weekly_test_result(MODEL_DECODE_WORKFLOW_UUID, decode_uuid)
        for trial_name, trial_result in results.items():
            results[trial_name] = clean_trial_result(trial_result)
        datestamp = get_create_time(results)
        print(f"Save to report_json/{datestamp}.decode.json...")
        with open(f"report_json/{datestamp}.decode.json", "w") as f:
            json.dump(results, f)


def parse_weekly_test_sheet():
    '''
    Get latest weekly test model info from lark sheet.
    lark sheet url: https://bytedance.feishu.cn/wiki/wikcnstEneHEqZ7rdgjzHwqbrPd?sheet=i6WjU4
    '''
    # First row is column name.
    weekly_test_data = get_data(WEEKLY_TEST_LARK_SHEET_TOKEN, WEEKLY_TEST_MODEL_PAGE_ID)
    for line in weekly_test_data:
        model_name = line[1]
        test_mode = line[2]
        config_file = line[6].split()[1]
        resource = f'{line[8]}机{int(line[8]) * int(line[9])}卡'
        gpu_number=int(line[8]) * int(line[9])
        collaborator = line[3]
        if isinstance(collaborator, (list, tuple)):
            collaborator = collaborator[0]['text']
        if model_name in UNUSED_MODELS:
            continue
        else:
            if test_mode == DECODE_MODE:
                if model_name not in WEEKLY_MODEL_DECODE:
                    WEEKLY_MODEL_DECODE[model_name] = {DECODE_MODE: model_name}
                WEEKLY_MODEL_DECODE[model_name]['config_file'] = config_file
                WEEKLY_MODEL_DECODE[model_name]['resource'] = resource
                WEEKLY_MODEL_DECODE[model_name]['gpu_number'] = gpu_number
                WEEKLY_MODEL_DECODE[model_name]['collaborator'] = collaborator
            else:
                if model_name not in WEEKLY_MODEL_TRAIN:
                    if TRAIN_MODE in test_mode and DECODE_MODE in test_mode:
                        WEEKLY_MODEL_TRAIN[model_name] = {
                            TRAIN_MODE: model_name, DECODE_MODE: model_name + DECODE_MODE}
                    else:
                        WEEKLY_MODEL_TRAIN[model_name] = {TRAIN_MODE: model_name}
                WEEKLY_MODEL_TRAIN[model_name]['config_file'] = config_file
                WEEKLY_MODEL_TRAIN[model_name]['resource'] = resource
                WEEKLY_MODEL_TRAIN[model_name]['gpu_number'] = gpu_number
                WEEKLY_MODEL_TRAIN[model_name]['collaborator'] = collaborator


def load_history(mode):
    # Load history data from json files.
    history_files = os.listdir("./report_json")
    if mode == TRAIN_MODE:
        history_files = [f for f in history_files if f.endswith(".train.json")]
    elif mode == DECODE_MODE:
        history_files = [f for f in history_files if f.endswith(".decode.json")]
    history_data = {}
    for hf in history_files:
        datestamp = hf.split(".")[0]
        with open(os.path.join("./report_json", hf), "r") as f:
            history_data[datestamp] = json.load(f)
    return history_data


def get_latest_instance_uuid(workflow_uuid):
    '''
    get workflow's newest run id.
    '''
    url = f'{FALCON_BASE_URL}/{workflow_uuid}/instances?offset=0&limit=10'
    response = requests.get(url, auth=Agw())
    response = response.json()
    if response['code'] == '00000':
        print(f'{workflow_uuid} 获取instance_uuid成功')
        return response['data']['data'][0]['uuid']
    else:
        print(f'{workflow_uuid} 获取instance_uuid失败')
        return None


def get_artifacts(trial_uuid):
    '''Get artifacts for certen trial.'''
    response = requests.get(
        FALCON_ARTIFACTS_URL,
        params={"limit": 100, "task_uuid": trial_uuid},
        auth=Agw()
    ).json()
    artifacts = response['data']['artifacts']
    return artifacts


def parse_weekly_test_result(workflow_uuid, instance_uuid):
    '''get workflow instance information by workflow_id and instance_id.'''
    response = requests.get(
        f'{FALCON_BASE_URL}/{workflow_uuid}/tasks?',
        params={'instance_uuid': instance_uuid},
        auth=Agw()
    ).json()
    weekly_test_data = response["data"]["data"]
    weekly_test_result = {}
    for trial in weekly_test_data:
        trial_result = {}
        trial_result['name'] = trial["name"]
        # Empty trial point used to connect all trials into a graph.
        if trial_result["name"] in ["模型解码测试", "生成报告", "start", "end"]:
            continue
        trial_result['create_time'] = re.search(
            r'\d{4}-\d{2}-\d{2}', trial['create_time']).group()
        params = json.loads(trial['params'])
        trial_result['args'] = params['attrs']['args'].split()[1]
        if not params['references']:
            continue
        trial_result['trial_url'] = params['references']['trial_url']

        artifacts = get_artifacts(trial["uuid"])
        if artifacts.get('metric'):
            # decode: config, metric, report; train: model
            trial_result['metric_graph'] = artifacts['metric'][0]['metainfo']['hdfs_path']
        reports = artifacts.get('report', None)
        metric = []
        if reports:
            for report in reports:
                metric.append(report['metainfo']['content'])
        trial_result['metric'] = metric
        weekly_test_result[trial_result['name']] = trial_result
    return weekly_test_result


def get_task_id(trial_uuid):
    '''send request to arnold and fetch the task id.'''
    headers = {
        'Authorization': 'Token %s' % WTEST_ARNOLD_TOKEN,
        'Content-Type': 'application/json',
    }
    trial_url = f"{ARNOLD_URL_PREFIX}/api/v2/trial/{trial_uuid}"
    response = requests.get(trial_url, headers=headers)
    task_id = response.json()['task']['id']
    return task_id


def get_weekly_test_env(trial_uuid):
    '''Get weekly test env settings.'''
    headers = {
        'Authorization': 'Token %s' % WTEST_ARNOLD_TOKEN,
        'Content-Type': 'application/json',
    }
    task_id = get_task_id(trial_uuid)
    task_url = f"{TASK_BASE_URL}/{task_id}"
    response = requests.get(task_url, headers=headers).json()
    falconpai_version = response['envs']['FALCONPAI_VERSION']
    panther_version = response['envs']['PANTHER_ARNOLD_VERSION']
    dolphin_scm_id = None
    for repo in response['repos']:
        if repo['path'] == 'dolphin':
            dolphin_scm_id = repo['version']
    scm_url = f"{SCM_BASE_URL}/{dolphin_scm_id}"
    response = requests.get(scm_url, auth=(WTEST_OWNER, WTEST_SCM_TOKEN)).json()
    dolphin_version = response['branch_name']
    return dolphin_version, falconpai_version, panther_version


def clean_trial_result(trial_result, is_decode=False):
    # Merge all metric dict into one.
    # metric format: list of dict;
    metric = {k: v for dic in trial_result['metric'] for k, v in dic.items()}
    # Flatten nested metric in metric['train'] and metric['val'].
    # metric['train'] is valid only for train trial.
    if not is_decode and 'train' in metric:
        train = metric['train']
        for k, v in train.items():
            new_k = f'train {k}'
            metric[new_k] = v
        del metric['train']
    if 'val' in metric:
        val = metric['val']
        for k, v in val.items():
            new_k = f'val {k}'
            metric[new_k] = v
        del metric['val']

    # Remove unused metric.
    unused_metric_keys = ['best.pth', 'iter', 'train', 'val', 'test', ]
    for k in unused_metric_keys:
        if k in metric:
            del metric[k]

    # Merge metric into new_result.
    result = {
        'name': trial_result['name'],
        'create_time': trial_result['create_time'],
        'trial_url': trial_result['trial_url'],
        **metric
    }

    return result


def collect_history_trial_results(trial_name, metric_key, history_data):
    """Collect history trial results."""
    trial_results = {}
    for datestamp, weekly_test_results in history_data.items():
        if trial_name in weekly_test_results:
            trial_result = weekly_test_results[trial_name]
            if metric_key in trial_result:
                mk_result = trial_result[metric_key]
                if mk_result is not None:
                    if not isinstance(mk_result, float):
                        mk_result = float(mk_result)
                    trial_results[datestamp] = mk_result
    return trial_results


def is_metric_value_valid(metric_value, metric_median, metric_std, num_std=3.0):
    # Drop value out of num_std.
    norm_position = (metric_value - metric_median) / (metric_std + 1e-9)
    if norm_position >= (-1.0 * num_std) and norm_position <= num_std:
        return True
    else:
        return False


def clean_trial_results(trial_results):
    # No history.
    if len(trial_results) == 0:
        # history_start, history_end, history_min, history_max, history_mean, history_std.
        return [None, None, None, None, None, None]

    datestamps = list(trial_results.keys())
    metric_values = list(trial_results.values())
    history_start = min(datestamps)
    history_end = max(datestamps)
    metric_min = min(metric_values)
    metric_max = max(metric_values)
    metric_mean = mean(metric_values)
    metric_std = stdev(metric_values) if len(metric_values) > 1 else None
    if len(trial_results) == 1:
        return [history_start, history_end, metric_min, metric_max, metric_mean, metric_std]
    metric_median = median(metric_values)
    valid_results = {}
    for k, v in trial_results.items():
        if is_metric_value_valid(v, metric_median, metric_std):
            valid_results[k] = v
    datestamps = list(valid_results.keys())
    metric_values = list(valid_results.values())
    history_start = min(datestamps)
    history_end = max(datestamps)
    metric_min = min(metric_values)
    metric_max = max(metric_values)
    metric_mean = mean(metric_values)
    metric_std = stdev(metric_values) if len(metric_values) > 1 else None

    return [history_start, history_end, metric_min, metric_max, metric_mean, metric_std]


def get_mertric_keys(cur_trial_result, prev_trial_result, trial_name,
                     history_data, is_decode=False):
    metric_keys = list(cur_trial_result.keys())
    for key in prev_trial_result.keys():
        if key not in metric_keys:
            metric_keys.append(key)
    for result in history_data.values():
        if trial_name in result:
            for key in result[trial_name].keys():
                if key not in metric_keys:
                    metric_keys.append(key)
    metric_keys = [k for k in metric_keys if k not in ['name', 'create_time', 'trial_url']]
    if is_decode:
        metric_keys = [mk for mk in metric_keys if not mk.startswith("train")]
    trial_name = cur_trial_result['name']
    if trial_name in OBSOLETE_METRICS:
        for mk in OBSOLETE_METRICS[trial_name]:
            if mk in metric_keys:
                metric_keys.remove(mk)
    return metric_keys


def generate_result_table(cur_trial_result, prev_trial_result, trial_name,
                          history_data, is_decode=False):
    metric_keys = get_mertric_keys(
        cur_trial_result, prev_trial_result, trial_name, history_data, is_decode)
    # Get all history results and its stats.
    history_trial_results = {}
    history_starts = []
    history_ends = []
    for mk in metric_keys:
        trial_results = collect_history_trial_results(trial_name, mk, history_data)
        hstart, hend, hmin, hmax, hmean, hstd = clean_trial_results(trial_results)
        history_trial_results[mk] = [hmin, hmax, hmean, hstd]
        if hstart is not None:
            history_starts.append(hstart)
        if hend is not None:
            history_ends.append(hend)
    if len(history_starts) > 0:
        history_start = min(history_starts)
    else:
        history_start = None
    if len(history_ends) > 0:
        history_end = max(history_ends)
    else:
        history_end = None
    # Remove all zero (less than 1e-4) metric.
    zero_metrics = []
    for i in range(len(metric_keys)):
        mk = metric_keys[i]
        if mk in cur_trial_result:
            cur_mk = cur_trial_result[mk]
        else:
            cur_mk = 0.0
        if mk in prev_trial_result:
            prev_mk = prev_trial_result[mk]
        else:
            prev_mk = 0.0
        mk_values = [cur_mk, prev_mk] + history_trial_results[mk][:3]
        skip_mk = False
        for i in range(len(mk_values)):
            if isinstance(mk_values[i], (int, float, str)):
                mk_values[i] = float(mk_values[i])
            else:
                skip_mk = True
        if skip_mk:
            print(f"{cur_trial_result['name']} - {mk} skipped...")
            continue
        if max(abs(min(mk_values)), abs(max(mk_values))) < 1e-4:
            zero_metrics.append(mk)
    for mk in zero_metrics:
        metric_keys.remove(mk)

    # Draw markdown table.
    md_table = ''
    # Table header.
    table_keys = ['name', *metric_keys]
    for key in table_keys:
        md_table += f'| **{key}** '
    md_table += '|\n'
    # Table header seperator.
    md_table += '| :-: ' * len(table_keys) + "|\n"
    # Insert cur_trail_result.
    url_name = f'{cur_trial_result["name"]}_{cur_trial_result["create_time"]}'
    md_table += f'| [{url_name}]({cur_trial_result["trial_url"]}) '
    for mk in metric_keys:
        # mk not found.
        if mk not in cur_trial_result:
            md_table += f'| <span style="color:red">NOT FOUND</span> '
            continue
        mk_value = cur_trial_result[mk]
        # mk is None.
        if mk_value is None:
            md_table += fr'| <span style="color:blue">None</span> '
            continue
        # mk value type not in desired types: (str, float, int).
        if not isinstance(mk_value, (str, float, int)):
            md_table += fr'| <span style="color:yellow">{mk_value}</span> '
            continue
        if isinstance(mk_value, (str, int)):
            mk_value = float(mk_value)
        mk_mean = history_trial_results[mk][2]
        mk_std = history_trial_results[mk][3]
        if mk_mean is None or mk_std is None:
            md_table += f'| {mk_value:.4f} '
            continue
        if is_metric_value_valid(mk_value, mk_mean, mk_std, num_std=3):
            md_table += f'| {mk_value:.4f} '
        else:
            md_table += fr'| <span style="color:red">{mk_value:.4f}($>3\sigma$)</span> '
    md_table += "|\n"
    # Insert prev_trial_result.
    url_name = f'{prev_trial_result["name"]}_{prev_trial_result["create_time"]}'
    md_table += f'| [{url_name}]({prev_trial_result["trial_url"]}) '
    for mk in metric_keys:
        # mk not found.
        if mk not in prev_trial_result:
            md_table += f'| <span style="color:red">NOT FOUND</span> '
            continue
        mk_value = prev_trial_result[mk]
        # mk is None.
        if mk_value is None:
            md_table += fr'| <span style="color:blue">None</span> '
            continue
        # mk value type not in desired types: (str, float, int).
        if not isinstance(mk_value, (str, float, int)):
            md_table += fr'| <span style="color:yellow">{mk_value}</span> '
            continue
        if isinstance(mk_value, (str, int)):
            mk_value = float(mk_value)
        mk_mean = history_trial_results[mk][2]
        mk_std = history_trial_results[mk][3]
        if mk_mean is None or mk_std is None:
            md_table += f'| {mk_value:.4f} '
            continue
        if is_metric_value_valid(mk_value, mk_mean, mk_std, num_std=3):
            md_table += f'| {mk_value:.4f} '
        else:
            md_table += fr'| <span style="color:red">{mk_value:.4f}($>3\sigma$)</span> '
    md_table += "|\n"
    # Insert history results stat.
    if history_start is None and history_end is None:
        return md_table
    # min row.
    row_name = f'{history_start}_{history_end}_min'
    md_table += f'| {row_name} '
    for mk in metric_keys:
        if history_trial_results[mk][0] is None:
            md_table += '| None '
        else:
            md_table += f'| {history_trial_results[mk][0]:.4f} '
    md_table += "|\n"
    # max row.
    row_name = f'{history_start}_{history_end}_max'
    md_table += f'| {row_name} '
    for mk in metric_keys:
        if history_trial_results[mk][1] is None:
            md_table += '| None '
        else:
            md_table += f'| {history_trial_results[mk][1]:.4f} '
    md_table += "|\n"
    # mean row.
    row_name = f'{history_start}_{history_end}_mean'
    md_table += f'| {row_name} '
    for mk in metric_keys:
        if history_trial_results[mk][2] is None:
            md_table += '| None '
        else:
            md_table += f'| {history_trial_results[mk][2]:.4f} '
    md_table += "|\n"
    return md_table


def get_create_time(trial_results):
    create_time = ""
    for trial_name, metrics in trial_results.items():
        this_create_time = metrics["create_time"]
        if not create_time:
            create_time = this_create_time
        else:
            create_time = min(create_time, this_create_time)
    return create_time.replace("-", "_")


def archive_report(cur_train_result, cur_decode_result,
                   prev_train_result, prev_decode_result,
                   cur_train_instance_uuid, cur_decode_instance_uuid,
                   prev_train_instance_uuid, prev_decode_instance_uuid):
    '''
    put the report data to hdfs
    '''
    # Archive cur train and decode result.
    datestamp = get_create_time(cur_train_result)
    train_filename = f"./report_json/{datestamp}.train.json"
    with open(train_filename, "w") as f:
        json.dump(cur_train_result, f)
    print(f"Save {train_filename} to hdfs.")
    os.system(f"doas hdfs dfs -put -f {train_filename} {REPORT_JSON}/")
    datestamp = get_create_time(cur_decode_result)
    decode_filename = f"./report_json/{datestamp}.decode.json"
    with open(decode_filename, "w") as f:
        json.dump(cur_decode_result, f)
    print(f"Save {decode_filename} to hdfs.")
    os.system(f"doas hdfs dfs -put -f {decode_filename} {REPORT_JSON}/")

    # Archive prev train and decode result.
    datestamp = get_create_time(prev_train_result)
    train_filename = f"./report_json/{datestamp}.train.json"
    with open(train_filename, "w") as f:
        json.dump(prev_train_result, f)
    print(f"Save {train_filename} to hdfs.")
    os.system(f"doas hdfs dfs -put -f {train_filename} {REPORT_JSON}/")
    datestamp = get_create_time(prev_decode_result)
    decode_filename = f"./report_json/{datestamp}.decode.json"
    with open(decode_filename, "w") as f:
        json.dump(prev_decode_result, f)
    print(f"Save {decode_filename} to hdfs.")
    os.system(f"doas hdfs dfs -put -f {decode_filename} {REPORT_JSON}/")

    # Deduplicate workflow uuid file and append latest two.
    train_workflow_uuids = []
    with open(os.path.join('report_json', 'train_workflow_uuid.txt'), 'r') as f:
        for wid in f.readlines():
            if wid not in train_workflow_uuids:
                train_workflow_uuids.append(wid)
    prev_wid = prev_train_instance_uuid + '\n'
    if prev_wid not in train_workflow_uuids:
        train_workflow_uuids.append(prev_wid)
    cur_wid = cur_train_instance_uuid + '\n'
    if cur_wid not in train_workflow_uuids:
        train_workflow_uuids.append(cur_wid)
    os.remove(os.path.join('report_json', 'train_workflow_uuid.txt'))
    with open(os.path.join('report_json', 'train_workflow_uuid.txt'), 'w') as f:
        for wid in train_workflow_uuids:
            f.write(wid)
    print(f"Save train_workflow_uuid.txt to hdfs.")
    os.system(f'doas hdfs dfs -put -f report_json/train_workflow_uuid.txt {REPORT_JSON}/')

    decode_workflow_uuids = []
    with open(os.path.join('report_json', 'decode_workflow_uuid.txt'), 'r') as f:
        for wid in f.readlines():
            if wid not in decode_workflow_uuids:
                decode_workflow_uuids.append(wid)
    prev_wid = prev_decode_instance_uuid + '\n'
    if prev_wid not in decode_workflow_uuids:
        decode_workflow_uuids.append(prev_wid)
    cur_wid = cur_decode_instance_uuid + '\n'
    if cur_wid not in decode_workflow_uuids:
        decode_workflow_uuids.append(cur_wid)
    os.remove(os.path.join('report_json', 'decode_workflow_uuid.txt'))
    with open(os.path.join('report_json', 'decode_workflow_uuid.txt'), 'w') as f:
        for wid in decode_workflow_uuids:
            f.write(wid)
    print(f"Save decode_workflow_uuid.txt to hdfs")
    os.system(f'doas hdfs dfs -put -f report_json/decode_workflow_uuid.txt {REPORT_JSON}/')

    print(f'Save train_workflow_gpu_time.txt to hdfs')
    os.system(f'doas hdfs dfs -put -f report_json/train_workflow_gpu_time.txt {REPORT_JSON}/')
    print('Save decode_workflow_gpu_time.txt to hdfs')
    os.system(f'doas hdfs dfs -put -f report_json/decode_workflow_gpu_time.txt {REPORT_JSON}/')


def generate_model_train_and_decode_report():
    '''
    Parsing workflow results and write them to markdown report.
    '''
    # Step 1: Parse latest weekly test info from lark sheet.
    print("Step 1: Parse latest weekly test info from lark sheet.")
    parse_weekly_test_sheet()

    # Step 2: Get previous weekly test report.
    print("Step 2: Get previous weekly test report.")
    os.system(f"doas hdfs dfs -get {REPORT_JSON}/ ./")
    train_history_data = load_history(mode=TRAIN_MODE)
    decode_history_data = load_history(mode=DECODE_MODE)

    # Step 3: Get result for last two weekly test.
    print(f"Step 3: Get result for last two weekly test.")
    cur_train_instance_uuid = get_latest_instance_uuid(MODEL_TRAIN_WORKFLOW_UUID)
    cur_decode_instance_uuid = get_latest_instance_uuid(MODEL_DECODE_WORKFLOW_UUID)
    cur_train_result = parse_weekly_test_result(
        MODEL_TRAIN_WORKFLOW_UUID, cur_train_instance_uuid)
    cur_decode_result = parse_weekly_test_result(
        MODEL_DECODE_WORKFLOW_UUID, cur_decode_instance_uuid)
    with open('report_json/train_workflow_uuid.txt', 'r') as f:
        history_train_workflow_uuids = [tid.strip() for tid in f.readlines()]
    prev_train_instance_uuid = history_train_workflow_uuids[-1]
    if prev_train_instance_uuid == cur_train_instance_uuid:
        prev_train_instance_uuid = history_train_workflow_uuids[-2]
    with open('report_json/decode_workflow_uuid.txt', 'r') as f:
        history_decode_workflow_uuids = [tid.strip() for tid in f.readlines()]
    prev_decode_instance_uuid = history_decode_workflow_uuids[-1]
    if prev_decode_instance_uuid == cur_decode_instance_uuid:
        prev_decode_instance_uuid = history_decode_workflow_uuids[-2]
    prev_train_result = parse_weekly_test_result(
        MODEL_TRAIN_WORKFLOW_UUID, prev_train_instance_uuid)
    prev_decode_result = parse_weekly_test_result(
        MODEL_DECODE_WORKFLOW_UUID, prev_decode_instance_uuid)
    for trial_name, trial_result in cur_train_result.items():
        cur_train_result[trial_name] = clean_trial_result(trial_result, is_decode=False)
    for trial_name, trial_result in cur_decode_result.items():
        cur_decode_result[trial_name] = clean_trial_result(trial_result, is_decode=True)
    for trial_name, trial_result in prev_train_result.items():
        prev_train_result[trial_name] = clean_trial_result(trial_result, is_decode=False)
    for trial_name, trial_result in prev_decode_result.items():
        prev_decode_result[trial_name] = clean_trial_result(trial_result, is_decode=True)

    # Step 4: Get weekly test environment.
    print(f"Step 4: Get weekly test environment.")
    dolphin_version, falconpai_version, panther_version = get_weekly_test_env(
        list(cur_train_result.values())[0]['trial_url'].split('/')[-1])

    # Step 5: Get GPU time.
    print(f"Step 5: Get GPU time.")
    train_task_gpu_time = get_total_gpu_time_for_workflow(
        MODEL_TRAIN_WORKFLOW_UUID, cur_train_instance_uuid)
    decode_task_gpu_time = get_total_gpu_time_for_workflow(
        MODEL_DECODE_WORKFLOW_UUID, cur_decode_instance_uuid)
    print(f"train gpu time for current run: {train_task_gpu_time:.2f} GPU*h")
    print(f"decode gpu time for current run: {decode_task_gpu_time:.2f} GPU*h")

    # Step 6: Generate report.
    print(f"Step 6: Generate report.")
    datastamp = min(get_create_time(cur_train_result), get_create_time(cur_decode_result))
    report_filename = f'./Dolphin周测{datastamp}.md'
    with open(report_filename, 'w') as f:
        # Environment.
        f.write('# 测试环境\n')
        f.write(f'- dolphin版本: [{dolphin_version}]({DOLPHIN_TREE_URL}/{dolphin_version})\n')
        f.write(f'- falconpai版本: {falconpai_version}\n')
        f.write(f'- panther版本: {panther_version}\n')
        f.write(
            '- 精度测试workflow: [精度测试workflow]'
            f'({WORKFLOW_BASE_URL}/{MODEL_TRAIN_WORKFLOW_UUID}/content/instances/'
            f'{cur_train_instance_uuid}/content)\n'
        )
        f.write(
            '- 解码测试workflow: [精度测试workflow]'
            f'({WORKFLOW_BASE_URL}/{MODEL_DECODE_WORKFLOW_UUID}/content/instances/'
            f'{cur_decode_instance_uuid}/content)\n'
        )
        f.write(f'- 精度测试workflow 资源消耗: {train_task_gpu_time:.2f} GPU*h\n')
        f.write(f'- 解码测试workflow 资源消耗: {decode_task_gpu_time:.2f} GPU*h\n')
        # Model train result.
        f.write("# 模型精度测试\n")
        for model_name in WEEKLY_MODEL_TRAIN.keys():
            f.write(f"## {model_name}\n")
            f.write(
                f"- **模型名称：** {model_name}\n"
                f"- **机器资源：** {WEEKLY_MODEL_TRAIN[model_name]['resource']}\n"
                f"- **配置文件：** {WEEKLY_MODEL_TRAIN[model_name]['config_file']}\n"
                f"- **算法对接人：** {WEEKLY_MODEL_TRAIN[model_name]['collaborator']}\n"
            )
            f.write("- **模型训练指标对比：** \n\n")  # two blank line for markdown table.
            # Train result.
            trial_name = WEEKLY_MODEL_TRAIN[model_name][TRAIN_MODE]
            print(f"\n\nProcess {model_name} train, trial_name: {trial_name}\n" + "=" * 60)
            if trial_name in cur_train_result and trial_name in prev_train_result:
                cur_trial_result = cur_train_result[trial_name]
                prev_trial_result = prev_train_result[trial_name]
            elif trial_name in cur_train_result:
                cur_trial_result = cur_train_result[trial_name]
                prev_trial_result = cur_trial_result
            elif trial_name in prev_train_result:
                prev_trial_result = prev_train_result[trial_name]
                cur_trial_result = prev_trial_result
            else:
                print(f"trial result not found, pass this trial...")
                continue
            f.write(f"- **{TRAIN_MODE}** \n\n")

            f.write(generate_result_table(
                cur_trial_result, prev_trial_result, trial_name,
                train_history_data, is_decode=False
            ))
            f.write("\n\n")
            # Decode result.
            if DECODE_MODE in WEEKLY_MODEL_TRAIN[model_name]:
                trial_name = WEEKLY_MODEL_TRAIN[model_name][DECODE_MODE]
                print(f"\n\nProcess {model_name} decode, trial_name: {trial_name}\n" + "=" * 60)
                if trial_name in cur_train_result and trial_name in prev_train_result:
                    cur_trial_result = cur_train_result[trial_name]
                    prev_trial_result = prev_train_result[trial_name]
                elif trial_name in cur_train_result:
                    cur_trial_result = cur_train_result[trial_name]
                    prev_trial_result = cur_trial_result
                elif trial_name in prev_train_result:
                    prev_trial_result = prev_train_result[trial_name]
                    cur_trial_result = prev_trial_result
                else:
                    print(f"trial result not found, pass this trial...")
                    continue
                f.write(f"- **{DECODE_MODE}** \n\n")

                f.write(generate_result_table(
                    cur_trial_result, prev_trial_result, trial_name,
                    train_history_data, is_decode=True
                ))
                f.write("\n\n")

        # Model decode result.
        f.write("# 模型解码测试\n")
        for model_name in WEEKLY_MODEL_DECODE.keys():
            f.write(f"## {model_name}\n")
            f.write(
                f"- **模型名称：** {model_name}\n"
                f"- **机器资源：** {WEEKLY_MODEL_DECODE[model_name]['resource']}\n"
                f"- **配置文件：** {WEEKLY_MODEL_DECODE[model_name]['config_file']}\n"
                f"- **算法对接人：** {WEEKLY_MODEL_DECODE[model_name]['collaborator']}\n"
            )
            f.write("- **模型训练指标对比：** \n\n")  # two blank line for markdown table.
            # Decode result.
            trial_name = WEEKLY_MODEL_DECODE[model_name][DECODE_MODE]
            if trial_name in cur_decode_result and trial_name in prev_decode_result:
                cur_trial_result = cur_decode_result[trial_name]
                prev_trial_result = prev_decode_result[trial_name]
            elif trial_name in cur_decode_result:
                cur_trial_result = cur_decode_result[trial_name]
                prev_trial_result = cur_trial_result
            elif trial_name in prev_decode_result:
                prev_trial_result = prev_decode_result[trial_name]
                cur_trial_result = prev_trial_result
            else:
                print(f"trial result not found, pass this trial...")
                continue
            f.write(f"- **{DECODE_MODE}** \n\n")

            f.write(generate_result_table(
                cur_trial_result, prev_trial_result, trial_name,
                decode_history_data, is_decode=True
            ))
            f.write("\n\n")

    # Step 7: Archive latest result.
    print(f"Step 7: Archive latest result.")
    archive_report(cur_train_result, cur_decode_result,
                   prev_train_result, prev_decode_result,
                   cur_train_instance_uuid, cur_decode_instance_uuid,
                   prev_train_instance_uuid, prev_decode_instance_uuid)


def generate_report(workflow_type: str = 'model'):
    if workflow_type == 'model':
        generate_model_train_and_decode_report()
    elif workflow_type == 'onnx':
        generate_onnx_report()
    elif workflow_type == 'both':
        generate_model_train_and_decode_report()
        generate_onnx_report()
    else:
        raise Exception(f'Invalid value found: workflow_type = {workflow_type}.'
                        'Valid values for workflow_type: "model", "onnx", or "both"')
