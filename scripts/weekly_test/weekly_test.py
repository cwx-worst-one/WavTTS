'''
run the weekly test
1.create new branch from develop
2.post a new SCM
3.update the operator of weekly test
4.run the weekly test workflow
'''

import json
import time
import datetime

import gitlab
import requests
from fire import Fire

from config import (MODEL_DECODE_WORKFLOW_UUID, MODEL_TRAIN_WORKFLOW_UUID,
                    ONNX_EXPORT_WORKFLOW_UUID, TASK_BASE_URL,
                    WTEST_ARNOLD_TOKEN, WTEST_GITLAB_TOKEN, WTEST_OWNER,
                    WTEST_SCM_TOKEN, URL_PREFIX, Agw)


def checkout_branch(branch_name):
    '''checkout branch from develop.'''

    gl = gitlab.Gitlab('https://code.byted.org', private_token=WTEST_GITLAB_TOKEN)
    project = gl.projects.get(121303)
    project.branches.create({'branch': branch_name, 'ref': 'develop'})
    print(f"Checkout branch {branch_name} from develop success.")


def release_scm(create_user, branch_name):
    '''release scm using branch which name is branch_name.'''

    url = "https://scm.byted.org/api/v2/versions/cicd_create/"
    headers = {'Content-Type': 'application/json'}
    data = {
        "create_user": create_user,
        "repo_id": "61223",
        "branch_name": branch_name,
        "type": 'test',
    }
    data = json.dumps(data)
    response = requests.post(url, headers=headers, data=data, auth=(WTEST_OWNER, WTEST_SCM_TOKEN))
    response = response.json()
    if response['data']['state'] != 'passed':
        raise Exception(response)
    scm_result = {}
    scm_result['version_version'] = response['data']['context']['version_version']
    scm_result['version_id'] = response['data']['context']['version_id']
    print(f"scm release success, {scm_result['version_version']}, {scm_result['version_id']}")
    return scm_result


def query_scm_state(version_id: str = ""):
    '''query scm state.'''

    url = f"https://scm.byted.org/api/v2/versions/{version_id}/cicd_poll/"
    response = requests.get(url).json()
    state = response['data']['state']
    if state == 'passed':
        return True
    elif state == 'failed':
        raise Exception(f'scm编译失败, version_id: {version_id}.')


def get_workflow_information(workflow_id: str = "dc0188e6-86cd-4330-b8fa-ca74fe40c53f"):
    '''get workflow information by workflow id.'''

    url = f"{URL_PREFIX}/falcon/apps/v1/studio/workflows/{workflow_id}?version=0"
    response = requests.get(url, auth=Agw()).json()
    return response['data']


def run_workflow(workflow_info,
                 task_id="1739338",
                 workflow_id: str = 'dc0188e6-86cd-4330-b8fa-ca74fe40c53f',
                 group_ids: str = '52',
                 cluster_id: int = 17,
):
    '''
    start workflow.

    Args:
        workflow_info: workflow中对应的每个结点信息
        task_id: 启动workflow所对应的task_id
        workflow_id: 启动workflow的id
    '''
    url = f'{URL_PREFIX}/falcon/apps/v1/studio/workflows/{workflow_id}/instances'
    for node in workflow_info['dag']['nodes']:
        print(f"update task_id for node name: {node['name']}")
        node['attrs']['task_id'] = task_id
        if node['name'] == '模型精度测试' or node['name'] == '模型解码测试':
            continue
    for node in workflow_info['dag']['nodes']:
        if node['type'] == 'arnold':
            node['attrs']['arnold_token'] = WTEST_ARNOLD_TOKEN
    data = {
        'version': 0,
        'style': workflow_info['style'],
        'dag': workflow_info['dag'],
        'temp_dag': {
            "nodes": [
                {
                    "attrs": {
                        "cluster_id": cluster_id,
                        "group_ids": [int(group_id) for group_id in group_ids.split(',')],
                        "mask_hosts": []
                    }
                },
            ] * len(workflow_info['dag']['nodes']),
            "vars": {"params": [], "flags": []}
        }
    }
    data = json.dumps(data)
    response = requests.post(url, data=data, auth=Agw()).json()
    if response['code'] == '00000':
        print(f'workflow {workflow_id} 启动成功！')
    else:
        print(f'workflow {workflow_id} 启动失败！')


def update_arnold_task(task_id, scm_version, envs_updated):
    url = TASK_BASE_URL + '/' + str(task_id) + '/'
    resp = requests.get(url, headers={'Authorization': "token %s" % WTEST_ARNOLD_TOKEN})
    resp.raise_for_status()

    envs = resp.json()['envs']
    envs.update(envs_updated)

    update_task_data = {
        'repos': [
            {
                "repo_name": "lab/speech/dolphin",
                "path": "dolphin",
                "version": scm_version
            }
        ],
        'envs': envs,
    }
    print(f"准备更新 Arnold Task {task_id}, data={update_task_data}")
    resp = requests.put(url, headers={
        'Authorization': "token %s" % WTEST_ARNOLD_TOKEN
    }, json=update_task_data)

    resp.raise_for_status()

    while True:
        resp = requests.get(url, headers={'Authorization': f"token {WTEST_ARNOLD_TOKEN}"})
        current_status = resp.json()['status']
        if current_status == 'active':
            print("更新 Arnold Task 成功")
            break
        elif current_status == 'failed':
            raise Exception(f"更新 Arnold Task {task_id} 失败")

        print(f"Arnold task {task_id} 更新中, status={current_status}")
        time.sleep(10)


def run_weekly_test(train_task_id, decode_task_id, onnx_task_id,
                    panther_version, group_ids, cluster_id,
                    run_train, run_decode, run_onnx):
    '''
    It is used to execute operations before weekly test, such as checkout branch,
    release scm, update operator and start workflow.

    Args:
        为onnx为onnx导出测试,为both则都启动。默认为model。
        owner: 启动人的邮箱前缀
    '''
    workflow_ids = []
    arnold_task_ids = []
    if run_decode:
        workflow_ids.append(MODEL_DECODE_WORKFLOW_UUID)
        arnold_task_ids.append(decode_task_id)
        print('启动 模型解码测试 工作流')
    if run_onnx:
        workflow_ids.append(ONNX_EXPORT_WORKFLOW_UUID)
        arnold_task_ids.append(onnx_task_id)
        print('启动 onnx导出测试 工作流')
    if run_train:
        workflow_ids.append(MODEL_TRAIN_WORKFLOW_UUID)
        arnold_task_ids.append(train_task_id)
        print('启动 模型精度测试 工作流')
    if len(workflow_ids) != len(arnold_task_ids):
        raise Exception("wrong workflow id and arnold task")


    branch_name = f'develop{str(datetime.date.today())}_{int(time.time())}'
    checkout_branch(branch_name)
    scm_result = release_scm(WTEST_OWNER, branch_name=branch_name)
    print('scm 开始编译')
    while not query_scm_state(scm_result['version_id']):
        time.sleep(10)
        print(f"scm 编译中...")
        pass
    scm_version = scm_result['version_version']
    print(f'scm编译成功, scm_version: {scm_version}...')

    for workflow_id, task_id in zip(workflow_ids, arnold_task_ids):
        if not workflow_id or not task_id:
            raise ValueError(f"workflow_id = {workflow_id}, task_id = {task_id}")
        update_arnold_task(
            task_id,
            scm_version,
            envs_updated={"PANTHER_ARNOLD_VERSION": panther_version},
        )
        workflow_info = get_workflow_information(workflow_id=workflow_id)
        run_workflow(
            workflow_info=workflow_info,
            task_id=task_id,
            workflow_id=workflow_id,
            group_ids=group_ids,
            cluster_id=cluster_id,
        )


if __name__ == '__main__':
    Fire(run_weekly_test)
