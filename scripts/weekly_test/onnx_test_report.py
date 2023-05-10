import json

import requests

from config import Agw, URL_PREFIX
from lark_sheet import (
    insert_line,
    get_root_folder,
    create_sheet,
    give_permission,
    get_list
)

def get_workflow_newest_id(workflow_id: str = 'e3cb86ea-ff67-4e99-832b-ae1798a1b014'):
    url = f"{URL_PREFIX}/falcon/apps/v1/studio/workflows/{workflow_id}/instances?offset=0&limit=10"
    response = requests.get(url, auth=Agw())
    response = response.json()
    if response['code'] == '00000':
        print('获取instace_uuid成功！')
        return response['data']['data'][0]['uuid']
    else:
        print('获取instance_uuid失败！')

def get_instance_information(workflows_id, instance_uuid):
    url = f'{URL_PREFIX}/falcon/apps/v1/studio/workflows/{workflows_id}/tasks?'
    params = {
        'instance_uuid': instance_uuid
    }
    response = requests.get(url, params=params, auth=Agw()).json()
    instance_information = response["data"]["data"]
    instance_info = []
    for node in instance_information:
        node_information = {}
        node_information['name'] = node["name"]
        params = json.loads(node['params'])
        if params['references'].get('trial_url'):
            node_information['trial_url'] = params['references']['trial_url']
        else:
            continue
        node_information['status'] = node['status_history'][-1]['status']
        instance_info.append(node_information)
    return instance_info

def write_data(file_token, sheet_token, instance_info):
    insert_line(file_token, sheet_token, 1, ['测试名', '状态', 'trila链接'])
    for i, node in enumerate(instance_info):
        insert_line(file_token, sheet_token, i + 2, [node['name'], node['status'], node['trial_url']])

def generate_onnx_report():
    folder_token = get_root_folder()
    file_id = create_sheet(folder_token)
    give_permission(file_id)
    instance_id = get_workflow_newest_id('d115b4c1-da9e-435d-8c24-2672636524aa')
    instance_info = get_instance_information('d115b4c1-da9e-435d-8c24-2672636524aa', instance_id)
    sheet_id = get_list(file_id)[-1]['sheetId']
    write_data(file_id, sheet_token=sheet_id, instance_info=instance_info)

if __name__ == '__main__':
    generate_onnx_report()
    
    
