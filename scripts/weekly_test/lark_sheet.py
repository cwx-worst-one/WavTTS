# --coding:utf-8--
from http.server import BaseHTTPRequestHandler, HTTPServer
from os import path, system
import json
from urllib import request, parse
import datetime
import time
import sys
import requests

# szy-test
# 目前只有sunzhouyi可用，没有申请更大的权限
# APP_ID = "cli_a181dd85b1f8100d"
# APP_SECRET = "qSTdX3jh295aOjcsQyCeUdBbH4jut7ga"
# APP_VERIFICATION_TOKEN = "hD6LwAMLZUBV2XVHRbQF8dCYK5SFmtTf"

# dolphin-test
APP_ID = "cli_a1917ea101f8100d"
APP_SECRET = "HMugE8wrel4inLFFJGuICeaVmuntapn3"
APP_VERIFICATION_TOKEN = "NOCqN6P1oy93m5QyVBvR2fKm8EBsTAZX"


#dolphin mr board
DOLPHIN_MR_BOARD = "shtcnwHRgjIeQD0bfU2jEfzlNGb"
#falconpai mr board
FALCONPAI_MR_BOARD = "shtcnJWONfYaoch1vnP7wbtwkud"
#test mr board
JoneySun = 'shtcnxpdKqSvaM9SEBVxUARMIub'


ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def get_tenant_access_token():
    # 获取tenant_access_token
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/"
    headers = {"Content-Type": "application/json"}
    req_body = {"app_id": APP_ID, "app_secret": APP_SECRET}

    data = bytes(json.dumps(req_body), encoding='utf8')
    req = request.Request(url=url, data=data, headers=headers, method='POST')
    try:
        response = request.urlopen(req)
    except Exception as e:
        print(e.read().decode())
        return ""

    rsp_body = response.read().decode('utf-8')
    rsp_dict = json.loads(rsp_body)
    code = rsp_dict.get("code", -1)
    if code != 0:
        print("get tenant_access_token error, code =", code)
        return ""
    return rsp_dict.get("tenant_access_token", "")


def get_user_access_token(code):
    # 获取user_access_token
    url = "https://open.feishu.cn/open-apis/authen/v1/access_token"
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Authorization": "Bearer " + get_tenant_access_token(),
    }
    req_body = {
        "grant_type": "authorization_code",
        "code": code,
    }

    data = bytes(json.dumps(req_body), encoding='utf8')
    req = request.Request(url=url, data=data, headers=headers, method='POST')
    try:
        response = request.urlopen(req)
    except Exception as e:
        print(e.read().decode())
        return "", ""

    rsp_body = response.read().decode('utf-8')
    rsp_dict = json.loads(rsp_body)
    # print(rsp_dict)
    if rsp_dict.get('code', '-1') != 0:
        print(rsp_dict.get('code', 'none'), rsp_dict.get("msg", "none"))
    return (
        rsp_dict.get('data', {}).get("access_token", ""),
        rsp_dict.get('data', {}).get("refresh_token", ""),
    )


refresh_token = ""


def fresh_token():
    # 刷新token，目前没申请权限
    url = "https://open.feishu.cn/open-apis/authen/v1/refresh_access_token"
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Authorization": "Bearer " + get_tenant_access_token(),
    }
    req_body = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }

    data = bytes(json.dumps(req_body), encoding='utf8')
    req = request.Request(url=url, data=data, headers=headers, method='POST')
    try:
        response = request.urlopen(req)
    except Exception as e:
        print(e.read().decode())
        return "", ""

    rsp_body = response.read().decode('utf-8')
    rsp_dict = json.loads(rsp_body)
    # print(rsp_dict)
    if rsp_dict.get('code', '-1') != 0:
        print(rsp_dict.get('code', 'none'), rsp_dict.get("msg", "none"))
    return rsp_dict.get('data', {}).get("access_token", ""), rsp_dict.get('data', {}).get(
        "refresh_token", ""
    )


def read_user_access_token():
    # 从文件读取token
    try:
        with open("status.temp", "r") as f:
            user_access_token = f.readline().strip()
            refresh_token = f.read()
            return user_access_token
    except Exception:
        return ""


def write_user_access_token(code):
    # 保存token到文件
    user_access_token, refresh_token = get_user_access_token(code)
    if user_access_token:
        with open("status.temp", "w") as f:
            f.write(user_access_token + '\n')
            f.write(refresh_token)
    # print(user_access_token)


def fresh_user_access_token():
    # 刷新用户token，目前没申请权限
    read_user_access_token()
    user_access_token, refresh_token = fresh_token()
    if user_access_token:
        with open("status.temp", "w") as f:
            f.write(user_access_token + '\n')
            f.write(refresh_token)


def get_data(spreadsheetToken, sheet_id):
    # 获取表格数据
    url = "https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{}/values/{}".format(
        spreadsheetToken, sheet_id
    )
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Authorization": "Bearer " + get_tenant_access_token(),
    }
    req_body = {
        # "spreadsheetToken" : spreadsheetToken,
    }

    data = bytes(json.dumps(req_body), encoding='utf8')
    req = request.Request(url=url, data=data, headers=headers, method='GET')
    try:
        response = request.urlopen(req)
    except Exception as e:
        print(e.read().decode())
        return ""
    rsp_body = response.read().decode('utf-8')
    rsp_dict = json.loads(rsp_body)
    return list(filter(lambda x: x[0], rsp_dict['data']['valueRange']['values']))


def pre_work(dct):
    #对单元格数据做一些预处理，兼容飞书表格
    EMAIL = {
        '@邵子奇': 'shaoziqi.sam@bytedance.com',
        '@李勇': 'liyong.0517@bytedance.com',
        '@张骏': 'zhangjun.jarry@bytedance.com',
        '@侯俊峰': 'houjunfeng@bytedance.com',
        '@田垚': 'tianyao.11@bytedance.com',
        '@吴培昊': 'wupeihao@bytedance.com',
        '@陈金坤': 'chenjinkun.kv7@bytedance.com',
        '@唐玉峰': 'tangyufeng.aladdin@bytedance.com',
        '@黄露': 'huanglu.thu19@bytedance.com',
        '@刘艺': 'liuyi.ai@bytedance.com',
        '@刘亚明': 'liuyaming.lamy@bytedance.com',
        '@梁镇麟': 'liangzhenlin.lzl@bytedance.com',
        '@江波': 'jiangbo.jacob@bytedance.com',
        '@吴培昊': 'wupeihao@bytedance.com',
        '@沈辰': 'shenchen.0622@bytedance.com',
        '@唐宇': 'tangyu.yt@bytedance.com',
    }
    if dct is None:
        return
    if isinstance(dct, dict):
        if dct.get('type') == 'mention' and dct.get('en_name'):
            if dct.get('text') not in EMAIL:
                print("error", dct.get('text'))
            dct['mentionNotify'] = False
            dct['text'] = EMAIL.get(dct.get('text').strip(), dct.get('text'))
        elif dct.get('type') == 'mention':
            dct.setdefault('textType', 'fileToken')
            dct['text'] = dct['token']
            dct.setdefault('objType', 'doc')
        elif dct.get('type') == 'url':
            dct.setdefault('link', dct.get('text'))
        else:
            pass
            # print("error",dct)
        return
    if isinstance(dct, list):
        for i in range(len(dct)):
            pre_work(dct[i])


def write_data(spreadsheetToken, range_, data):
    # 覆盖数据
    for i in range(len(data)):
        for j in range(len(data[i])):
            pre_work(data[i][j])
    url = "https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{}/values".format(
        spreadsheetToken
    )
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Authorization": "Bearer " + read_user_access_token(),
    }
    req_body = {
        "valueRange": {
            "values": data,
            "range": range_,
        },
    }
    data = bytes(json.dumps(req_body), encoding='utf8')
    req = request.Request(url=url, data=data, headers=headers, method='PUT')
    try:
        response = request.urlopen(req)
    except Exception as e:
        print(e.read().decode())
        return ""
    rsp_body = response.read().decode('utf-8')
    rsp_dict = json.loads(rsp_body)
    return rsp_dict['code']


def write_data(spreadsheetToken, sheet_id, row, col, data):
    # 覆盖原来的数据
    data = [[data]]
    for i in range(len(data)):
        for j in range(len(data[i])):
            pre_work(data[i][j])
    range_ = '{}!{}{}:{}{}'.format(sheet_id, ALPHA[col], row + 1, ALPHA[col], row + 1)
    url = "https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{}/values".format(
        spreadsheetToken
    )
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Authorization": "Bearer " + read_user_access_token(),
    }
    req_body = {
        "valueRange": {
            "values": data,
            "range": range_,
        },
    }
    data = bytes(json.dumps(req_body), encoding='utf8')
    req = request.Request(url=url, data=data, headers=headers, method='PUT')
    try:
        response = request.urlopen(req)
    except Exception as e:
        print(e.read().decode())
        return ""
    rsp_body = response.read().decode('utf-8')
    rsp_dict = json.loads(rsp_body)
    return rsp_dict['code']


def set_style(spreadsheetToken, sheet_id, style):
    url = "https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{}/styles_batch_update".format(
        spreadsheetToken
    )
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Authorization": "Bearer " + read_user_access_token(),
    }

    data = []
    for i in range(len(style)):
        for j in range(len(style[i])):
            data.append(
                {
                    'ranges': ['{}!{}{}:{}{}'.format(sheet_id, ALPHA[j], i + 1, ALPHA[j], i + 1)],
                    'style': style[i][j],
                }
            )
    req_body = {
        "data": data,
    }

    data = bytes(json.dumps(req_body), encoding='utf8')
    req = request.Request(url=url, data=data, headers=headers, method='PUT')
    try:
        response = request.urlopen(req)
    except Exception as e:
        print(e.read().decode())
        return ""
    rsp_body = response.read().decode('utf-8')
    rsp_dict = json.loads(rsp_body)
    return rsp_dict['code']


def get_list(spreadsheetToken):
    # 获取工作表列表
    url = "https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{}/metainfo".format(
        spreadsheetToken
    )
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Authorization": "Bearer " + get_tenant_access_token(),
    }
    req_body = {
        # "spreadsheetToken" : spreadsheetToken,
    }

    data = bytes(json.dumps(req_body), encoding='utf8')
    req = request.Request(url=url, data=data, headers=headers, method='GET')
    try:
        response = request.urlopen(req)
    except Exception as e:
        print(e.read().decode())
        return ""
    rsp_body = response.read().decode('utf-8')
    rsp_dict = json.loads(rsp_body)
    return rsp_dict['data']['sheets']


def add_sheet(spreadsheetToken, title=None):
    # 添加一个工作表
    if not title:
        title = time.strftime("%Y%m%d", time.localtime())
    url = "https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{}/sheets_batch_update".format(
        spreadsheetToken
    )
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Authorization": "Bearer " + read_user_access_token(),
    }
    req_body = {
        "requests": [
            {
                "addSheet": {
                    "properties": {
                        "title": title,
                        "index": 1000,  # 在最后面添加
                    }
                },
            }
        ]
    }

    data = bytes(json.dumps(req_body), encoding='utf8')
    req = request.Request(url=url, data=data, headers=headers, method='POST')
    try:
        response = request.urlopen(req)
    except Exception as e:
        print(e.read().decode())
        return ""
    rsp_body = response.read().decode('utf-8')
    rsp_dict = json.loads(rsp_body)
    if rsp_dict['code'] != 0:
        return rsp_dict['code']
    return rsp_dict['data']['replies'][0]['addSheet']['properties']['sheetId']


def copy_sheet(spreadsheetToken, sheet_id, title=None):
    # 复制一个工作表
    if not title:
        title = time.strftime("%Y%m%d", time.localtime())
    url = "https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{}/sheets_batch_update".format(
        spreadsheetToken
    )
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Authorization": "Bearer " + read_user_access_token(),
    }
    req_body = {
        "requests": [
            {"copySheet": {"source": {"sheetId": sheet_id}, "destination": {"title": title}}}
        ]
    }

    data = bytes(json.dumps(req_body), encoding='utf8')
    req = request.Request(url=url, data=data, headers=headers, method='POST')
    try:
        response = request.urlopen(req)
    except Exception as e:
        print(e.read().decode())
        return ""
    rsp_body = response.read().decode('utf-8')
    rsp_dict = json.loads(rsp_body)
    if rsp_dict['code'] != 0:
        return rsp_dict['code']
    return rsp_dict['data']['replies'][0]['copySheet']['properties']['sheetId']


def del_line(spreadsheetToken, sheet_id, row):
    # 在工作表中删除第row行
    url = "https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{}/dimension_range".format(
        spreadsheetToken
    )
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Authorization": "Bearer " + read_user_access_token(),
    }
    req_body = {
        "dimension": {
            "sheetId": sheet_id,
            "majorDimension": "ROWS",
            "startIndex": row,
            "endIndex": row,
        }
    }
    data = bytes(json.dumps(req_body), encoding='utf8')
    req = request.Request(url=url, data=data, headers=headers, method='DELETE')
    try:
        response = request.urlopen(req)
    except Exception as e:
        print(e.read().decode())
        return ""
    rsp_body = response.read().decode('utf-8')
    rsp_dict = json.loads(rsp_body)
    return rsp_dict['code']


def add_line(spreadsheetToken, sheet_id, row):  # 添加空行
    url = (
        "https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{}/insert_dimension_range".format(
            spreadsheetToken
        )
    )
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Authorization": "Bearer " + get_tenant_access_token(),
    }
    req_body = {
        "dimension": {
            "sheetId": sheet_id,
            "majorDimension": "ROWS",
            "startIndex": row,
            "endIndex": row + 1,
        },
        "inheritStyle": "AFTER",
    }
    data = bytes(json.dumps(req_body), encoding='utf8')
    req = request.Request(url=url, data=data, headers=headers, method='POST')
    try:
        response = request.urlopen(req)
    except Exception as e:
        print(e.read().decode())
        return ""
    rsp_body = response.read().decode('utf-8')
    rsp_dict = json.loads(rsp_body)
    # print(rsp_dict)
    return rsp_dict['code']


def insert_line(spreadsheetToken, sheet_id, row, data):
    # 覆盖内容或新增行覆盖
    add_line(spreadsheetToken, sheet_id, row)
    url = "https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{}/values_append".format(
        spreadsheetToken
    )
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Authorization": "Bearer " + get_tenant_access_token(),
    }
    range_ = '{}!{}{}:{}{}'.format(sheet_id, ALPHA[0], row, ALPHA[len(data) - 1], row)
    print(range_)
    req_body = {
        "valueRange": {
            "range": range_,
            "values": [data],
        }
    }
    data = bytes(json.dumps(req_body), encoding='utf8')
    req = request.Request(url=url, data=data, headers=headers, method='POST')
    try:
        response = request.urlopen(req)
    except Exception as e:
        print(e.read().decode())
        return ""
    rsp_body = response.read().decode('utf-8')
    rsp_dict = json.loads(rsp_body)
    # print(rsp_dict)
    return rsp_dict['code']

def set_cell_backColor(spreadsheetToken, sheet_id, row, col, backColor):
    range_ = '{}!{}{}:{}{}'.format(sheet_id, ALPHA[col], row + 1, ALPHA[col], row + 1)
    url = "https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{}/style".format(
        spreadsheetToken
    )
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Authorization": "Bearer " + read_user_access_token(),
    }
    req_body = {
        "appendStyle": {
            "range": range_,
            "style":{
                "backColor": backColor
            }
        },
    }
    data = bytes(json.dumps(req_body), encoding='utf8')
    req = request.Request(url=url, data=data, headers=headers, method='PUT')
    try:
        response = request.urlopen(req)
    except Exception as e:
        print(e.read().decode())
        return ""
    rsp_body = response.read().decode('utf-8')
    rsp_dict = json.loads(rsp_body)
    return rsp_dict['code']

def create_sheet(folder_token:str = "fldcnXicxNXs524fKSYEd0cGXFe"):
    url = "https://open.feishu.cn/open-apis/sheets/v3/spreadsheets"
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Authorization": "Bearer " + get_tenant_access_token(),
    }
    body = {
        "title": "onnx导出测试" + str(datetime.date.today()),
        "folder_token": folder_token
    }
    data = json.dumps(body)
    response = requests.post(url, data=data, headers=headers)
    response = response.json()
    print('sheet_url:', response['data']['spreadsheet']['url'])
    return response['data']['spreadsheet']['spreadsheet_token']

def get_root_folder():
    url = "https://open.feishu.cn/open-apis/drive/explorer/v2/root_folder/meta"
    headers = {
        "Authorization": "Bearer " + get_tenant_access_token(),   
        }
    response = requests.get(url, headers=headers)
    response = response.json()
    folder_token = response['data']['token']
    return folder_token

def give_permission(file_token):
    url = f"https://open.feishu.cn/open-apis/drive/v1/permissions/{file_token}/members"
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Authorization": "Bearer " + get_tenant_access_token(),
    } 
    params = {
        'type': 'sheet'
    }
    body = {
        'member_type': 'email',
        'member_id': 'linjie.7674@bytedance.com',
        'perm': 'full_access'
    }
    data = json.dumps(body)
    response = requests.post(url, headers=headers, params=params, data=data)
    response = response.json()
    return response
    

if __name__ == '__main__':
    # create_sheet()
    folder_token = get_root_folder()
    file_id = create_sheet(folder_token)
    give_permission(file_id)