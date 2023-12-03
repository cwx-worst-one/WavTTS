#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:bytedance

import asyncio
import base64
import json
import time
import uuid
import os

import websockets

async def tts_ws(payload, result_path):
    api_url = "ws://sami.bytedance.com/internal/api/v1/ws"
    task_id = str(uuid.uuid4())
    req = {
        "token": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MDQwMDAwNDksImlhdCI6MTcwMDE5ODQ1MSwiaXNzIjoiU0FNSSBNZXRhIiwidmVyc2lvbiI6InZvbGMtYXV0aC12MSIsImFjY291bnRfaWQiOjIxMDAwMTAxMTQsImFjY291bnRfbmFtZSI6IlNBTUlfUUFfUHJvZCIsImFwcF9pZCI6ODg4MDA4MTQsImFwcF9uYW1lIjoiU0FNSV9OT1ZBIiwiYXBwa2V5IjoiQVBzZ0dKQnNBaCIsInNlcnZpY2UiOiJzYW1pIiwic291cmNlIjoicGxhdGZvcm0iLCJyZWdpb24iOiJjbiJ9.Gqa5Ad1vv5J0T60f16L60x8pCTK_9ytSWyU6C6B2Ja7kzhM3Og7-o4Vt--BXvNcaGTca2Cw4T9NaEwhY3725Tg",
        "appkey": "APsgGJBsAh",
        "namespace": "TTS",
        "event": "StartTask",
        "payload": json.dumps(payload),
        # "task_id": task_id
    }
    # try:
    st = time.perf_counter()
    flag = 0
    result_data = open(result_path, "wb+")
    async with websockets.connect(api_url, ping_interval=None) as ws:
#     async with websockets.connect(api_url, ping_interval=None,extra_headers = {
        # 'X-TT-ENV':'your_ppe_env',  # 测试泳道
        # 'X-USE-PPE':1  
        # 先发送开始事件
#             }) as ws:
        await ws.send(json.dumps(req))
        # 然后发送该事件是否发送完成
        req["event"] = "FinishTask"
        first_package_time = None
        await ws.send(json.dumps(req))
        while True:
            res = await ws.recv()
            try:
                if isinstance(res, str):
                    print("receive text message, ", end="")
                    res_dict = json.loads(res)
                    if "data" in res_dict:
                        if flag == 0:
                            first_package_time = time.perf_counter() - st
                            flag = 1
                        result_data.write(base64.b64decode(res_dict["data"]))
                        # print(base64.b64decode(res_dict["data"]))
                        print(" data=byte[%d]" % len(res_dict["data"]), end="")
                    if "payload" in res_dict:
                        print(" payload=%s" % res_dict["payload"], end="")
                    print(" task_id=%s, event=%s status_code=%d status_text=%s" % (
                        res_dict["task_id"], res_dict["event"], res_dict["status_code"], res_dict["status_text"]))
                    if res_dict["status_code"] != 20000000:
                        print("task failed: ", res_dict)
                        await ws.close()
                        break
                    if res_dict["event"] == "TaskFinished":
                        await ws.close()
                        break
                else:
                    print("receive binary message, len=%d" % len(res))
                    result_data.write(res)
                    if flag == 0:
                        first_package_time = time.perf_counter() - st
                        flag = 1
                    # print(res)
            except Exception as e:
                print("exception", e)
                break
        if first_package_time is not None:
            print("首包时间：", first_package_time)
    result_data.close()


if __name__ == '__main__':

    result_dir = '/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha_bigtts_ref_enc/recipes/text2semantic/utils/get_online_wav/taozi/taozi_online'
    os.makedirs(result_dir, exist_ok=True)

    with open('/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/flow_tts_zh/meta.lst.listen_all_new_zh_select',  'r') as f:
        for line in f:
            # print(line)
            part = line.strip().split('|')
            name, text = part[0], part[1]
            print(name, text)

            result_path = os.path.join(result_dir, name + '.wav')
            # if os.path.exists(result_path):
            #     continue

            # 音频保存路径
            # result_path = "./output.wav"
            payload = {
                "text": text,
                "speaker": "zh_female_taozi_conversation_wvae_bigtts", # zh_female_xiaoqian zh_female_taozi
                "audio_config": {
                    "format": "wav",
                    "speech_rate": 0,
                    "enable_timestamp": False,
                    "sample_rate": 24000
                }}
            # for payload in pay:
            tasks = [asyncio.ensure_future(tts_ws(payload, result_path)) for i in range(1)]
            loop = asyncio.get_event_loop()
            loop.run_until_complete(asyncio.wait(tasks))
            time.sleep(1)
            exit()
            # exit(0)
