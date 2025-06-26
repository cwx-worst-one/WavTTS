import datetime
import os
import glob
from enum import Enum
import requests
# from logger import logger
import uuid
import base64
import argparse
import pandas as pd

cycle_path_small_data_upload = "/sail/internal/data/file"
cycle_path_large_data_upload_init = "/sail/internal/data/large-file/init"
cycle_path_large_data_upload_chunk = "/sail/internal/data/large-file/chunk"
cycle_path_large_data_upload_confirm = "/sail/internal/data/large-file/confirm"
cycle_path_get_url = "/sail/internal/data/file/download-url"


class Host(Enum):
    CN = 1
    US = 2

    def url(self):
        if self.value == self.CN.value:
            return "https://sami.bytedance.net"
        if self.value == self.US.value:
            return "https://sami-useast2a.bytedance.net"
        raise Exception(f"unknown Host {self.value}")


class Client:
    __host = ""
    __user = ""
    __user_token = ""

    def __init__(self, host: Host, user):
        self.__host = host
        self.__user = user

    def do(self, path: str, params: dict):
        url = self.__host.url() + path

        headers = {
            "content-type": "application/json",
            "x-sail-user": self.__user,
        }

        ppe_env = os.getenv("SAIL_PPE_ENV")
        if ppe_env is not None:
            headers["x-tt-env"] = ppe_env
            headers["x-use-ppe"] = '1'

        # logger.debug(
        #     "post %s, header: %s, params: %s", url, headers.__str__(), params.__str__()
        # )
        # print(
        #     "post %s, header: %s, params: %s", url, headers.__str__(), params.__str__()
        # )

        resp = requests.post(url, json=params, headers=headers)

        # logger.info(
        #     "post %s, status code: %d, body: %s, headers: %s",
        #     url,
        #     resp.status_code,
        #     resp.text,
        #     resp.headers.__str__(),
        # )
        print(
            "post %s, status code: %d, body: %s, headers: %s",
            url,
            resp.status_code,
            resp.text,
            resp.headers.__str__(),
        )

        if resp.status_code != 200:
            # logger.error("status code is not 200")
            print("status code is not 200")
            raise Exception("status code is not 200")

        resp_dict = resp.json()
        if "biz_resp" not in resp_dict:
            # logger.error("cannot found key biz_resp field in resp")
            print("cannot found key biz_resp field in resp")
            raise Exception("cannot found key biz_resp field in resp")

        biz_resp = resp_dict["biz_resp"]
        code = biz_resp["code"]
        msg = biz_resp["msg"]
        if code != 20000000:
            # logger.error("biz code: %d, msg %s", code, msg)
            print("biz code: %d, msg %s", code, msg)
            raise Exception(f"biz_resp error {msg}")

        return resp_dict

# Returns url
def upload_data_and_get_public_url(
    host: Host,
    user: str,
    file_content: bytes,
    space: str,
    unique_file_name: str,
    expire_seconds: int,
):
    client = Client(host, user)
    upload_req = {
        "name": unique_file_name,
        "path": {"space": space, "space_ak": "temp_no_need", "v": "/"},
        "data": base64.b64encode(file_content).decode("utf-8"),
    }
    client.do(cycle_path_small_data_upload, upload_req)
    get_req = {
        "path": {"space": space, "space_ak": "temp_no_need", "v": unique_file_name},
        "expires": expire_seconds,
    }
    get_resp = client.do(cycle_path_get_url, get_req)
    if get_resp["url"] is None or get_resp["url"] == "":
        raise Exception("empty url")
    return get_resp["url"]


def upload_a_file(filename, space_name=None):

    if not space_name:
        space_name=datetime.datetime.now().date().strftime("%Y-%m-%d")
    fd = open(filename, "rb")
    content = fd.read()
    fd.close()
    _, format = os.path.splitext(filename)
    file_name = "my_file_" + uuid.uuid4().hex + format
    expires = 60*60*24*365*10       # 10 years
    url = upload_data_and_get_public_url(Host.CN, 'wangtuo.todd', content, space_name, file_name, expires)

    return url


def upload_a_folder(path_folder, filename_links=None, format=".wav"):

    if not filename_links:
        filename_links = os.path.join(path_folder, "uploaded_links.csv")
    os.makedirs(os.path.dirname(filename_links), exist_ok=True)
    if os.path.exists(filename_links):   # load existing info, to avoid repeating upload (suppose you fail in previous function call)
        df = pd.read_csv(filename_links, encoding='utf-8-sig', header=None)
        links = df.values.tolist()
    else:
        links = []

    filenames = glob.glob(os.path.join(path_folder, "*" + format))
    filenames.sort()
    for filename in filenames:
        names_already = [x[0] for x in links]
        name = os.path.splitext(os.path.basename(filename))[0]
        if name in names_already:
            print("Skip (file already uploaded: %s)" % name)
            continue
        url = upload_a_file(filename)
        links.append([name, url])
        df = pd.DataFrame(links)
        df.to_csv(filename_links, index=False, header=False, encoding='utf-8-sig')

    
def upload_a_folder_recursively_sorted(path_folder, filename_links=None, format='.wav'):

    folder_name = os.path.basename(path_folder)
    if not filename_links:
        filename_links = os.path.join(path_folder, folder_name+".csv")
    os.makedirs(os.path.dirname(filename_links), exist_ok=True)
    if os.path.exists(filename_links):
        df = pd.read_csv(filename_links, encoding='utf-8-sig', header=None)
        links = df.values.tolist()
    else:
        links = []

    filenames = glob.glob(os.path.join(path_folder, '*/*'+format), recursive=True)
    names = [os.path.basename(x) for x in filenames]
    idx = sorted(range(len(names)), key=lambda i: names[i])
    filenames = [filenames[i] for i in idx]

    for filename in filenames:
        names_already = [x[0] for x in links]
        name = os.path.splitext(os.path.basename(filename))[0]
        if name in names_already:
            print("Skip (file already uploaded: %s)" % name)
            continue
        url = upload_a_file(filename)
        links.append([name, url])
        df = pd.DataFrame(links)
        df.to_csv(filename_links, index=False, header=False, encoding='utf-8-sig')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default=None, type=str)

    args = parser.parse_args()
    filename = args.filename

    url = upload_a_file(filename)

    print ("=========================================")
    print ("uploaded url:")
    print (url)
    print ("=========================================")

    '''
    ------------------------------
    example 1: upload one file to easycycle and print the link

    from recipes.bigmusic.utils.upload_to_easycycle import upload_a_file
    url = upload_a_file('/mnt/bn/music-llm-nas-lq/bochen/results/20250325-114146_freeform_95k/20250325-114146_freeform_95k.mp4')
    print ('file as been uploaded as: ', url)

    ------------------------------
    example 2: upload the inference result folder to easycycle and output a csv file with 2 columns: index, link (sorted by index)

    from recipes.bigmusic.utils.upload_to_easycycle import upload_a_folder_recursively_sorted
    upload_a_folder_recursively_sorted('/mnt/bn/music-llm-nas-lq/bochen/results/20250325-114146_freeform_95k')
    '''