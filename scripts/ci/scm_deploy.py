import os
import re
import sys
import time

import requests

SCM_REPO_ID = "61223"
SCM_USER = os.environ.get("SCM_USER")
SCM_PASSWORD = os.environ.get("SCM_PASSWORD")
BRANCH_NAME = os.environ.get("CI_COMMIT_REF_NAME", "develop")
TAG_NAME = os.environ.get("CI_COMMIT_TAG", None)
VERSION_TYPE = "online" if BRANCH_NAME == "master" and TAG_NAME is not None else "test"

CREATE_URL = "https://scm.byted.org/api/v2/versions/cicd_create/"
QUERY_URL = "https://scm.byted.org/api/v2/versions/{}/cicd_poll/"

def create_build():
    ''' start SCM build '''

    payload = {
        "repo_id": SCM_REPO_ID,
        "create_user": SCM_USER,
        "type": VERSION_TYPE,
        "branch_name": BRANCH_NAME
    }


    if BRANCH_NAME == "master" and TAG_NAME is not None:
        # tag should be like "v1.8", and scm version would be "1.1.8.0"
        if re.match(r"^v\d.\d+$", TAG_NAME):
            payload["version"] = f"1.{TAG_NAME[1:]}.0"

    print(payload)

    response = requests.post(
        url=CREATE_URL,
        auth=(SCM_USER, SCM_PASSWORD),
        json=payload
    )

    result = response.json()["data"]
    print("Create result: ", result)

    return result

def check_build(version_id):
    ''' poll build state '''

    build_state = "running"

    while build_state == "running":
        time.sleep(20)
        query_resp = requests.get(url=QUERY_URL.format(version_id))
        query_result = query_resp.json()["data"]
        build_state = query_result["state"]
        if build_state != "running":
            print("Build result: ", query_result)

    return build_state


if __name__ == "__main__":
    create_result = create_build()
    version_id = create_result["context"]["version_id"]
    version = create_result["context"]["version_version"]

    build_result = check_build(version_id)

    if build_result == "passed":
        with open(".build_version", 'w') as f:
            f.write(version)
    else:
        sys.exit(1)
