from typing import List, Dict, Any, Optional
import os
import json
import subprocess

from pydantic import BaseModel
from bytedrh2.http_client import authenticate, _get_rh2_client

current_dir = os.path.dirname(os.path.abspath(__file__))


class ClusterPreset(BaseModel):
    """
    clusterId:
        17: cloudnative-lq
        22: cloudnative-lf
        20: cloudnative-hl
    groupId:
        530: seed_speech_bigmusic
        615: seed_speech_bigmusic_h800
    """
    cluster_id: int
    group_id: int
    quota_pool: str
    bytenas_volumes: List[Dict[str, Any]]


class ResourcePreset(BaseModel):
    cpu: int
    gpu: int
    gpuv: Optional[str]
    memory: str
    num_workers: int


class RepoPreset(BaseModel):
    repo_name: str
    repo_dir: str
    mnt_dir: str
    image_url: str


class ArnoldSubmitter:
    """Reference:
    https://bytedance.larkoffice.com/wiki/Zxo7wa5gNig6Fak0gWZc6j7jnSh
    """
    payload_template_path = os.path.join(current_dir, "rh2template.json")
    def __init__(self):
        authenticate(host='rh2.bytedance.net')
        self.client = _get_rh2_client()
        with open(ArnoldSubmitter.payload_template_path) as fori:
            self.payload = json.load(fori)
    
    def setup_repo(self, repo_preset: RepoPreset):
        branch, sha = ArnoldSubmitter.get_git_info(repo_preset.repo_dir)
        self.payload["jobDefVersion"]["gitRepo"]["repoName"] = repo_preset.repo_name
        self.payload["jobDefVersion"]["gitRepo"]["mnt"] = repo_preset.mnt_dir
        self.payload["jobDefVersion"]["gitRepo"]["branchName"] = branch
        self.payload["jobDefVersion"]["gitRepo"]["commitSha"] = sha
        self.payload["jobDefVersion"]["imageMeta"]["imageUrl"] = repo_preset.image_url
        return self

    def setup_resource(self, resource_preset: ResourcePreset):
        self.payload["jobRunParams"]["resource"]["arnoldConfig"]["roles"][0] = {
            "name": "worker",
            "num": resource_preset.num_workers,
            "ports": 1,
            "cpu": resource_preset.cpu,
            "gpu": resource_preset.gpu,
            "gpuv": "NONE" if resource_preset.gpuv is None else resource_preset.gpuv,
            "memory": int(ArnoldSubmitter.convert_to_bytes(resource_preset.memory) / 1024 / 1024),
        }
        return self

    def setup_cluster(self, cluster_preset: ClusterPreset):
        self.payload["jobRunParams"]["resource"]["arnoldConfig"]["clusterId"] = cluster_preset.cluster_id
        self.payload["jobRunParams"]["resource"]["arnoldConfig"]["groupIds"] = [cluster_preset.group_id]
        self.payload["jobRunParams"]["resource"]["arnoldConfig"]["quotaPool"] = cluster_preset.quota_pool
        self.payload["jobRunParams"]["resource"]["arnoldConfig"]["bytenasVolumes"] = cluster_preset.bytenas_volumes
        return self
    
    def setup_caption(self, exp_id: str):
        self.payload["caption"] = exp_id
        return self

    def setup_cmd(self, cmd: str):
        self.payload["jobRunParams"]["entrypointFullScript"] = cmd
        self.payload["jobDefVersion"]["entrypointFullScript"] = cmd
        return self

    def setup_envs(self, env_var_dict):
        self.payload["jobRunParams"]["envsList"].update(env_var_dict)
        return self

    def submit_job(self):
        result = self.client.http_call(method='POST',
            path='api/v1/job_run/launch',
            data=json.dumps(self.payload)
        )
        if result[0] == 200:
            job_id = json.loads(result[1])["jobRunId"]
            print(f"Success. https://ml.bytedance.net/development/instance/jobs/{job_id}")
        else:
            print(f"Failed. {result}")

    @staticmethod
    def get_git_info(repo_dir):
        # Make sure the directory exists
        if not os.path.isdir(repo_dir):
            return None, None

        # Change the current working directory to the repo_dir
        original_directory = os.getcwd()
        os.chdir(repo_dir)

        try:
            # Get the current branch name
            branch_name = subprocess.check_output(['git', 'branch', '--show-current']).decode('utf-8').strip()

            # Get the latest commit SHA
            commit_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()

            return branch_name, commit_sha
        except subprocess.CalledProcessError:
            # The directory is not a Git repository
            return None, None
        finally:
            # Change back to the original directory
            os.chdir(original_directory)
    
    @staticmethod
    def convert_to_bytes(value: str) -> int:
        unit = value[-2:].lower()
        val = int(value[:-2])
        if unit == "kb":
            return val * 1024
        elif unit == "mb":
            return val * 1024**2
        elif unit == "gb":
            return val * 1024**3
        elif unit == "tb":
            return val * 1024**4
        else:
            raise ValueError(f"Invalid unit: {unit}")