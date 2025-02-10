#!/usr/bin/env python3
#
# Copyright 2024 ByteDance Inc. All rights reserved.
# Author: Gao Lu (gaolu.e820@bytedance.com)

import shutil
import sys
import os
import pkg_resources
import tempfile
import subprocess

_INSTALL_CMD_WHL = "pip3 install *.whl"
_INSTALL_CMD_DIST = "pip3 install ."

_ABBRS_ = {
    "panther": "lab/speech/panther_arnold",
    "bumi": "seed/speech/bumi",
    "triton": "seed/speech/triton",
    "lsdp": "seed/speech/lsdp",
}
_INSTALL_CMD = {
    "lab/speech/panther_arnold": "pip3 install *torch*/panther_gpu-*.whl",
    "seed/speech/bumi": _INSTALL_CMD_DIST,
    "seed/speech/triton": _INSTALL_CMD_WHL,
    "seed/speech/lsdp": _INSTALL_CMD_DIST,
}


def _get_package_version(package_name):
    try:
        version = pkg_resources.get_distribution(package_name).version
        print(f"Already installed {package_name}={version}")
    except pkg_resources.DistributionNotFound:
        version = None
    return version


def _install_python_package(name, version):
    cmd = _INSTALL_CMD.get(name, "")

    name = name.replace("/", ".")
    url = f"http://luban-source.byted.org/repository/scm/{name}_{version}.tar.gz"

    # Special case
    if cmd == _INSTALL_CMD_DIST:
        subprocess.check_call(f"pip3 install {url}", shell=True)
        return

    dir = tempfile.mkdtemp()
    try:
        print(f"Fetching package [{name}] from SCM ...")
        subprocess.check_call(f"wget -qO- {url} | tar -xz -C {dir}", shell=True)
        if not cmd:
            contains_whl = len([x for x in os.listdir(dir) if x.endswith(".whl")]) == 1
            cmd = _INSTALL_CMD_WHL if contains_whl else _INSTALL_CMD_DIST

        print(f"Installing package [{name}] using [{cmd}] ...")
        subprocess.check_call(f"cd {dir} && {cmd}", shell=True)
    finally:
        shutil.rmtree(dir)


name = sys.argv[1]
version = sys.argv[2]

installed_version = _get_package_version(name)

if version != installed_version:
    _install_python_package(_ABBRS_.get(name, name), version)
