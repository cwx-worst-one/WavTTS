import re
from importlib.metadata import version


def get_s3a_version():
    return version("s3a")


def get_pybind11_func_args(func):
    func_def = getattr(func, "__doc__", None)
    if func_def:
        pattern = r"\((.*?)\)"
        match = re.findall(pattern, func_def)[0]
        return match.replace(" ", "").split(",")
    else:
        return []
