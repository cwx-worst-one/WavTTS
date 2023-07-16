import os
import warnings


def set_envs(**kwargs):
    for key, value in kwargs.items():
        if key in os.environ:
            warnings.warn(
                f"{key} already in environ: {os.environ[key]}, "
                f"will overwrite it with {value}"
            )
        os.environ.update({key: str(value)})


def getenv_int(env_var, default=1):
    return int(os.getenv(env_var, default))
