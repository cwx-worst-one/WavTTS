import os


def getenv_int(env_var, default=1):
    return int(os.getenv(env_var, default))
