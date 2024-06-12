def get_s3a_version():
    from importlib.metadata import version

    return version("s3a")
