import euler

from .idl import sami_thrift


class SGC:
    # SGC aka sami gateway client
    _client = None

    @classmethod
    def make_client(cls):
        if cls._client is None:
            cls._client = euler.Client(
                sami_thrift.SamiService,
                "sd://lab.sami.gateway?cluster=release_thrift",
                timeout=1200,
            )
        return cls._client


client = SGC.make_client()
