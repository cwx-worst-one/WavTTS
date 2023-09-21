# coding:utf-8


# noinspection PyPep8Naming, PyShadowingNames
class MetaData(object):
    access_key: str
    user_id: str
    task_id: str
    message_id: str
    model_id: str
    event_name: str
    version: str

    def __init__(self,
                 access_key: str = None,
                 user_id: str = None,
                 task_id: str = None,
                 message_id: str = None,
                 model_id: str = None,
                 event_name: str = None,
                 version: str = 'v1') -> None:
        ...


# noinspection PyPep8Naming, PyShadowingNames
class ChargeData(object):
    type: str
    amount: int

    def __init__(self, type: str = None, amount: int = None) -> None:
        ...


# noinspection PyPep8Naming, PyShadowingNames
class InvokeRequest(object):
    meta_data: MetaData
    data: str
    payload: str

    def __init__(self,
                 meta_data: MetaData = None,
                 data: str = None,
                 payload: str = None) -> None:
        ...


# noinspection PyPep8Naming, PyShadowingNames
class InvokeResponse(object):
    meta_data: MetaData
    status: int
    data: str
    payload: str
    charge_data: ChargeData
    status_code: int
    status_text: str

    def __init__(self,
                 meta_data: MetaData = None,
                 status: int = None,
                 data: str = None,
                 payload: str = None,
                 charge_data: ChargeData = None,
                 status_code: int = None,
                 status_text: str = None) -> None:
        ...


# noinspection PyPep8Naming, PyShadowingNames
class Service(object):
    def Invoke(self, req: InvokeRequest = None) -> InvokeResponse:
        ...
