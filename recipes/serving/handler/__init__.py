import json
import logging
import os

import euler

euler.install_thrift_import_hook()
from recipes.serving.idls.sami_app_thrift import ChargeData, InvokeRequest, InvokeResponse

STATUS_CODE_OK = 20000000
STATUS_TEXT_OK = "OK"
STATUS_CODE_INVALID_PAYLOAD = 40000020
STATUS_TEXT_INVALID_PAYLOAD = "InvalidPayload"


class Handler:
    def __init__(self, req: InvokeRequest) -> None:
        logging.info("PROCESS_ID in Handler.init(): " + str(os.getpid()))
        self.req: InvokeRequest = req
        self.resp: InvokeResponse = None
        self.status_text: str = STATUS_TEXT_OK
        self.status_code: int = STATUS_CODE_OK
        self.resp_payload: str = None
        self.resp_data: bytes = None
        self.resp_charge_Data: ChargeData = None
        self.req_param_url: str = None
        self.payload_req = {}

    def check_health(self) -> bool:
        return (
            self.status_code == STATUS_TEXT_OK
            and self.status_text == STATUS_TEXT_OK
        )

    def parse_payload(self) -> bool:
        logging.info(
            "begin with payload: %s"
            % (
                self.req.payload
                if len(self.req.payload) < 100
                else self.req.payload[:100] + "...}"
            )
            if self.req.payload is not None
            else "None"
        )
        if self.req.payload != None and len(self.req.payload) > 0:
            try:
                self.payload_req = json.loads(self.req.payload)
                return True
            except:
                self.status_code = STATUS_CODE_INVALID_PAYLOAD
                self.status_text = STATUS_TEXT_INVALID_PAYLOAD
                return False

    def get_req_params(self) -> bool:
        if "url" in self.payload_req.keys():
            self.req_param_url = self.payload_req["url"]

    def get_resp(
        self, status_code: int = None, status_text: str = None
    ) -> InvokeResponse:
        ret = InvokeResponse(
            meta_data=self.req.meta_data,
            charge_data=self.resp_charge_Data,
            status=200,
            status_code=self.status_code,
            status_text=self.status_text,
            payload=self.resp_payload,
            data=self.resp_data,
        )
        if status_code is not None:
            ret.status_code = status_code
        if status_text is not None:
            ret.status_text = status_text
        return ret

    def __call__(self, ctx) -> InvokeResponse:
        pass
