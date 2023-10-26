import json
import logging
import time

import euler

from . import Handler

euler.install_thrift_import_hook()

from recipes.serving.idls.sami_app_thrift import InvokeRequest, InvokeResponse
from recipes.serving.utils.const import (
    STATUS_CODE_INVALID_PAYLOAD,
    STATUS_CODE_SERVER_FAILED_INVOKE,
    STATUS_TEXT_DEFAULT_HANDLER_SETUP_FAILED,
)


class DefaultHandler(Handler):

    app_name = None
    api_main = None
    preload_models = None
    inputs = None
    outputs = None

    def __init__(self, req: InvokeRequest) -> None:
        super().__init__(req)

    def __call__(self, ctx, audios=None) -> InvokeResponse:
        logging.info("***** invoke *****")
        start_time = time.time()
        if not DefaultHandler.api_main or not DefaultHandler.preload_models:
            logging.warnning(
                "default handler app {} setup failed", DefaultHandler.app_name
            )
            return super().get_resp(
                STATUS_CODE_SERVER_FAILED_INVOKE,
                STATUS_TEXT_DEFAULT_HANDLER_SETUP_FAILED,
            )

        if not super().parse_payload():
            logging.warnning("parse fail")
            return super().get_resp()

        # Prepare input args
        args = []
        for input in DefaultHandler.inputs:
            if input.source == "payload":
                input_value = ""
                if input.name in self.payload_req:
                    input_value = self.payload_req[input.name]

                logging.info(f"input param, name: {input.name}, value: {input_value}")
                args.append(input_value)
            elif input.source == "data":
                args.append(self.req.data)
            else:
                return super().get_resp(
                    STATUS_CODE_INVALID_PAYLOAD,
                    f"Illegal input source {input.source}",
                )

        # Call main func
        res = DefaultHandler.api_main(*args)

        # Prepare output data and payload
        resp_payload = {
            "process_time": time.time() - start_time,
            "message": f"app [{DefaultHandler.app_name}] using default handler done",
        }

        for i, output in enumerate(DefaultHandler.outputs):
            if output.destination == "data":
                self.resp_data = res[i]
            elif output.destination == "payload":
                if output.name == 'message':
                    if len(res[i]) > 0:
                        return super().get_resp(
                            STATUS_CODE_INVALID_PAYLOAD,
                            res[i],
                        )
                else:
                    resp_payload[output.name] = res[i]
            else:
                return super().get_resp(
                    STATUS_CODE_INVALID_PAYLOAD,
                    f"Illegal output destination {output.destination}",
                )

        self.resp_payload = json.dumps(resp_payload)
        return super().get_resp()

    @classmethod
    def load_context(self):
        DefaultHandler.preload_models()
