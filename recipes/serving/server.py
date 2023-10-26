import euler
import logging
import atexit
euler.install_thrift_import_hook()

from recipes.serving.idls.sami_app_thrift import Service, InvokeRequest, InvokeResponse
from recipes.serving.handler.default import DefaultHandler
from recipes.serving.utils.setup import setup_app
from recipes.serving.utils.const import STATUS_CODE_SERVER_FAILED_INVOKE


PSM = "seed.speech.bigtts_sinong_conversation"
PORT = 8888
handler_map = {}


def register_by_cli(psm, port):
    import subprocess
    result = subprocess.run(['/opt/tiger/consul_deploy/bin/go/sd', 'up', psm, str(port), '--dual-stack',
                             '--tags', '{"env":"prod","weight":"10", "cluster":"default"}'],
                            capture_output=True, text=True)
    logging.info(f'sd up output: {result.stdout}')


def exit_handler():
    import subprocess
    subprocess.run(['/opt/tiger/consul_deploy/bin/go/sd', 'down', PSM, str(PORT)], capture_output=True, text=True)
    logging.info(f"deregister {PSM} {PORT}")


# 注册 exit_handler() 函数
atexit.register(exit_handler)


# Load non-forksafe models here
def load_handler_contexts():
    method = 'load_context'
    for name, handler_cls in handler_map.items():
        if hasattr(handler_cls, method) and callable(getattr(handler_cls, method)):
            logging.info(f'{name}: load_context')
            getattr(handler_cls, method)()


server = euler.Server(Service, post_fork_callback=load_handler_contexts)

register_by_cli(PSM, PORT)
logging.info("=========== register sd success ===========")


@server.register('Invoke')
def Invoke(ctx, req: InvokeRequest):
    if not req.meta_data or not req.meta_data.model_id:
        return InvokeResponse(status_code=STATUS_CODE_SERVER_FAILED_INVOKE, status_text="NoModelID")
    model_id = req.meta_data.model_id
    if model_id not in handler_map.keys():
        logging.info("The model_id: " + model_id + " not in handler_map.keys(): " + str(handler_map.keys()))
        return InvokeResponse(meta_data=req.meta_data, status_code=STATUS_CODE_SERVER_FAILED_INVOKE,
                              status_text="NotSupportModelID")
    else:
        logging.info(
            f'calling {model_id} invoke request: len(data)={len(req.data) if req.data else 0}, req_payload={(req.payload if len(req.payload) < 100 else req.payload[:100] + "...}") if req.payload is not None else "None"}), req_meta_data={req.meta_data}')
        handler = handler_map[model_id]
        resp:InvokeResponse = handler(req)(ctx=ctx)
        logging.info(
            f'{model_id} invoke response: len(data)={len(resp.data) if len(resp.data) == 0 else 0}, resp_payload={(resp.payload if len(resp.payload) < 100 else resp.payload[:100] + "...}") if resp.payload is not None else "None"}, status_code={resp.status_code}, status_text={resp.status_text}')
        return resp



app = "BigTTS"
handler_map[app] = DefaultHandler
setup_app(app, DefaultHandler)

