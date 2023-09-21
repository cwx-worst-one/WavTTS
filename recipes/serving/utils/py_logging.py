import logging
import logging.config
import logging.handlers
import os
from pathlib import Path

import bytedlogger


def init_logging_config():
    bytedlogger.config_default()

    runtime_logdir = os.environ.get("RUNTIME_LOGDIR")
    if runtime_logdir:
        app_log_file = os.path.join(runtime_logdir, "app", "py_demo_server.log")
    else:
        app_log_file = os.path.join(Path(__file__).parent, "runtimelogs", "py_demo_server.log")

    # Create log file if not exist
    log_file = Path(app_log_file)
    os.makedirs(log_file.parent, exist_ok=True)
    log_file.touch(exist_ok=True)

    appFileLogHandler = logging.handlers.TimedRotatingFileHandler(app_log_file, when='midnight', backupCount=10)
    pyLogFormater = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    appFileLogHandler.setFormatter(pyLogFormater)
    root_log = logging.getLogger()
    root_log.addHandler(appFileLogHandler)


if __name__ == '__main__':
    init_logging_config()
