'''
logging module, the usage is the same as standard logging.
'''
import sys
import logging

initialized_logger = {}


def get_logger(name="dolphin", log_level=logging.ERROR):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    if name in initialized_logger:
        return initialized_logger[name]

    logger = logging.getLogger(name)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    handlers = [stream_handler]

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)

    # if logger.propagate is True, it will call dolphin::logging.info and
    # root::logging.info. so we set logger.propagate=False
    logger.propagate = False
    initialized_logger[name] = logger

    return logger


def info(msg, *args, name="dolphin", **kwargs):
    '''logger info'''
    logger = get_logger(name)
    logger.info(msg, *args, **kwargs)


def error(msg, *args, name="dolphin", **kwargs):
    '''logger error'''
    logger = get_logger(name)
    logger.error(msg, *args, **kwargs)


def warning(msg, *args, name="dolphin", **kwargs):
    '''logger warning'''
    logger = get_logger(name)
    logger.warning(msg, *args, **kwargs)


class LogLevelChange:
    '''
    Change log level for temporary
    '''

    def __init__(self, logger, new_level="INFO"):
        '''
        set the logger to new level
        '''
        self.logger = logger
        self.origin_level = logger.level
        self.new_level = new_level

    def __enter__(self):
        '''
        set log level
        '''
        self.logger.setLevel(self.new_level)

    def __exit__(self, exc_type, exc_value, tb):
        '''
        reset to original level
        '''
        self.logger.setLevel(self.origin_level)


def all_rank_info(msg, *args, name="dolphin", **kwargs):
    '''log with level info'''
    logger = get_logger(name)
    with LogLevelChange(logger):
        logger.info(msg, *args, **kwargs)


def all_rank_warning(msg, *args, name="dolphin", **kwargs):
    '''log with level info'''
    logger = get_logger(name)
    with LogLevelChange(logger, "WARNING"):
        logger.warning(msg, *args, **kwargs)
