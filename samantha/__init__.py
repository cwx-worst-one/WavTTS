r"""Main package of SAMI AI."""

import logging.config
import os

if not int(os.getenv("SAMANTHA_DISABLE_LOGGING_BASIC_CONFIG", 0)):
    logging.config.fileConfig(os.path.join(os.path.dirname(__file__), "logging.conf"))

__version__ = "0.3.0"
