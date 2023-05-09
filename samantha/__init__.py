r"""Main package of SAMI AI."""

import logging.config
import os

logging.config.fileConfig(os.path.join(os.path.dirname(__file__), "logging.conf"))

__version__ = "0.3.0"
