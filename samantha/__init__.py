r"""Main package of SAMI AI."""

import logging.config
import os
import sys

if not int(os.getenv("SAMANTHA_DISABLE_LOGGING_BASIC_CONFIG", 0)):
    logging.config.fileConfig(os.path.join(os.path.dirname(__file__), "logging.conf"))

__version__ = "0.3.0"

_mariana_dir = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "apps", "mariana")
)
if _mariana_dir not in sys.path and os.path.exists(_mariana_dir):
    sys.path.append(_mariana_dir)
del _mariana_dir
