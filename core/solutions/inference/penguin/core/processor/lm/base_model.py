"""Lm base model"""
import logging
import os

import openfst_python as fst


class FST:
    """Fst model, including hotword fst type and sentence fst type currently."""

    def load(self, path):
        """Load model from path. Support local and hdfs."""
        if os.path.exists(path):
            # pylint:disable=no-member
            self.fst = fst.Fst.read(path)
        else:
            logging.fatal("load fst_path error")
            exit()

    def start(self):
        """Retrun start state of the fst."""
        start = -1
        if self.fst:
            start = self.fst.start()
        return start
