import os

import thriftpy2

base_thrift = thriftpy2.load(os.path.join(os.path.dirname(__file__), "base.thrift"))

sami_thrift = thriftpy2.load(os.path.join(os.path.dirname(__file__), "sami.thrift"))
