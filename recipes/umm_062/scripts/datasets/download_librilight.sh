#!/bin/bash

# Required disk space: ~8 Terabytes

sudo apt-get install -y axel


# By combining these subsets, one can construct the 3 splits described in the Libri-Light paper:
# unlab-60k : small + medium + large
# unlab-6k : small + medium
# unlab-600 : small

axel https://dl.fbaipublicfiles.com/librilight/data/small.tar
pv small.tar | tar -x
# md5: c49207eb86a8e8ac895561c37232041e


axel https://dl.fbaipublicfiles.com/librilight/data/medium.tar
pv medium.tar | tar -x
# md5: c75e7ac62471bfbf2db77528d62a9b74

axel https://dl.fbaipublicfiles.com/librilight/data/large.tar
pv large.tar | tar -x
# md5: 4dfbac018f50b99797ece101fc9f0c30
