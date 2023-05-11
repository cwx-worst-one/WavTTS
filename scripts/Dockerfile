# Load base image
FROM hub.byted.org/base/lab.pytorch2:2.0.0.post2

RUN sudo apt -q update && sudo apt -q install -y libsndfile1 ffmpeg tmux git-lfs python3-tk

RUN pip3 install -q --upgrade pip setuptools -i https://bytedpypi.byted.org/simple

COPY scripts/requirements.txt .

RUN pip3 config set global.trusted-host bytedpypi.byted.org
RUN pip3 config set global.index-url https://bytedpypi.byted.org/simple
RUN pip3 install -r requirements.txt

# install latest online version of s3a
RUN pip3 install --no-build-isolation --no-cache-dir https://luban-source.byted.org/repository/scm/api/v1/download_latest/?name=lab/audio/sami_ai_acceleration

# install apex (compile from source)
ENV https_proxy=http://sys-proxy-rd-relay.byted.org:8118
ENV http_proxy=http://sys-proxy-rd-relay.byted.org:8118

RUN git clone https://github.com/NVIDIA/apex && pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex