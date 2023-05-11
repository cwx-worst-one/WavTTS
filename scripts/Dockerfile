# Load base image
FROM hub.byted.org/base/lab.pytorch2:2.0.0.post2

RUN sudo apt -q update && sudo apt -q install -y libsndfile1 ffmpeg tmux git-lfs python3-tk

RUN pip3 install -q --upgrade pip setuptools -i https://bytedpypi.byted.org/simple

COPY scripts/requirements.txt .

RUN pip3 config set global.trusted-host bytedpypi.byted.org
RUN pip3 config set global.index-url https://bytedpypi.byted.org/simple
RUN pip3 install -r requirements.txt

# install latest online version of s3a
RUN pip3 install --no-build-isolation --no-cache-dir https://luban-source.byted.org/repository/scm/api/v1/download_latest/?name=lab_audio/seed/s3a

# install panther
RUN mkdir panther_package && cd panther_package && wget http://luban-source.byted.org/repository/scm/lab.speech.panther_arnold_1.7.5.35.tar.gz  \
    && tar -xzvf lab.speech.panther_arnold_1.7.5.35.tar.gz  && pip3 install torch/panther_gpu-1.7.5-cp39-cp39-linux_x86_64.whl && cd ..

ENV https_proxy=http://sys-proxy-rd-relay.byted.org:8118
ENV http_proxy=http://sys-proxy-rd-relay.byted.org:8118

# install apex (compile from source)
RUN git clone https://github.com/NVIDIA/apex && cd apex && git checkout 85e9eddece9d4ac72b48c2407f8162f2173e1bf4 && cd .. \
 &&pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

# install xformers
RUN git clone https://github.com/facebookresearch/xformers.git && cd xformers && git checkout 6425fd0cacb1a6579aa2f0c4a570b737cb10e9c3 \
    && git submodule update --init --recursive && cd .. \
    && XFORMERS_DISABLE_FLASH_ATTN=1 pip3 install -v --disable-pip-version-check --no-cache-dir ./xformers