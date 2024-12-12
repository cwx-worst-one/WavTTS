ARG REGION

FROM aliyun-va-hub.byted.org/compile/seed.speech.pytorch2:159c9e2f0fb158d293de72e25644d07b as aliyun_va
FROM use-hub.byted.org/compile/seed.speech.pytorch2:159c9e2f0fb158d293de72e25644d07b as us-east
FROM hub.tiktoke.org/compile/seed.speech.pytorch2:159c9e2f0fb158d293de72e25644d07b as us-east-red
FROM aliyun-sin-hub.byted.org/compile/seed.speech.pytorch2:159c9e2f0fb158d293de72e25644d07b as aliyun_sg
FROM hub.byted.org/compile/seed.speech.pytorch2:159c9e2f0fb158d293de72e25644d07b as china-north-lf

ENV http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
ENV https_proxy="http://sys-proxy-rd-relay.byted.org:8118"
ENV no_proxy="byted.org,anaconda.org"

FROM ${REGION}

# multi-stage build, env are independent; hdfs-stdenv need env $REGION;
ARG REGION

ENV PYTORCH_CUDA_ALLOC_CONF expandable_segments:True
ENV CUBLASLT_WORKSPACE_SIZE 32768
# Temporary solution, more investigation needed.
# ENV NCCL_PROTO ^LL128

ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH /opt/tiger/native_libhdfs/lib/native:$LD_LIBRARY_PATH
ENV ARNOLD_HDFS_NATIVE 1
ENV ARNOLD_HDFS_CELER 1
ENV INFSEC_HADOOP_ENABLED 1
ENV CPP_HDFS_CONF /opt/tiger/arnold/hdfs_client/conf/celer_us/core-site.xml:/opt/tiger/arnold/hdfs_client/conf/celer_us/hdfs-site.xml

ENV ARNOLD_SORT_IP 1
ENV NCCL_IB_QPS_PER_CONNECTION 5
ENV NCCL_IB_TIMEOUT 23
ENV SPEECH_USE_SYSTEM_TRITON 1
ENV PANTHER_PLIR_EXCLUDE_OPS batch_gemm_fwd,batch_gemm_bwd
ENV TORCH_NCCL_AVOID_RECORD_STREAMS 1
ENV TOKENIZERS_PARALLELISM false

ENV MASON_SKIP_LEGO_AUTO_PIP_INSTALL 1
ENV MASON_SKIP_BPEX_AUTO_PIP_INSTALL 1
ENV MASON_SKIP_MEGATRON_AUTO_PIP_INSTALL 1
ENV CRS_LOGGING_LEVEL INFO
ENV TORCH_NCCL_HIGH_PRIORITY 1

ARG CRUISE_VERSION=1.0.0.3641
ARG PANTHER_VERSION=1.7.14.411
ARG OPENFST_VERSION=1.0.0.8
ARG ASR_EVAL_TOOL_VERSION=1.0.0.125
# For torch 2.4
ARG SPEECH_EVALS_VERSION=1.0.0.34
ARG I18N_TEXT_FORMAT_VERSION=1.0.0.112
ARG S3A_VERSION=1.0.0.5
ARG BUMI_VERSION=1.7.0.31
ARG TRITON_VERSION=1.0.0.102
ARG MARIANA_FMHA_PLUS_VERSION=1.0.0.18
# https://github.com/facebookresearch/xformers.git:6425fd0
ARG XFORMERS_VERSION=1.0.0.15
# https://github.com/google-research/bleurt.git:cebe7e6
ARG BLEURT_VERSION=1.0.0.14
# https://github.com/NVIDIA/apex.git:23c1f86
ARG APEX_VERSION=1.0.0.12
# https://code.byted.org/security/bytedkms
ARG BYTEDKMS_VERSION=1.0.0.16
# libsox.so
ARG LIBSOX_VERSION=1.0.0.18

RUN apt-get update && \
    apt-get install \
        -yq --no-install-recommends \
        libc6 \
        libucx0 \
        libmkl-rt \
        cmake \
        libopenmpi-dev \
        openmpi-bin \
        openssh-client \
        openssh-server \
        netbase \
        patchelf \
        git \
        vim \
        krb5-user \
        libcairo2-dev \
        clang-format \
        libsndfile1 \
        ffmpeg \
        tmux \
        git-lfs \
        python3-tk \
        libsndfile-dev \
        apt-transport-https \
        libpam-krb5 \
        screen \
        zip \
        espeak \
        espeak-ng \
        fonts-arphic-ukai \
        sox \
        libsox-dev \
        iproute2 \
        elfutils && \
    cd /tmp && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && rm -f cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-nsight-compute-12-4 \
                   cuda-nsight-systems-12-4 && \
    rm -f /usr/lib/x86_64-linux-gnu/libcudnn_*static.a && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install \
        --no-cache-dir \
        --no-deps \
        pip==24.0 \
        pybind11==2.12.0

RUN pip3 install \
        --no-cache-dir \
        --no-deps \
        absl-py==2.1.0 \
        accelerate==0.34.2 \
        addict==2.4.0 \
        aiohappyeyeballs==2.4.0 \
        aiohttp==3.10.5 \
        aiosignal==1.3.1 \
        alabaster==0.7.16 \
        annotated-types==0.7.0 \
        antlr4-python3-runtime==4.9.3 \
        anyio==4.4.0 \
        asn1crypto==1.5.1 \
        astor==0.8.1 \
        asttokens==2.4.1 \
        astunparse==1.6.3 \
        attrs==19.3.0 \
        audioread==3.0.1 \
        babel==2.16.0 \
        backoff==2.2.1 \
        bibtexparser==2.0.0b7 \
        bidict==0.23.1 \
        bitarray==2.9.2 \
        black==24.2.0 \
        blessed==1.20.0 \
        blobfile==3.0.0 \
        boto3==1.35.14 \
        botocore==1.35.14 \
        braceexpand==0.1.7 \
        byted-dataloader==0.5.4 \
        byted-hdfs-io==0.3.13 \
        byted-iceberg==0.2.151 \
        byted-janus==0.1.6.post6 \
        byted-kms-encryption==0.0.5 \
        byted-kmsv2inner==0.1.14 \
        byted-omnistore==0.2.54 \
        byted-wandb==0.13.72 \
        byted_encrypted_hdfs==0.7.2 \
        bytedance-context==0.7.1 \
        bytedance-metrics==0.5.2 \
        bytedance.ckpt_io_metrics==0.0.21 \
        bytedance.easycycle==1.1.33 \
        bytedance.hdfs-stdenv==0.0.30 \
        bytedance.modelhub==0.0.78 \
        bytedance.servicediscovery==0.1.2 \
        bytedbackgrounds==0.0.6 \
        byteddatabus==1.0.6 \
        byteddps==0.1.2 \
        bytedenv==0.6.2 \
        bytedeuler==0.42.1 \
        bytedevent==1.5.0 \
        bytedfeather==0.2.0 \
        bytedkmsv2==0.10.50 \
        bytedlogger==0.15.2 \
        bytedlogid==0.2.1 \
        bytedmemfd==0.2 \
        bytedmerlin==0.0.4.dev2 \
        bytedmetrics==0.10.2 \
        bytedpymongo==2.0.5 \
        bytedredis==1.7.6 \
        bytedrh2==1.18.9a19 \
        bytedservicediscovery==0.17.4 \
        bytedtcc==1.4.5 \
        bytedtos==1.1.9 \
        bytedtrace==0.3.0 \
        bytedzti==1.0.11 \
        bytedztijwthelper==0.0.23 \
        bytedztispiffe==0.0.14 \
        cachetools==5.5.0 \
        certifi==2024.8.30 \
        cffi==1.15.1 \
        cfgv==3.4.0 \
        chardet==5.2.0 \
        charset-normalizer==3.3.2 \
        click==8.1.3 \
        clldutils==3.22.2 \
        cloudpickle==1.6.0 \
        cmake==3.26.3 \
        codetiming==1.4.0 \
        colorama==0.4.6 \
        coloredlogs==15.0.1 \
        colorednoise==2.1.0 \
        colorlog==6.8.2 \
        Command==0.1.0 \
        ConfigArgParse==1.5.3 \
        configparser==7.1.0 \
        contourpy==1.3.0 \
        coqpit==0.0.17 \
        coverage==7.6.1 \
        cryptography==39.0.2 \
        csvw==3.3.0 \
        cxxfilt==0.3.0 \
        cycler==0.12.1 \
        Cython==0.29.34 \
        datasets==2.5.1 \
        decorator==5.1.1 \
        deepspeed==0.15.1 \
        Deprecated==1.2.14 \
        dill==0.3.5.1 \
        distlib==0.3.8 \
        distro==1.9.0 \
        dlinfo==1.2.1 \
        dnspython==2.6.1 \
        docker-pycreds==0.4.0 \
        docstring_parser==0.16 \
        docutils==0.19 \
        dotted-dict==1.1.3 \
        easydict==1.13 \
        ecdsa==0.19.0 \
        editdistance==0.8.1 \
        einops==0.8.0 \
        einx==0.3.0 \
        emoji==2.12.1 \
        et-xmlfile==1.1.0 \
        eventlet==0.33.3 \
        executing==2.1.0 \
        fairseq==0.12.2 \
        ffmpeg-python==0.2.0 \
        filelock==3.15.4 \
        fire==0.6.0 \
        flake8==7.0.0 \
        flatbuffers==24.3.25 \
        fonttools==4.53.1 \
        frozendict==2.4.4 \
        frozenlist==1.4.1 \
        fsspec==2023.6.0 \
        future==0.18.3 \
        gast==0.6.0 \
        gevent==22.10.2 \
        gitdb==4.0.11 \
        GitPython==3.1.43 \
        google-auth==2.34.0 \
        google-auth-oauthlib==1.0.0 \
        google-pasta==0.2.0 \
        gpustat==1.1.1 \
        greenlet==3.0.3 \
        grpcio==1.66.1 \
        gunicorn==20.1.0 \
        h11==0.14.0 \
        h5py==3.11.0 \
        hjson==3.1.0 \
        httpcore==1.0.5 \
        httpx==0.27.2 \
        huggingface-hub==0.26.3 \
        humanfriendly==10.0 \
        hydra-core==1.3.2 \
        HyperPyYAML==1.2.0 \
        identify==2.6.0 \
        idna==3.8 \
        imagesize==1.4.1 \
        iniconfig==2.0.0 \
        intel-openmp==2023.2.4 \
        ipaddress==1.0.23 \
        ipython==8.27.0 \
        iso8601==1.0.0 \
        isodate==0.6.1 \
        isort==5.12.0 \
        jedi==0.19.1 \
        jieba==0.42.1 \
        Jinja2==3.1.4 \
        jiter==0.5.0 \
        jiwer==3.0.4 \
        jmespath==1.0.1 \
        joblib==1.4.2 \
        jsonargparse==4.14.1 \
        jsonpatch==1.33 \
        jsonpointer==3.0.0 \
        jsonschema==4.23.0 \
        jsonschema-specifications==2023.12.1 \
        julius==0.2.7 \
        kaldiio==2.18.0 \
        keras==3.5.0 \
        kiwisolver==1.4.7 \
        LAC==2.1.2 \
        langchain==0.2.16 \
        langchain-core==0.2.39 \
        langchain-text-splitters==0.2.4 \
        langdetect==1.0.9 \
        langsmith==0.1.118 \
        language-tags==1.2.0 \
        libclang==18.1.1 \
        librosa==0.9.2 \
        lightning-utilities==0.11.7 \
        llvmlite==0.43.0 \
        lxml==5.3.0 \
        lz4==4.3.3 \
        Markdown==3.7 \
        markdown-it-py==2.2.0 \
        MarkupSafe==2.1.5 \
        matplotlib==3.7.1 \
        matplotlib-inline==0.1.7 \
        mccabe==0.7.0 \
        mdurl==0.1.2 \
        mir-eval==0.7 \
        miscreant==0.3.0 \
        mkl==2023.1.0 \
        mkl-devel==2023.1.0 \
        mkl-include==2023.1.0 \
        ml-dtypes==0.4.0 \
        mmh3==4.1.0 \
        mmhash3==3.0.1 \
        mock==5.1.0 \
        mpi4py==4.0.0 \
        mpmath==1.3.0 \
        msgpack==1.0.8 \
        multidict==6.0.5 \
        multiprocess==0.70.13 \
        mypy==1.11.2 \
        mypy-extensions==1.0.0 \
        namex==0.0.8 \
        networkx==3.3 \
        ninja==1.11.1 \
        nltk==3.9.1 \
        nodeenv==1.9.1 \
        none==0.1.1 \
        numba==0.60.0 \
        numpy==1.24.4 \
        nvidia-ml-py==12.560.30 \
        oauthlib==3.2.2 \
        omegaconf==2.3.0 \
        onnx==1.16.2 \
        onnxruntime==1.19.2 \
        openai==1.44.1 \
        openpyxl==3.1.2 \
        opt-einsum==3.3.0 \
        optree==0.12.1 \
        orjson==3.10.7 \
        packaging==24.1 \
        paddlepaddle==2.5.2 \
        pandas==1.5.3 \
        parso==0.8.4 \
        path==17.0.0 \
        pathlib2==2.3.7.post1 \
        pathspec==0.12.1 \
        pathtools==0.1.2 \
        peft==0.12.0 \
        pexpect==4.9.0 \
        phonemizer==3.3.0 \
        pillow==10.4.0 \
        platformdirs==4.3.2 \
        pluggy==1.5.0 \
        ply==3.11 \
        pooch==1.8.2 \
        portalocker==2.10.1 \
        POT==0.9.4 \
        pre-commit==3.6.2 \
        pre-commit-hooks==4.5.0 \
        primePy==1.3 \
        promise==2.3 \
        prompt_toolkit==3.0.47 \
        protobuf==3.20.0 \
        psutil==5.9.5 \
        ptyprocess==0.7.0 \
        pure_eval==0.2.3 \
        py==1.11.0 \
        py-cpuinfo==9.0.0 \
        py-spy==0.3.14 \
        pyahocorasick==2.1.0 \
        pyarrow==12.0.0 \
        pyasn1==0.6.0 \
        pyasn1_modules==0.4.1 \
        pycairo==1.23.0 \
        pycodestyle==2.11.1 \
        pycparser==2.22 \
        pycryptodomex==3.20.0 \
        pydantic==2.9.2 \
        pydantic_core==2.23.2 \
        pydub==0.25.1 \
        pyflakes==3.2.0 \
        Pygments==2.18.0 \
        PyJWT==2.8.0 \
        pylatexenc==2.10 \
        pyope==0.2.2 \
        pyOpenSSL==23.2.0 \
        pyparsing==3.0.9 \
        pypinyin==0.48.0 \
        pyre-extensions==0.0.29 \
        PySoundFile==0.9.0.post1 \
        pytest==6.2.5 \
        pytest-cov==3.0.0 \
        pytest-datadir==1.3.1 \
        pytest-mock==3.8.2 \
        pytest-runner==6.0.0 \
        python-consul==1.1.0 \
        python-dateutil==2.9.0.post0 \
        python-engineio==4.9.1 \
        python-etcd==0.4.5 \
        python-jose==3.3.0 \
        python-socketio==5.11.4 \
        pytorch-lightning==2.4.0 \
        pytz==2022.5 \
        pyvad==0.2.0 \
        pyworld==0.3.4 \
        PyYAML==6.0 \
        pyzstd==0.16.1 \
        rapidfuzz==3.9.7 \
        rdflib==7.0.0 \
        redis==3.5.3 \
        referencing==0.35.1 \
        regex==2024.7.24 \
        requests==2.31.0 \
        requests-oauthlib==2.0.0 \
        resampy==0.4.3 \
        responses==0.18.0 \
        retry==0.9.2 \
        rfc3986==1.5.0 \
        rich==13.3.5 \
        rotary-embedding-torch==0.8.3 \
        rouge==1.0.1 \
        rpds-py==0.20.0 \
        rsa==4.9 \
        ruamel.yaml==0.17.24 \
        ruamel.yaml.clib==0.2.8 \
        s3transfer==0.10.2 \
        sacrebleu==2.4.3 \
        safetensors==0.4.5 \
        schedule==1.2.2 \
        scikit-learn==1.2.2 \
        scipy==1.10.1 \
        segments==2.2.1 \
        sentencepiece==0.1.99 \
        sentry-sdk==2.14.0 \
        setproctitle==1.3.3 \
        shortuuid==1.0.13 \
        simple-websocket==1.0.0 \
        six==1.16.0 \
        smmap==5.0.1 \
        sniffio==1.3.1 \
        snowballstemmer==2.2.0 \
        soft-moe-pytorch==0.1.8 \
        soundfile==0.12.1 \
        sox==1.4.1 \
        soxbindings==1.2.3 \
        soxr==0.5.0.post1  \
        speechbrain==1.0.1 \
        Sphinx==5.3.0 \
        sphinxcontrib-applehelp==2.0.0 \
        sphinxcontrib-devhelp==2.0.0 \
        sphinxcontrib-htmlhelp==2.1.0 \
        sphinxcontrib-jsmath==1.0.1 \
        sphinxcontrib-qthelp==2.0.0 \
        sphinxcontrib-serializinghtml==2.0.0 \
        sphinxcontrib-websupport==2.0.0 \
        SQLAlchemy==2.0.34 \
        stack-data==0.6.3 \
        streamz==0.6.4 \
        subword-nmt==0.3.8 \
        sympy==1.13.2 \
        tabulate==0.9.0 \
        tbb==2021.13.1 \
        tenacity==8.5.0 \
        tensorboard==2.17.1 \
        tensorboard-data-server==0.7.2 \
        tensordict==0.6.0 \
        tensorflow==2.17.0 \
        tensorflow-io==0.37.1 \
        tensorflow-io-gcs-filesystem==0.37.1 \
        termcolor==2.4.0 \
        tf-slim==1.1.0 \
        threadpoolctl==3.5.0 \
        thriftpy2==0.4.16 \
        tiktoken==0.7.0 \
        timm==1.0.9 \
        tokenizers==0.15.2 \
        toml==0.10.2 \
        tomli==2.0.1 \
        toolz==0.12.1 \
        torch-complex==0.4.4 \
        torch-pitch-shift==1.2.4 \
        torch-stft==0.1.4 \
        torchaudio==2.4.0+cu124 \
        torchaudio-augmentations==0.2.4 \
        torchlibrosa==0.1.0 \
        torchmetrics==1.4.1 \
        torchvision==0.19.0+cu124 \
        tornado==6.4.1 \
        tox==3.28.0 \
        tqdm==4.65.0 \
        trainer==0.0.36 \
        traitlets==5.14.3 \
        transformers==4.37.0 \
        typing==3.7.4.3 \
        typing-inspect==0.9.0 \
        typing_extensions==4.12.2 \
        tzdata==2024.1 \
        universal_pathlib==0.2.3 \
        uritemplate==4.1.1 \
        urllib3==1.26.20 \
        vector-quantize-pytorch==1.17.3 \
        virtualenv==20.26.4 \
        watchdog==5.0.3 \
        wavaugment==0.2 \
        wcwidth==0.2.13 \
        webdataset==0.2.48 \
        webrtcvad==2.0.10 \
        websocket-client==1.8.0 \
        Werkzeug==3.0.4 \
        wget==3.2 \
        wrapt==1.16.0 \
        wsproto==1.2.0 \
        xxhash==3.5.0 \
        yamllint==1.26.3 \
        yapf==0.33.0 \
        yarl==1.10.0 \
        zhconv==1.4.3 \
        zhon==1.1.5 \
        zict==3.0.0 \
        zope.event==5.0 \
        zope.interface==7.0.3 \
        zstandard==0.19.0


# 2. install falconclaw && panther && dataloader
# install bytedkms==3.2.10, bytedkms must installed before cruise.
RUN \
    mkdir -p /tmp/py_bytedkms && \
    cd /tmp/py_bytedkms && \
    wget http://luban-source.byted.org/repository/scm/seed.speech.scm_packer_$BYTEDKMS_VERSION.tar.gz && \
    tar -zxf seed.speech.scm_packer_$BYTEDKMS_VERSION.tar.gz && \
    pip3 install --no-cache-dir --no-deps bytedkms*.whl && \
    rm -rf /tmp/py_bytedkms

RUN pip3 install --no-cache-dir --no-deps --use-pep517 \
        http://luban-source.byted.org/repository/scm/data.aml.cruise_$CRUISE_VERSION.tar.gz

RUN pip3 install --no-cache-dir --no-deps \
        http://luban-source.byted.org/repository/scm/lab.speech.openfst_python_$OPENFST_VERSION.tar.gz

RUN pip3 install --no-cache-dir --no-deps \
        http://luban-source.byted.org/repository/scm/seed.speech.bumi_$BUMI_VERSION.tar.gz

RUN mkdir -p /tmp/py_pkg.panther && \
    cd /tmp/py_pkg.panther && \
    wget http://luban-source.byted.org/repository/scm/lab.speech.panther_arnold_$PANTHER_VERSION.tar.gz && \
    tar -xvf lab.speech.panther_arnold_$PANTHER_VERSION.tar.gz && \
    pip3 install --no-cache-dir --no-deps *torch*/panther_gpu*.whl && \
    rm -rf /tmp/* /root/.cache

RUN pip3 uninstall -y s3a \
    && wget http://luban-source.byted.org/repository/scm/lab_audio.seed.s3a_torch24_$S3A_VERSION.tar.gz \
    && mkdir tmp.s3a \
    && tar -xvf lab_audio.seed.s3a_torch24_$S3A_VERSION.tar.gz -C tmp.s3a \
    && pip3 install --no-cache-dir --no-deps tmp.s3a/s3a-$S3A_VERSION-cp311-cp311-linux_x86_64.whl \
    && rm -fr tmp.s3a lab_audio.seed.s3a_torch24_$S3A_VERSION.tar.gz

# 3. install asr_eval_tool
RUN \
    mkdir -p /tmp/py_pkg.asr_eval_tool && \
    cd /tmp/py_pkg.asr_eval_tool && \
    wget http://luban-source.byted.org/repository/scm/lab.speech.asr_eval_tool_$ASR_EVAL_TOOL_VERSION.tar.gz && \
    tar -xvf lab.speech.asr_eval_tool_$ASR_EVAL_TOOL_VERSION.tar.gz && \
    rm lab.speech.asr_eval_tool_$ASR_EVAL_TOOL_VERSION.tar.gz && \
    cd src/I18N_text_format && \
    wget http://luban-source.byted.org/repository/scm/lab.speech.I18N_text_format_$I18N_TEXT_FORMAT_VERSION.tar.gz && \
    tar -xvf lab.speech.I18N_text_format_$I18N_TEXT_FORMAT_VERSION.tar.gz && \
    rm lab.speech.I18N_text_format_$I18N_TEXT_FORMAT_VERSION.tar.gz && \
    touch __init__.py && \
    cd ../.. && \
    python3 setup.py bdist_wheel && \
    pip3 install --no-cache-dir --no-deps --force-reinstall dist/asr_eval_tool-$ASR_EVAL_TOOL_VERSION-py3-none-any.whl && \
    rm -rf /tmp/py_pkg.asr_eval_tool

# install speech_evals
RUN \
    mkdir -p /tmp/py_pkg.speech_evals && \
    cd /tmp/py_pkg.speech_evals && \
    wget http://luban-source.byted.org/repository/scm/lab.speech.speech_evals_$SPEECH_EVALS_VERSION.tar.gz && \
    tar -xvf lab.speech.speech_evals_$SPEECH_EVALS_VERSION.tar.gz && \
    rm lab.speech.speech_evals_$SPEECH_EVALS_VERSION.tar.gz && \
    cd speech_evals/lib/process/I18N_text_format && \
    wget http://luban-source.byted.org/repository/scm/lab.speech.I18N_text_format_$I18N_TEXT_FORMAT_VERSION.tar.gz && \
    tar -xvf lab.speech.I18N_text_format_$I18N_TEXT_FORMAT_VERSION.tar.gz && \
    rm lab.speech.I18N_text_format_$I18N_TEXT_FORMAT_VERSION.tar.gz && \
    touch __init__.py && \
    cd /tmp/py_pkg.speech_evals && \
    python3 setup.py bdist_wheel && \
    pip3 install --no-cache-dir --no-deps --force-reinstall dist/speech_evals-$SPEECH_EVALS_VERSION-py3-none-any.whl && \
    rm -rf /tmp/py_pkg.speech_evals

# install flash_attn_plus
RUN \
    mkdir -p /tmp/py_mariana_fmha_plus && \
    cd /tmp/py_mariana_fmha_plus && \
    wget http://luban-source.byted.org/repository/scm/data.speech.flash_attn_plus_$MARIANA_FMHA_PLUS_VERSION.tar.gz && \
    tar -zxf data.speech.flash_attn_plus_$MARIANA_FMHA_PLUS_VERSION.tar.gz && \
    pip3 install --no-cache-dir --no-deps fmha_plus*.whl && \
    rm -rf /tmp/py_mariana_fmha_plus

# install xformers
RUN mkdir -p /tmp/py_xformers && \
    cd /tmp/py_xformers && \
    wget http://luban-source.byted.org/repository/scm/seed.speech.scm_packer_$XFORMERS_VERSION.tar.gz && \
    tar -xvzf seed.speech.scm_packer_$XFORMERS_VERSION.tar.gz && \
    pip3 install --no-cache-dir --no-deps xformers*.whl && \
    rm -rf /tmp/py_xformers

# 4. add tools and env
# 4.1 HDFS env
ENV LIBHDFS_OPTS "-Xms4g -Xmx16g -Dhadoop.root.logger=${HADOOP_ROOT_LOGGER:-ERROR,console}"
# 4.2 ssh config
RUN mkdir -p /var/run/sshd && \
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
# 4.3 for kinit
ADD ./scripts/krb5.conf /etc/krb5.conf

# Install bleurt.
RUN mkdir -p /tmp/py_bleurt && \
    cd /tmp/py_bleurt && \
    wget http://luban-source.byted.org/repository/scm/seed.speech.scm_packer_$BLEURT_VERSION.tar.gz && \
    tar -xvzf seed.speech.scm_packer_$BLEURT_VERSION.tar.gz && \
    pip3 install --no-cache-dir --no-deps BLEURT*.whl && \
    rm -rf /tmp/py_bleurt

# Add apex
RUN mkdir -p /tmp/py_apex && \
    cd /tmp/py_apex && \
    wget http://luban-source.byted.org/repository/scm/seed.speech.scm_packer_$APEX_VERSION.tar.gz && \
    tar -xvzf seed.speech.scm_packer_$APEX_VERSION.tar.gz && \
    pip3 install --no-cache-dir --no-deps apex*.whl && \
    rm -rf /tmp/py_apex

# Install triton 3.0 which well performs on H800
RUN mkdir -p /tmp/py_triton && \
    cd /tmp/py_triton && \
    wget http://luban-source.byted.org/repository/scm/seed.speech.triton_$TRITON_VERSION.tar.gz && \
    tar -xvf seed.speech.triton_$TRITON_VERSION.tar.gz && \
    pip3 uninstall -y pytorch-triton && \
    pip3 uninstall -y triton && \
    pip3 install --no-cache-dir --no-deps triton*.whl && \
    rm -rf /tmp/py_triton /root/.cache

# # Fix libsox.so issue.
# RUN \
#     mkdir -p /tmp/libsox && \
#     cd /tmp/libsox && \
#     wget http://luban-source.byted.org/repository/scm/seed.speech.scm_packer_$LIBSOX_VERSION.tar.gz && \
#     tar -zxf seed.speech.scm_packer_$LIBSOX_VERSION.tar.gz && \
#     rm /usr/lib/x86_64-linux-gnu/libsox.so && \
#     mv libsox.so /usr/lib/x86_64-linux-gnu/libsox.so && \
#     rm -rf /tmp/libsox

# Samantha-specific settings.
COPY ./scripts/samantha.bashrc /etc/samantha.bashrc
ENV BASH_ENV=/etc/samantha.bashrc

# hack for pl 2.2 requires wandb>=0.12.10
COPY ./scripts/setup_wandb_311.sh .
RUN bash ./setup_wandb_311.sh

# Make debugging easier.
COPY ./scripts/scm_install.py /usr/bin/scm_install
RUN chmod +x /usr/bin/scm_install
