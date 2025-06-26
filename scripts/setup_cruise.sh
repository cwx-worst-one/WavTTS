#!/bin/bash

set -x
# FIXME: Use `set -e`.

CUDA_VER=$(nvcc -V | grep 'Cuda compilation tools' | awk -F', ' '{print $2}' | awk '{print $2}')
echo "Using CUDA ${CUDA_VER}"

# keep it here for compatibility issue
pip3 install bytedance.trainingmetrics -i https://bytedpypi.byted.org/simple/

if [ -z "$CRUISE_ENABLE_STEP_CLIENT_TRACKING" ]
then
    echo "User did not set up CRUISE_ENABLE_STEP_CLIENT_TRACKING, do not install bytedkafka package"
else
    pip3 install bytedkafka==0.2.9
    echo "User set up CRUISE_ENABLE_STEP_CLIENT_TRACKING, install bytedkafka package"
fi

if [ -z "$MARIANA_OVERRIDE_CRUISE_VERSION" ]; then
    echo "MARIANA_OVERRIDE_CRUISE_VERSION not set, will not update cruise"
elif [ "$MARIANA_OVERRIDE_CRUISE_VERSION" == "scm" ]; then
    echo "MARIANA_OVERRIDE_CRUISE_VERSION is set to scm, will use SCM cruise"
    python3 /opt/tiger/mariana/setup_cruise.py
else
    echo "MARIANA_OVERRIDE_CRUISE_VERSION set, will update cruise to $MARIANA_OVERRIDE_CRUISE_VERSION"
    cd /opt/tiger;
    # only download cruise when it doesn't exist, this is to save unnecessary installation when robust training starts the worker.
    # In this case, cruise is already installed.
    if [ ! -f "cruise/data.aml.cruise_1.0.0.$MARIANA_OVERRIDE_CRUISE_VERSION.tar.gz" ]; then
      echo "Downloading data.aml.cruise_1.0.0.$MARIANA_OVERRIDE_CRUISE_VERSION.tar.gz..."
      rm -rf cruise;
      mkdir -p cruise && cd cruise;
      wget http://luban-source.byted.org/repository/scm/data.aml.cruise_1.0.0.$MARIANA_OVERRIDE_CRUISE_VERSION.tar.gz;
      tar -xf data.aml.cruise*.tar.gz;
      # redundant installation, please comment back if encounter any issues with setting up cruise package
      # pip3 install -e .[magnus];
      # Execute the python script to confirm dependencies are satisfied
      python3 /opt/tiger/mariana/setup_cruise.py
    else
      echo "File data.aml.cruise_1.0.0.$MARIANA_OVERRIDE_CRUISE_VERSION.tar.gz already exists. Skipping download."
    fi
fi

if [ -z "$MARIANA_OVERRIDE_TRANSFORMERS_VERSION" ]
then
    echo "MARIANA_OVERRIDE_TRANSFORMERS_VERSION not set, will not update transformers"
else
    echo "MARIANA_OVERRIDE_TRANSFORMERS_VERSION set, will update transformers to $MARIANA_OVERRIDE_TRANSFORMERS_VERSION"
    cd /opt/tiger
    wget http://luban-source.byted.org/repository/scm/lab.speech.transformers_$MARIANA_OVERRIDE_TRANSFORMERS_VERSION.tar.gz;
    mkdir -p transformers_ws && tar -xf lab.speech.transformers*.tar.gz -C transformers_ws
    cd transformers_ws
    pip3 install .
fi

if [ -z "${MARIANA_PANTHER_VERSION}" ]
then
    echo "MARIANA_PANTHER_VERSION not set, will not update panther"
else
    echo "MARIANA_PANTHER_VERSION set, will update panther to ${MARIANA_PANTHER_VERSION}"
    MARIANA_PANTHER_TMP=/tmp/panther_${MARIANA_PANTHER_VERSION}
    mkdir -p ${MARIANA_PANTHER_TMP}
    pushd ${MARIANA_PANTHER_TMP}
    wget http://luban-source.byted.org/repository/scm/lab.speech.panther_arnold_${MARIANA_PANTHER_VERSION}.tar.gz
    tar -zxf lab.speech.panther_arnold_${MARIANA_PANTHER_VERSION}.tar.gz
    sudo pip3 uninstall -y panther-gpu
    sudo pip3 uninstall -y panther-gpu
    pip3 uninstall -y panther-gpu
    pip3 uninstall -y panther-gpu # double check
    sudo -E pip3 install --no-deps *torch*/panther_gpu-*.whl
    popd
    rm -fr ${MARIANA_PANTHER_TMP}
fi

if [ -z "${MARIANA_FMHA_PLUS_VERSION}" ]
then
    echo "MARIANA_FMHA_PLUS_VERSION not set, will not update fmha_plus"
else
    echo "MARIANA_FMHA_PLUS_VERSION set, will update fmha_plus to ${MARIANA_FMHA_PLUS_VERSION}"
    # at least >= 1.0.0.7
    MARIANA_FMHA_TMP=/tmp/fmha_${MARIANA_FMHA_PLUS_VERSION}
    mkdir -p ${MARIANA_FMHA_TMP}
    pushd ${MARIANA_FMHA_TMP}
    wget http://luban-source.byted.org/repository/scm/data.speech.flash_attn_plus_${MARIANA_FMHA_PLUS_VERSION}.tar.gz
    sudo pip3 uninstall -y fmha_plus
    pip3 uninstall -y fmha_plus
    tar -zxf data.speech.flash_attn_plus_${MARIANA_FMHA_PLUS_VERSION}.tar.gz
    sudo -E pip3 install --no-deps fmha_plus-*.whl
    popd
    python3 -m fmha_plus.setup_lib --sudo
    rm -fr ${MARIANA_FMHA_TMP}
fi

if [ -z "$MARIANA_INSTALL_EXTRA_PIP_DEPS" ]; then
    echo "MARIANA_INSTALL_EXTRA_PIP_DEPS not set, will not install extra pip deps in setup_cruise.sh"
else
    echo "MARIANA_INSTALL_EXTRA_PIP_DEPS is deprecated, nothing will be installed!"
    exit 1
fi


if [ "${MARIANA_CONTEXT_PARALLEL_BACKEND_VERSION}" = "auto" ]; then
    echo "The MARIANA_CONTEXT_PARALLEL_BACKEND_VERSION is set to auto, context parallel uses default dist-attn version"
    addr=$(python3 -c "from mariana.models.utils.context_parallel_backend import DIST_ATTN_ADDR as v; print(v)" | tail -n 1)
    cd /opt/tiger;
    rm -rf dist-attn;
    mkdir -p dist-attn && cd dist-attn;
    echo $addr;
    wget $addr --progress=bar:force:noscroll;
    tar -xf data.aml.dist_attn*.tar.gz;
    pip3 install --user --force-reinstall --no-deps --no-build-isolation bytedance.dist_attn*.whl;
elif [ "${MARIANA_CONTEXT_PARALLEL_BACKEND_VERSION}" == "scm" ]; then
    echo "The MARIANA_CONTEXT_PARALLEL_BACKEND_VERSION is set to scm, context parallel uses SCM dist-attn version"
    pip3 install --user --force-reinstall --no-deps --no-build-isolation /opt/tiger/dist-attn/bytedance.dist_attn*.whl
elif [[ -v MARIANA_CONTEXT_PARALLEL_BACKEND_VERSION ]]; then
    repo=$(python3 -c "from mariana.models.utils.context_parallel_backend import SCM_REPO as v; print(v)" | tail -n 1)
    echo "The MARIANA_CONTEXT_PARALLEL_BACKEND_VERSION is set to '$MARIANA_CONTEXT_PARALLEL_BACKEND_VERSION'"
    # data/aml/dist_flash_sparse_attn
    echo "MARIANA_CONTEXT_PARALLEL_BACKEND_VERSION set, will update context parallel backend to $MARIANA_CONTEXT_PARALLEL_BACKEND_VERSION ($repo)"
    cd /opt/tiger;
    rm -rf dist-attn;
    mkdir -p dist-attn && cd dist-attn;
    wget https://luban-source.byted.org/repository/scm/data.aml.$repo'_'$MARIANA_CONTEXT_PARALLEL_BACKEND_VERSION.tar.gz;
    tar -xf data.aml.dist_attn*.tar.gz;
    pip3 install --user --force-reinstall --no-deps --no-build-isolation bytedance.dist_attn*.whl;
fi
