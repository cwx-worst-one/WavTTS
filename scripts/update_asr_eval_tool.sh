
# Get version info of asr_eval_tool from https://cloud.bytedance.net/scm/detail/21975/versions
# Get version info of I18N_text_format from https://cloud.bytedance.net/scm/detail/119792/versions

# Latest version of asr_eval_tool and I18N_text_format.
# This will be updated at dolphin maintainence if needed.
ASR_EVAL_TOOL_VERSION=1.0.0.66
I18N_TEXT_FORMAT_VERSION=1.0.0.105

case "$1" in
    -h|--help|?)
    echo "Usage: bash update_asr_eval_tool.sh arg1 arg2"
    echo "       arg1: asr_eval_tool_version, default $ASR_EVAL_TOOL_VERSION"
    echo "       arg2: I18N_text_format_version, default $I18N_TEXT_FORMAT_VERSION"
    echo "If arg1 and arg2 not available, latest version will be used"
    exit 0
;;
esac

if [ $# == 2 ]
then
    ASR_EVAL_TOOL_VERSION=$1
    I18N_TEXT_FORMAT_VERSION=$2
    echo "asr_eval_tool: $ASR_EVAL_TOOL_VERSION"
    echo "I18N_text_format: $I18N_TEXT_FORMAT_VERSION"
fi

# ASR_EVAL_TOOL_VERSION=$1
# I18N_TEXT_FORMAT_VERSION=$2
SCM_URL_PREFIX='http://luban-source.byted.org/repository/scm'

# install asr_eval_tool
pip3 uninstall -y asr-eval-tool
pip3 uninstall -y asr-eval-tool

dir=/tmp/py_pkg.asr_eval_tool
rm -rf $dir
mkdir $dir

cd $dir && \
    wget ${SCM_URL_PREFIX}/lab.speech.asr_eval_tool_$ASR_EVAL_TOOL_VERSION.tar.gz && \
    tar -xvf lab.speech.asr_eval_tool_$ASR_EVAL_TOOL_VERSION.tar.gz && \
    rm lab.speech.asr_eval_tool_$ASR_EVAL_TOOL_VERSION.tar.gz && \
    cd src/I18N_text_format && \
    wget ${SCM_URL_PREFIX}/lab.speech.I18N_text_format_$I18N_TEXT_FORMAT_VERSION.tar.gz && \
    tar -xvf lab.speech.I18N_text_format_$I18N_TEXT_FORMAT_VERSION.tar.gz && \
    rm lab.speech.I18N_text_format_$I18N_TEXT_FORMAT_VERSION.tar.gz && \
    cd ../.. && \
    python3 setup.py bdist_wheel && \
    pip3 install --force-reinstall dist/asr_eval_tool-$ASR_EVAL_TOOL_VERSION-py3-none-any.whl

rm -rf $dir
