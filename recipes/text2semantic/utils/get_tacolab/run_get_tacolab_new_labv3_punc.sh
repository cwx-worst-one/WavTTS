stage=$1

### zh
if [ ${stage} -eq 1 ];then
    # git clone git@code.byted.org:lab-audio/sami_tts_api.git -b zhcn_punkt_recover $YOUR_DIR
    text_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/compare_with_vallex/text.infer.en2zh
    out_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/compare_with_vallex/infer_lab_newv3_punc

    cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/sami_tts_api_zhcn_punkt_recover # $YOUR_DIR
    pip3 install bytedeuler --index-url=http://pypi.byted.org/simple/pypi/+simple --trusted-host=pypi.byted.org
    pip3 install ./sami_tts_api/
    num_job=2
    num=`wc -l $text_path | awk -F' ' '{print $1}'`
    num_per_thread=`expr $num / $num_job + 1`
    split -l $num_per_thread -d -a2 $text_path $text_path.
    export LD_LIBRARY_PATH=scm/current/libs:/usr/local/cuda-11.7/targets/x86_64-linux/lib/:/opt/nvidia/nsight-systems/2022.1.3/target-linux-x64/:$LD_LIBRARY_PATH
    for i in {00..1};do
        python3 users/dy/run_label_v3_to_recover_punkt.py \
            -i $text_path.$i \
            -o $out_dir \
            -v V3 &
    done
    wait
    rm -f $text_path.*
    cd -
fi


### en
if [ ${stage} -eq 2 ];then
    # git clone git@code.byted.org:lab-audio/sami_tts_api.git -b en_punkt_recover $YOUR_DIR
    text_path=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/compare_with_vallex/text.prompt.zh2en
    out_dir=/mnt/bn/huangzhiying-nas-speech2speech-volume1/code/bigtts_testset/compare_with_vallex/prompt_lab_newv3_punc

    cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/sami_tts_api # $YOUR_DIR
    pip3 install bytedeuler --index-url=http://pypi.byted.org/simple/pypi/+simple --trusted-host=pypi.byted.org
    pip3 install ./sami_tts_api/
    num_job=32
    num=`wc -l $text_path | awk -F' ' '{print $1}'`
    num_per_thread=`expr $num / $num_job + 1`
    split -l $num_per_thread -d -a2 $text_path $text_path.

    export LD_LIBRARY_PATH=scm/current/libs:/usr/local/cuda-11.7/targets/x86_64-linux/lib/:/opt/nvidia/nsight-systems/2022.1.3/target-linux-x64/:$LD_LIBRARY_PATH
    for x in {00..31};do
        python3 users/dy/run_label_v3_to_recover_punkt.py \
            -i $text_path.$x \
            -o $out_dir \
            -v V3 &
    done
    wait
    rm -f $text_path.*
fi
