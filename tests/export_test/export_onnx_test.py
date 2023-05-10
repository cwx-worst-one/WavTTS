'''begin export test'''
# pylint: disable=line-too-long
import os
import sys

configs = {
    'asr': {
        'tel_rnnt_transfomer_80kh': '--config configs/asr/tel_rnnt_transfomer_80kh.py --solution.encoder_convert_stream 0',
        'tele_lstm_rnnt_200kh_witheos': '--config hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/saved_models/dolphin/configs/tele_200kh/tele_lstm_rnnt_200kh_witheos.py --solution.onnx_stack_frame 320',
        'child_zh_lstm_rnnt_ef_aug_1500h': '--config hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/dolphin/configs/child_zh_lstm_rnnt_ef_aug_1500h.py --solution.onnx_stack_frame 320',
        'child_en_rnnt_baseline.lrsche_manualsame': '--config hdfs://haruna/home/byte_arnold_hl_speech_asr/user/tangyu.yt/train_config/en_child/20210524_fuxian_baseline/child_en_rnnt_baseline.lrsche_manualsame.py --solution.onnx_stack_frame 320',
        'child_en_transformer_rnnt_baseline': '--config hdfs://haruna/home/byte_arnold_hl_speech_asr/user/chenjinkun/dolphin/configs/child_en_transformer_rnnt_baseline.py  --solution.onnx_stack_frame 320 --solution.encoder_convert_stream 0',
    },
    'sid': {
        'resnet101': '--config hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/dolphin_config/vox2_resnet101_fbank64.py --train.resume_hdfs_chkpt 0 --data.fbank_dim 64 --data.batch_class_num 16 --optimizer.weight_decay 1e-3 --optimizer_config.bmuf_config 0  --data.prefetch_worker_num 2 --train.amp_level O1 --solution.export_nodes utt_output',
        'ecapa': '--config hdfs://haruna/home/byte_arnold_hl_speech_asr/user/liuyi/sid/dolphin_config/vox2_ecapa_1024_amsoftmax_fbank80_spdaug_novad.py --train.resume_hdfs_chkpt 0 --data.fbank_dim 80 --data.batch_class_num 16 --optimizer.weight_decay 1e-3 --optimizer_config.bmuf_config 0  --data.prefetch_worker_num 2 --train.amp_level O1 --solution.export_nodes utt_output',
    },
}

HDFS = "hdfs://haruna/home/byte_arnold_hl_speech_asr/user/sunzhouyi/export_test"
CHKPT = " --data.max_batch_size 1000 --train.max_iters 200 --train.checkpoint_config.interval 200 --train.save_dir dir_name --train.save_name save_name --train.resume_progress 0"
EXPORT = " --export-onnx 1 --solution.onnx_graph_optimization 1 --train.resume_optimizer 0"


def hdfs_test(remote_path, mode='-e', retry=3):
    '''test hdfs path.
    Args:
        @remote_path: remote path, begin with 'hdfs://'
        @mode: -d  return 0 if @remote_path is a directory.
               -e  return 0 if @remote_path exists.
               -f  return 0 if @remote_path is a file.
               -s  return 0 if file @remote_path is greater than
                   zero bytes in size.
               -z  return 0 if file @remote_path is zero bytes in size,
                   else return 1.

    Return:
        0 if success.
        other if fail.
    '''
    assert mode in ('-e', '-d', '-f', '-s', '-z')
    cmd = 'hdfs dfs -test {} {}'.format(mode, remote_path)

    ret = -1
    while ret != 0 and retry > 0:
        ret = os.system(cmd)
        retry -= 1
    return ret


def get_path(config):
    '''get remote_save_root
    note: hdfs pathComponentName limit 255 characters
    '''
    return HDFS + config.replace('/', '_').replace(' ', '_').replace('hdfs:', '_').replace(
        '--', '/'
    )


def get_chkpt(config):
    '''get chkpt path'''
    return get_path(config) + "/dir_name/save_name/checkpoints/step_200.pth"


def save_chkpt(name, config):
    '''save chkpt'''
    cmd = 'python3 -u train.py {} {} --train.remote_save_root {}'.format(
        config, CHKPT, get_path(config)
    )
    print("start {} save_chkpt".format(name))
    status = os.system(cmd)
    if status != 0:
        print(f"Error! fail to run {cmd}")
        raise RuntimeError(f"fail to run {name} save_chkpt")


def test_export_onnx(name, config):
    '''test export onnx'''
    os.system('rm -rf ./*/*/*/checkpoints/')
    print("start {} test_export_onnx".format(name))
    cmd = (
        'python3 -u train.py {} {} --train.remote_save_root {} --train.resume_hdfs_chkpt {}'.format(
            config, EXPORT, get_path(config), get_chkpt(config)
        )
    )
    status = os.system(cmd)
    if status != 0:
        print(f"Error! fail to run {cmd}")
        raise RuntimeError(f"fail to run {name} export_onnx")

    # export use_non_combined_adaptive_softmax
    cmd += ' --solution.use_non_combined_adaptive_softmax 1'
    status = os.system(cmd)
    if status != 0:
        print(f"Error! fail to run {cmd}")
        raise RuntimeError(f"fail to run {name} export non_combined onnx")


def check_chkpt(config):
    '''check chkpt exists'''
    chkpt = get_chkpt(config)
    return hdfs_test(chkpt, '-f')


if __name__ == '__main__':
    # pylint: disable=invalid-name
    test_name = sys.argv[1]
    if test_name not in configs:
        print(f"fail to start {test_name} export onnx test")
        raise RuntimeError(f"{test_name} not in configs")
    print(f"start {test_name} export onnx test")

    test_config = configs[test_name]
    for key, value in test_config.items():
        if check_chkpt(value) != 0:
            save_chkpt(key, value)
        test_export_onnx(key, value)
