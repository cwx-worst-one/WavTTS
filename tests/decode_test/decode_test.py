'''decode_test start'''
import math
import subprocess

EPS = 0.04
STANDARD_CER = {'tele_general_test': 20.71}


def get_cer(log_str):
    '''get cer from log'''
    cer_pos = log_str.find('CER:')
    if cer_pos == -1:
        return -1
    cer = log_str[cer_pos + 4 : cer_pos + 20].split('%')[0]
    return float(cer)


def decode_test(cmd):
    '''dolphin decode_test'''
    status, ret = subprocess.getstatusoutput(cmd)
    print(ret)
    if status != 0:
        print(f"Error! fail to run {cmd} decode_test")
        raise RuntimeError("fail to run decode_test")
    cer = get_cer(ret)
    if cer == -1:
        print("Error! cer not found")
        raise RuntimeError("cer not found")
    if math.fabs(STANDARD_CER['tele_general_test'] - cer) > EPS:
        print(f"Error! Your cer is {cer}, std cer is {STANDARD_CER['tele_general_test']}.")
        raise RuntimeError("Cer accuracy error is too large!")
    print("decode_test passed")


if __name__ == '__main__':
    CMD = (
        'python3 -u train.py --config hdfs://haruna/home/'
        'byte_arnold_hl_speech_asr/user/huanglu.thu19/saved_models/dolphin/configs/'
        'data_selected/tel_rnnt_transfomer_80kh.py --train.remote_save_root '
        'hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huanglu.thu19/saved_models/dolphin/'
        ' --train.save_name rnnt_transformer_base_drop0.15_pool8 --train.resume_optimizer 0'
        ' --inference 1 --data.test_data_root hdfs://haruna/home/byte_arnold_hl_speech_asr/'
        'user/sunzhouyi/dataset/ --inference.test_sets tele_general_test_min'
    )
    decode_test(CMD)
