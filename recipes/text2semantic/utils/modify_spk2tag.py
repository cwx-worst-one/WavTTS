import sys, os
import json

in_spk2tag_path = sys.argv[1]
out_spk2tag_path = sys.argv[2]

pattern2tag = {
    "11labs": 3,
    "tts_en_Sbigspeech_P1": 1,
    "tts_en_Slibrivox-bc13-seg_P1": 1,
    "duibiao/Dacey_conversation_new": 3,
    "duibiao/Dacey_emotion": 3,
    "duibiao/DaceyGPT": 1,
    "duibiao/DaceyNew0801": 1,
    "duibiao/Dina": 1,
    "duibiao/jason_conversation": 3,
    "duibiao/jason_conversation_update": 3,
    "duibiao/jason_emotion": 3,
    "duibiao/jason_podcast": 1,
    "duibiao/lyf_0810": 1,
    "duibiao/M174_conversation_1002": 3,
    "duibiao/m174_conversation": 3,
    "duibiao/M191": 3,
    "duibiao/M525_conversation_1002": 3,
    "duibiao/maomao_audiobook": 1,
    "duibiao/maomao_conversation_update": 3,
    "duibiao/maomao_english": 3,
    "duibiao/maomao_general": 3,
    "duibiao/MimicJason": 3,
    "duibiao/MimicTaoziEn": 3,
    "duibiao/MimicTim": 3,
    "duibiao/MimicTim_short": 1,
    "duibiao/sarah_conversation_0928": 3,
    "duibiao/sarah_conversation": 3,
    "duibiao/sarah_emotion": 3,
    "duibiao/sarah_emotion_part2": 3,
    "duibiao/sarah_podcast": 3,
    "duibiao/Sherrie": 1,
    "duibiao/sinong_conversation": 3,
    "duibiao/taozi_1700": 1,
    "duibiao/taozi_chatgpt": 1,
    "duibiao/taozi_conversation": 3,
    "duibiao/taozi_emotion": 3,
    "duibiao/Tim_emotion": 3,
    "duibiao/Tim_emotion_rcd_0908": 3,
    "duibiao/Tim_normal": 1,
    "duibiao/wuxue_conversation": 3,
    "demo_speakers/guanxin_conversation": 3,
    "demo_speakers/mingbo_conversation": 3,
    "duibiao/DaceyNew0717": 1,
    "duibiao/Dacey_conversation": 1,
    "librilight": 2,
    "libritts_clean_460": 2,
    "fanqie": 2,
    "rp_part_0-49": 2,
    "rp_part_50-100": 2,
    "BC2013_mos3.9_sim0.0_snr7_rms-13_asr0.8_internal_1_5_10s": 2,
    "tts_zh_Sbigspeech_P1": 1,
    "rp_2900_speaker_0925": 3,
    "BC2013_mos3.9_sim0.0_snr7_rms-13_asr0.8_internal_1_gt_10s": 2,
    "novel003_longform_v1": 3,
    "bc2013_5_10s/None": 2,
    "duibiao/Angel_conversation": 3,
    "duibiao/Corey": 1,
    "duibiao/sarah_speech": 1,
    "duibiao/Tim_conversation": 3,
    "duibiao/maomao_conversation": 3,
    }

with open(in_spk2tag_path) as f:
    spk2tag = json.load(f)

for spk in spk2tag.keys():
    for pattern in pattern2tag.keys():
        if pattern in spk:
            spk2tag[spk] = pattern2tag[pattern]

with open(out_spk2tag_path, "w") as f_w:
    json.dump(spk2tag, f_w, ensure_ascii=False, indent=2)