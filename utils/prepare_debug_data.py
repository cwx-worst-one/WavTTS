import os
import random
import shutil
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# ================= 配置路径 =================
SRC_ROOT = "/mnt/bn/jdy-lq-5/chenwenxi/data/LibriTTS/train-clean-100"
DST_ROOT_BASE = "/mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav/data/LibriTTS"

# 目标文件夹名称
CROSS_DIR_NAME = "train-clean-100-cross-sentence"
SAME_DIR_NAME = "train-clean-100-same-sentence"

SAMPLE_COUNT = 1127

# 时长限制配置
MIN_DUR = 2.5
MAX_DUR = 9.5

# ================= 工具函数 =================

def get_audio_info(wav_path):
    """
    读取音频时长和对应的Normalized文本
    返回: duration (float), text (str)
    """
    wav_path_str = str(wav_path)
    
    # 读取时长
    info = sf.info(wav_path_str)
    duration = info.duration  # 保持float
    
    # 读取文本
    txt_path = wav_path_str.replace(".wav", ".normalized.txt")
    
    if not os.path.exists(txt_path):
        print(f"Warning: Text file not found for {wav_path_str}")
        return duration, ""
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        
    return duration, text

def copy_file(src_path, dst_root):
    """
    将文件复制到目标目录，保持 spk_id/chapter_id 的结构
    """
    parts = src_path.name.split('_')
    if len(parts) >= 2:
        spk_id = parts[0]
        chapter_id = parts[1]
    else:
        spk_id = src_path.parent.parent.name
        chapter_id = src_path.parent.name

    target_dir = os.path.join(dst_root, spk_id, chapter_id)
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, src_path.name)
    if not os.path.exists(target_path):
        shutil.copy2(src_path, target_path)
    
    return spk_id, chapter_id, src_path.stem

# ================= 主逻辑 =================

def main():
    print(f"Scanning source directory: {SRC_ROOT} ...")
    src_path = Path(SRC_ROOT)
    
    # 1. 扫描所有文件
    all_possible_wavs = list(src_path.rglob("*.wav"))
    print(f"Total files found: {len(all_possible_wavs)}")

    # 2. 预处理：过滤时长 (2s - 10s) 并按 SPK 归类
    print(f"Filtering audio by duration ({MIN_DUR}s - {MAX_DUR}s)...")
    
    spk_to_wavs = defaultdict(list)
    valid_wavs_flat = []
    
    for wav in tqdm(all_possible_wavs):
        try:
            # 这里的 info 读取非常快，如果不读整个wav数据
            info = sf.info(str(wav))
            if MIN_DUR <= info.duration <= MAX_DUR:
                spk_id = wav.name.split('_')[0]
                spk_to_wavs[spk_id].append(wav)
                valid_wavs_flat.append(wav)
        except Exception as e:
            print(f"Error reading {wav}: {e}")
            continue

    # 过滤掉无法组成pair的说话人
    valid_spks_for_cross = [s for s, wavs in spk_to_wavs.items() if len(wavs) >= 2]
    
    print(f"Valid wavs within duration: {len(valid_wavs_flat)}")
    print(f"Valid speakers for cross-sentence: {len(valid_spks_for_cross)}")
    
    if len(valid_wavs_flat) < SAMPLE_COUNT:
        raise ValueError(f"Not enough audio files found! Need {SAMPLE_COUNT}")

    # -------------------------------------------------
    # 任务 1 & 2: Cross-Sentence (Pair) - 唯一 Target 补丁
    # -------------------------------------------------
    print("\nProcessing Cross-Sentence Dataset (Unique Targets)...")
    dst_cross = os.path.join(DST_ROOT_BASE, CROSS_DIR_NAME)
    meta_cross_path = os.path.join(DST_ROOT_BASE, f"{CROSS_DIR_NAME}.meta.lst")
    
    cross_pairs = []
    used_gen_paths = set() # [补丁] 用于记录已经作为target使用过的音频
    
    # 使用进度条，循环直到凑够数量
    pbar = tqdm(total=SAMPLE_COUNT)
    
    while len(cross_pairs) < SAMPLE_COUNT:
        # 1. 随机选一个说话人
        spk = random.choice(valid_spks_for_cross)
        wavs = spk_to_wavs[spk]
        
        # 2. 筛选出该说话人名下，还没被当做 gen 使用过的音频
        # 注意：这里我们只关心 gen 是否重复。ref 可以重复使用，也可以和别人的 gen 重复，这不影响。
        available_gens = [w for w in wavs if str(w) not in used_gen_paths]
        
        if not available_gens:
            # 如果这个说话人的所有音频都已经在之前的轮次中被用作 gen 了，就跳过
            continue
            
        # 3. 选定 gen
        gen_wav = random.choice(available_gens)
        
        # 4. 选定 ref (ref 必须是同一个人，但不能是 gen_wav 本身)
        # ref 可以是之前用过的，也可以是没用过的，只要不等于当前的 gen_wav 即可
        available_refs = [w for w in wavs if w != gen_wav]
        
        if not available_refs:
            # 理论上 len>=2 不会进这里，以防万一
            continue
            
        ref_wav = random.choice(available_refs)
        
        # 5. 记录与添加
        used_gen_paths.add(str(gen_wav))
        cross_pairs.append((ref_wav, gen_wav))
        
        pbar.update(1)
        
    pbar.close()
        
    lines_cross = []
    print(f"Copying files and generating meta for Cross-Sentence to {dst_cross}...")
    
    for ref_wav, gen_wav in tqdm(cross_pairs):
        copy_file(ref_wav, dst_cross)
        copy_file(gen_wav, dst_cross)
        
        ref_dur, ref_txt = get_audio_info(ref_wav)
        gen_dur, gen_txt = get_audio_info(gen_wav)
        
        ref_utt = ref_wav.stem
        gen_utt = gen_wav.stem
        
        # 保留3位小数
        line = f"{ref_utt}\t{ref_dur:.3f}\t{ref_txt}\t{gen_utt}\t{gen_dur:.3f}\t{gen_txt}"
        lines_cross.append(line)
        
    with open(meta_cross_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_cross))
    print(f"Done. Meta saved to {meta_cross_path}")

    # -------------------------------------------------
    # 任务 3: Same-Sentence (Reconstruction)
    # -------------------------------------------------
    print("\nProcessing Same-Sentence Dataset...")
    dst_same = os.path.join(DST_ROOT_BASE, SAME_DIR_NAME)
    meta_same_path = os.path.join(DST_ROOT_BASE, f"{SAME_DIR_NAME}.meta.lst")
    
    # 这里的 random.sample 本身就是无放回抽样，保证了唯一性
    same_wavs = random.sample(valid_wavs_flat, SAMPLE_COUNT)
    
    lines_same = []
    print(f"Copying files and generating meta for Same-Sentence to {dst_same}...")
    
    for wav in tqdm(same_wavs):
        copy_file(wav, dst_same)
        
        dur, txt = get_audio_info(wav)
        utt = wav.stem
        
        # Ref 和 Gen 完全一致
        line = f"{utt}\t{dur:.3f}\t{txt}\t{utt}\t{dur:.3f}\t{txt}"
        lines_same.append(line)
        
    with open(meta_same_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_same))
    print(f"Done. Meta saved to {meta_same_path}")

if __name__ == "__main__":
    main()