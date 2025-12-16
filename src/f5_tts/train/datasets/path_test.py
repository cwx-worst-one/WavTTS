# test_soundfile.py
import soundfile as sf
from pathlib import Path

test_file = Path('/mnt/bn/jdy-lq-5/chenwenxi/data/LibriTTS/train-clean-100/103/1241/103_1241_000000_000001.wav')

print("Testing soundfile with different inputs:")
print("=" * 50)

# 测试1：使用字符串
try:
    info1 = sf.info(str(test_file))
    print(f"✓ String path works: duration = {info1.duration:.2f}s")
except Exception as e1:
    print(f"✗ String path failed: {e1}")

# 测试2：使用 Path 对象
try:
    info2 = sf.info(test_file)
    print(f"✓ Path object works: duration = {info2.duration:.2f}s")
except Exception as e2:
    print(f"✗ Path object failed: {e2}")

# 测试3：直接读取音频数据
try:
    data, samplerate = sf.read(str(test_file))
    print(f"✓ sf.read works: shape={data.shape}, samplerate={samplerate}")
except Exception as e3:
    print(f"✗ sf.read failed: {e3}")

# 测试4：检查文件头
with open(test_file, 'rb') as f:
    header = f.read(44)  # WAV文件头通常是44字节
    print(f"\nFile header (first 44 bytes):")
    print(f"RIFF identifier: {header[0:4]}")
    print(f"File size: {int.from_bytes(header[4:8], 'little')}")
    print(f"WAVE identifier: {header[8:12]}")
    print(f"fmt identifier: {header[12:16]}")

# python /mnt/bn/jdy-lq-5/chenwenxi/code/F5_TTS_Wav/src/f5_tts/train/datasets/path_test.py