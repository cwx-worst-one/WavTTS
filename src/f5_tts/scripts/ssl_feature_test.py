import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModel

# ================= 配置参数 =================
# 模型路径 (可以是HuggingFace ID或本地路径)
SSL_MODEL_PATH = "/inspire/hdd/global_user/chenxie-25019/wenxichen/models/hubert/models--facebook--hubert-large-ll60k/snapshots/ff022d095678a2995f3c49bab18a96a9e553f782"  # hubert-large-ll60k, hubert-base-ls960
# 提取哪一层的特征 (-1 表示最后一层, 0-N 表示特定隐藏层)
LAYER_INDEX = -1 
# 模拟音频采样率
SAMPLE_RATE = 16000
# 模拟音频时长 (秒)
DURATION_SEC = 5
# 设备选择
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================

def main():
    print(f"--- Environment: {DEVICE} ---")
    print(f"Loading SSL model: {SSL_MODEL_PATH} ...")
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(SSL_MODEL_PATH)
        model = AutoModel.from_pretrained(SSL_MODEL_PATH).eval().to(DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    print("Model loaded successfully.")

    # Shape: (num_samples,)
    num_samples = SAMPLE_RATE * DURATION_SEC
    # 生成 -1 到 1 之间的随机浮点数
    dummy_wav = np.random.uniform(-1.0, 1.0, num_samples).astype(np.float32)
    print(f"Generated dummy audio -> Shape: {dummy_wav.shape}, Duration: {DURATION_SEC}s")

    # 3. 预处理 (Feature Extractor)
    # 将 raw waveform 转换为模型输入的 input_values
    inputs = feature_extractor(
        dummy_wav, 
        sampling_rate=SAMPLE_RATE, 
        return_tensors="pt"
    ).input_values.to(DEVICE)
    
    print(f"Model input shape (batch, samples): {inputs.shape}")

    # 4. 模型前向推理
    print("Running inference...")
    with torch.no_grad():
        # output_hidden_states=True 允许我们访问中间层
        outputs = model(inputs, output_hidden_states=True)

    # 5. 提取指定层的特征
    if LAYER_INDEX == -1:
        # last_hidden_state
        features = outputs.last_hidden_state
        layer_name = "Last Hidden State"
    else:
        # 特定层
        # hidden_states 是一个 tuple，包含了每一层的输出
        if 0 <= LAYER_INDEX < len(outputs.hidden_states):
            features = outputs.hidden_states[LAYER_INDEX]
            layer_name = f"Hidden Layer {LAYER_INDEX}"
        else:
            print(f"Layer index {LAYER_INDEX} out of bounds. Using last_hidden_state.")
            features = outputs.last_hidden_state
            layer_name = "Last Hidden State (Fallback)"

    # 6. 后处理与打印结果
    # 移除 batch 维度: (1, Seq, Dim) -> (Seq, Dim)
    features = features.squeeze(0)
    
    print("-" * 30)
    print(f"Extraction Source: {layer_name}")
    print(f"Final Feature Shape: {features.shape}")
    print(f"Feature Type: {features.dtype}")
    print("-" * 30)

    # 简单验证输出不是全0
    if torch.count_nonzero(features) > 0:
        print("Test Passed: Features contain non-zero values.")
    else:
        print("Warning: Features are all zeros.")

if __name__ == "__main__":
    main()
    
    
# python src/f5_tts/scripts/ssl_feature_test.py