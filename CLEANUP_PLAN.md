# WavTTS Cleanup Plan

本计划记录当前仓库从 F5-TTS 派生代码整理为 WavTTS 的维护状态。现在重点不是完整改名，而是先把训练主线、配置、入口脚本和验证基线收紧。

---

## 当前原则

- 先保持可运行，再继续删除和改名。
- README / 说明文档暂时少动，后续统一重写。
- `f5_tts` 包名和 `F5TTS` 模型名暂时保留，避免一次性破坏 import、checkpoint 和配置加载。
- `LICENSE`、citation、acknowledgement 后续单独整理，不能直接删除原 F5-TTS 合规信息。
- 每轮清理后至少运行：

```bash
conda run -n wavtts python -m pytest tests
conda run -n wavtts python -m compileall -q tests src/f5_tts
```

如果不用 conda，也可以在固定 `.venv` 中执行同样命令。

---

## 当前保留范围

### 训练

保留训练主线和两个公开 launcher：

- `src/f5_tts/train/train.py`
- `src/f5_tts/train/run_main_train.sh`
- `src/f5_tts/train/run_train_libritts.sh`

### 配置

`src/f5_tts/configs/` 只保留以下 7 个配置：

- `F5TTS_Base.yaml`
- `F5TTS_v1_Base.yaml`
- `F5TTS_v1_Large_mel_baseline.yaml`
- `F5TTS_v1_Large_wav_x_pred_scale_8_aux_mel_w_0_05_noise_schedule_0_8_16k_dropout_0_joint_drop_0_1.yaml`
- `F5TTS_v1_Large_wav_x_pred_scale_8_aux_mel_w_0_05_noise_schedule_0_8_16k_dropout_0_joint_drop_0_1_LibriTTS.yaml`
- `F5TTS_v1_Large_wav_x_pred_scale_9_aux_mel_w_0_05_noise_schedule_0_8_16k_dropout_0_joint_drop_0_1.yaml`
- `F5TTS_v1_Large_wav_x_pred_scale_10_aux_mel_w_0_05_noise_schedule_0_8_16k_dropout_0_joint_drop_0_1.yaml`

当前主训练配置是：

```text
src/f5_tts/configs/F5TTS_v1_Large_wav_x_pred_scale_9_aux_mel_w_0_05_noise_schedule_0_8_16k_dropout_0_joint_drop_0_1.yaml
```

### 推理与评测

- CLI inference 暂时保留。
- `src/f5_tts/eval/` 暂时保留。
- Gradio / Web demo 不再维护。
- TensorRT / Triton runtime 不再维护。

### 测试

已新增最小 smoke tests：

- `tests/test_smoke.py`

覆盖内容：

- `import f5_tts`
- 主配置加载
- toy waveform CFM/DiT 初始化

---

## 已完成清理

- [x] 建立 conda `wavtts` 验证环境。
- [x] 新增 `dev` optional dependency：`pytest`。
- [x] 给包初始化设置默认 `NUMBA_CACHE_DIR` / `MPLCONFIGDIR`，规避本地缓存目录不可写导致的 import / CLI help 问题。
- [x] 补充 smoke tests。
- [x] 清理 configs，只保留 7 个指定配置。
- [x] 删除旧实验配置子目录。
- [x] 新增并整理 `run_main_train.sh`。
- [x] 将 LibriTTS launcher 整理为 `run_train_libritts.sh`。
- [x] 删除旧 `run_libritts/` 和 `runs_emilia/` 实验脚本目录。
- [x] 删除 Gradio 实现文件。
- [x] 删除 TensorRT / Triton runtime 目录。
- [x] 删除 SSL feature extraction 相关实验脚本。
- [x] 删除 `speed_test.sh` 等非主线脚本。

---

## 暂不处理

- README、train README、infer README 等文档后续统一修。
- `f5_tts` 包名迁移暂缓。
- `F5TTS` / `E2TTS` 命名残留暂缓分类处理。
- checkpoint 兼容策略暂缓最终决定。
- `eval/` 暂时保留，不做大删。

---

## 下一步建议

1. **检查当前删除后的断链**
   - 扫描已删除文件名和目录名：

```bash
rg "infer_gradio|finetune_gradio|triton_trtllm|extract_ssl_features|ssl_feature_test|speed_test|run_libritts|runs_emilia" .
```

2. **清理非文档代码里的旧模型分支**
   - 重点看 `E2TTS_Base`、`F5TTS_Small`、`F5TTS_v1_Small`、`F5TTS_v1_Ultra`。
   - 先处理 CLI / API / finetune 中明显引用已删除配置的分支。

3. **决定是否保留以下入口**
   - `src/f5_tts/api.py`
   - `src/f5_tts/socket_server.py`
   - `src/f5_tts/socket_client.py`
   - `src/f5_tts/infer/speech_edit.py`
   - `src/f5_tts/train/finetune_cli.py`
   - `src/f5_tts/train/debug_train.sh`

4. **收紧 pyproject**
   - 后续新增 WavTTS CLI entry point。
   - 旧 `f5-tts_*` 命令是否保留为兼容别名，需要单独决定。

5. **最后统一文档**
   - README 类文件最后重写，避免清理过程中反复改同一批说明。

---

## 当前验证基线

每次提交前建议运行：

```bash
conda run -n wavtts python -m pytest tests
conda run -n wavtts python -m compileall -q tests src/f5_tts
bash -n src/f5_tts/train/run_main_train.sh
bash -n src/f5_tts/train/run_train_libritts.sh
```

如改动 CLI / import / pyproject，额外运行：

```bash
conda run -n wavtts f5-tts_infer-cli --help
pip install -e .
```

---

## 最近进度

| 日期 | 内容 | 验证 |
| --- | --- | --- |
| 2026-05-20 | 建立 smoke tests 和 conda 验证基线 | `pytest tests`、`compileall` 通过 |
| 2026-05-20 | 清理 configs，只保留 7 个配置 | `pytest tests` 通过 |
| 2026-05-20 | 整理训练 launcher，删除旧实验 launcher 目录 | `bash -n`、`pytest tests`、`compileall` 通过 |
| 2026-05-20 | 删除 Gradio、Triton runtime、SSL feature extraction、speed/path/HDFS 辅助脚本 | 待最终复跑并提交 |
