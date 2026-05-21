# WavTTS Cleanup Plan

本计划记录当前仓库从 F5-TTS 派生代码整理为 WavTTS 的维护状态。当前重点仍然不是一次性完整改名，而是先把 **wav-only 训练主线、CLI 推理、配置、入口脚本和验证基线** 收紧。

---

## 当前原则

- 先保持可运行，再继续删除和改名。
- `f5_tts` 包名暂时保留，避免一次性破坏 import、checkpoint 和配置加载。
- `F5TTS` / `E2TTS` 等旧模型命名残留暂缓分类处理，但训练/推理主线优先使用 `WavTTS_*` 配置名。
- README / train README / infer README 等文档后续统一重写，避免清理过程中反复改同一批说明。
- `LICENSE`、citation、acknowledgement 后续单独整理，不能直接删除原 F5-TTS 合规信息。
- 默认开发环境使用项目 `.venv`；重型真实推理/训练脚本不放进默认 smoke baseline。
- Bash 脚本变量按项目习惯使用直接赋值，例如 `CONFIG_NAME="WavTTS_scale_9_16k"`，不使用 `${VAR:-default}` 形式。

---

## 当前保留范围

### 训练

保留训练主线和两个公开 launcher：

- `src/f5_tts/train/train.py`
- `src/f5_tts/train/run_main_train.sh`
- `src/f5_tts/train/run_train_libritts.sh`

当前主训练 launcher 默认配置：

```text
src/f5_tts/configs/WavTTS_scale_9_16k.yaml
```

LibriTTS launcher 默认配置：

```text
src/f5_tts/configs/WavTTS_scale_8_16k_libritts.yaml
```

### 配置

`src/f5_tts/configs/` 当前保留 wav-only 主线配置：

- `WavTTS_scale_8_16k.yaml`
- `WavTTS_scale_8_16k_libritts.yaml`
- `WavTTS_scale_9_16k.yaml`
- `WavTTS_scale_10_16k.yaml`

旧 F5-TTS mel/base 配置已从当前维护集合移除；如后续需要 mel baseline，可从历史提交恢复或单独建立 legacy 配置。

### 推理与评测

- CLI inference 保留并优先支持 wav-only / `no_vocoder` 路径。
- `src/f5_tts/infer/debug_infer.sh` 默认使用已有 WavTTS checkpoint 做真实推理验证。
- `src/f5_tts/eval/` 暂时保留，但不是当前清理重点。
- Gradio / Web demo 不再维护。
- TensorRT / Triton runtime 不再维护。

当前 debug 推理 checkpoint：

```text
/mnt/bn/jdy-lq-5/chenwenxi/exp/nar_wav_tts/emilia/F5TTS_v1_Large_wav_x_pred_scale_8_aux_mel_w_0_05_noise_schedule_0_8_16k_dropout_0_joint_drop_0_1-emilia-8gpus-19200sample_per_gpu-bf16/ckpts/model_1000000.pt
```

对应配置：

```text
src/f5_tts/configs/WavTTS_scale_8_16k.yaml
```

### 测试 / 验证脚本

轻量 smoke baseline：

- `scripts/check_baseline.sh`
- `tests/test_smoke.py`
- `tests/test_waveform_dataset_collate.py`

真实 checkpoint 加载脚本：

- `scripts/smoke_load_wavtts_checkpoint.py`

注意：默认 baseline 不跑模型 forward/sample，避免耗时或卡住；真实模型加载/推理由单独脚本手动执行。

---

## 已完成清理与更新

- [x] 建立 WavTTS cleanup 计划。
- [x] 新增 `dev` optional dependency：`pytest`。
- [x] 给包初始化设置默认 `NUMBA_CACHE_DIR` / `MPLCONFIGDIR`，规避本地缓存目录不可写导致的 import / CLI help 问题。
- [x] 补充轻量 smoke tests。
- [x] 清理 configs，并将 wav-only 主线配置重命名为 `WavTTS_scale_*_16k.yaml`。
- [x] 新增并整理 `run_main_train.sh`。
- [x] 将 LibriTTS launcher 整理为 `run_train_libritts.sh`。
- [x] 更新训练 launcher 默认配置名：`WavTTS_scale_9_16k` / `WavTTS_scale_8_16k_libritts`。
- [x] 更新 `src/f5_tts/infer/debug_infer.sh`，默认加载 `model_1000000.pt` 进行 WavTTS 推理验证。
- [x] 删除旧 `run_libritts/` 和 `runs_emilia/` 实验脚本目录。
- [x] 删除 Gradio 实现文件。
- [x] 删除 TensorRT / Triton runtime 目录。
- [x] 删除 SSL feature extraction 相关实验脚本。
- [x] 删除 `speed_test.sh` 等非主线脚本。
- [x] 从 `debug_train.sh` 移除真实 `WANDB_API_KEY`，避免密钥泄露。
- [x] 基于可用环境整理精简依赖计划：`requirements/wavtts-good-min.txt` / `requirements/wavtts-good-constraints.txt`。
- [x] 修正 `src/f5_tts/infer/utils_infer.py` wav-only 推理细节：
  - 默认 sample rate 改为 16k。
  - `no_vocoder` 直接返回 `None`。
  - 从模型/config 读取 `target_sample_rate` 和 `hop_length`。
  - CUDA autocast 只在 CUDA 可用时启用。
  - wav-only 参考音频按 `wav_frame_len` 对齐。
- [x] 在 `CFM` 中保存 `target_sample_rate`，方便 wav-only 推理获取音频几何信息。
- [x] 清理 WavTTS 配置音频几何命名：
  - `model.mel_spec` 改为 `model.waveform`。
  - 删除配置中的 `n_mel_channels`，统一用 `wav_frame_len` 表示 waveform frame 维度。
  - 训练、推理、eval batch、smoke 脚本改为读取 `model.waveform`。
- [x] 清理主线 DiT 维度命名：`mel_dim` 参数改为 `wav_frame_len`，调用侧同步更新。
- [x] 删除未使用的 legacy backbone：`UNetT`；`MMDiT` 与 modules 中对应实现先保留不动。

---

## 暂不处理

- `f5_tts` 包名迁移暂缓。
- checkpoint 兼容策略暂缓最终决定。
- `eval/` 暂时保留，不做大删。
- `api.py` / socket / finetune / speech_edit 入口已从主线删除。
- README、train README、infer README 等文档后续统一修。

---

## 当前验证基线

每次提交前建议运行轻量 baseline：

```bash
bash scripts/check_baseline.sh
```

等价核心检查：

```bash
PYTHONPATH=src .venv/bin/python tests/test_smoke.py
PYTHONPATH=src .venv/bin/python tests/test_waveform_dataset_collate.py
PYTHONPATH=src .venv/bin/python -m compileall -q tests src/f5_tts
bash -n src/f5_tts/train/run_main_train.sh
bash -n src/f5_tts/train/run_train_libritts.sh
bash -n src/f5_tts/infer/debug_infer.sh
```

如改动 CLI / import / 推理依赖，额外运行：

```bash
PYTHONPATH=src .venv/bin/python src/f5_tts/infer/infer_cli.py --help
PYTHONPATH=src .venv/bin/python -c "import f5_tts.infer.utils_infer; print('ok')"
```

真实 checkpoint 加载验证：

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_load_wavtts_checkpoint.py
```

真实推理验证：

```bash
bash src/f5_tts/infer/debug_infer.sh
```

---

## 环境建议

当前可用环境以 `/mnt/bn/jdy-lq-5/chenwenxi/code/wavtts_good_env_freeze.txt` 为参考。由于 `pyproject.toml` 中部分依赖范围较宽，直接 `pip install -e .` 可能导致依赖版本漂移。

推荐安装流程：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements/wavtts-good-min.txt -c requirements/wavtts-good-constraints.txt
.venv/bin/python -m pip install -e . --no-deps
```

重点：editable 安装最后使用 `--no-deps`，避免重新拉取不兼容的新版本依赖。

---

## 下一步建议

1. **验证并继续收紧 wav-only CLI 推理**
   - 运行 `debug_infer.sh` 确认 `model_1000000.pt` 可以稳定生成 wav。
   - 如有问题，优先定位 `infer_cli.py` 与 `utils_infer.py` 的 wav-only 路径。

2. **重写 `debug_train.sh`**
   - 改为公开可复用 debug 训练脚本。
   - 使用 `WavTTS_scale_8_16k` 或 `WavTTS_scale_9_16k`。
   - 移除私有绝对路径、HF mirror、debugpy、旧配置名等。

3. **清理非文档代码里的旧模型分支**
   - 重点看 `E2TTS_Base`、`F5TTS_Small`、`F5TTS_v1_Small`、`F5TTS_v1_Ultra`。
   - 先处理 CLI / API / finetune 中明显引用已删除配置的分支。

4. **决定是否保留以下入口**

5. **收紧 pyproject / CLI entry point**
   - 后续新增 WavTTS CLI entry point。
   - 旧 `f5-tts_*` 命令是否保留为兼容别名，需要单独决定。

6. **最后统一文档**
   - README 类文件最后重写。
   - 保留必要 F5-TTS acknowledgement / license / citation 信息。

---

## 最近进度

| 日期 | 内容 | 验证 |
| --- | --- | --- |
| 2026-05-20 | 建立 smoke tests 和验证基线 | `pytest tests`、`compileall` 通过 |
| 2026-05-20 | 清理 configs，只保留主线配置 | `pytest tests` 通过 |
| 2026-05-20 | 整理训练 launcher，删除旧实验 launcher 目录 | `bash -n`、`pytest tests`、`compileall` 通过 |
| 2026-05-20 | 删除 Gradio、Triton runtime、SSL feature extraction、speed/path/HDFS 辅助脚本 | 已提交到 WavTTS `final-release` |
| 2026-05-21 | 将 wav-only 配置重命名为 `WavTTS_scale_*_16k`，更新训练/推理 debug 入口 | `bash -n` 通过 |
| 2026-05-21 | 修正 `utils_infer.py` wav-only/no_vocoder/16k 推理逻辑，`CFM` 保存 `target_sample_rate` | `py_compile` 与轻量 import 通过 |
| 2026-05-21 | 阶段 1 主链稳定确认：baseline、debug inference、训练 launcher 均可用 | 用户本地验证通过 |
| 2026-05-21 | 阶段 2 开始：删除旧 `api.py`、socket server/client、`finetune_cli.py`，移除 finetune CLI entry point | `bash -n` / `py_compile` 通过 |

---

## 后续阶段规划：正式改名与消融代码删减

### 总体顺序

后续不要立即全局改名。推荐顺序是：

```text
阶段 1：稳定 WavTTS 主链
阶段 2：收敛 legacy 入口
阶段 3：删除/迁移消融实验代码
阶段 4：正式 f5_tts -> wavtts 改名
阶段 5：统一文档、license、citation、README
```

原因：如果先改包名，旧入口、旧实验分支和消融代码都会扩大改名范围，增加无效迁移成本。先删减主线外代码，再改名更稳。

---

### 阶段 1：稳定 WavTTS 主链

目标：确保当前 wav-only 主线可训练、可推理。

必须确认：

- `src/f5_tts/infer/debug_infer.sh` 可以使用 `model_1000000.pt` 稳定生成 wav。
- `src/f5_tts/train/run_main_train.sh` 指向 `WavTTS_scale_9_16k`，至少能正常进入 model / dataloader 初始化。
- `src/f5_tts/train/run_train_libritts.sh` 不再引用旧配置名。
- `scripts/check_baseline.sh` 稳定通过。
- `src/f5_tts/infer/utils_infer.py` 的 wav-only / `no_vocoder` / 16k sample rate 路径稳定。

当前状态：阶段 1 已完成，下一步进入阶段 2（收敛 legacy 入口）。

---

### 阶段 2：收敛 legacy 入口

正式改名前，先决定以下入口是否维护：

- `src/f5_tts/eval/`

已删除旧入口：

- `src/f5_tts/api.py`
- `src/f5_tts/socket_server.py`
- `src/f5_tts/socket_client.py`
- `src/f5_tts/train/finetune_cli.py`

建议策略：

- 当前 WavTTS 主线只维护：
  - `train/`
  - `infer_cli.py`
  - `model/`
  - `dataset.py`
  - `configs/WavTTS_*.yaml`
  - `scripts/`
- 不维护的入口移动到 `legacy/`，或在文件头明确标注 legacy。
- 暂时保留的入口不要阻塞主线改名，但必须避免引用已删除配置。

---

### 阶段 3：删除/迁移消融实验代码

在正式改包名前，先收敛模型和训练代码的实验开关。

#### WavTTS v0 主线建议保留

```text
wav_input_only=True
prediction=x_pred
loss_space=v
use_aux_mel_loss=True
frontend_type=reshape（已固定为默认 reshape，相关配置分支已删除）
target_sample_rate=16000
wav_frame_len=160
```

#### 候选删除或迁移到 legacy 的实验分支

以下内容后续需要逐项确认是否仍被当前配置使用：

- `use_aux_hubert_loss`
- `use_aux_eres2net_loss`
- `use_repa_ctc_loss`（已删除）
- `use_repa_ssl_feature_loss`（已删除）
- `loss_space="spec_scaled"`（已删除）
- SSL feature loading / `ssl_feature_dataset_root`（已删除）
- `load_ssl_features`（已删除）
- `speech_align_depth`（已删除）
- `text_align_depth`（已删除）
- `SpeechAlignMLP` / `TextAlignMLP` 相关训练分支（已删除）
- 多种 wav frontend：`conv` / `embed_v1` / `embed_v2`（已删除，固定 reshape）
- `aux_mel_loss_start_t`（已删除，aux mel loss 不再按时间阈值跳过）
- aux mel loss energy scaling（已删除）
- aux mel loss normalized/mag-log/align 开关（已删除，固定 masked log-mel loss）
- `CFM.sample_rate` 默认值已改为 16000
- `flow_loss_weight`（已删除，固定主 loss 权重为 1）
- `noise_scale`（已删除，固定标准正态噪声 scale=1）
- 与当前 wav-only 主线无关的 mel/vocoder 分支
- 旧 eval / finetune 中的 mel-only 假设

#### 删除策略

不要一次性大删。建议先做一个清单：

```bash
rg "use_aux_hubert|use_aux_eres2net|use_repa|ssl_feature|spec_scaled|speech_align|text_align|frontend_type|embed_v1|embed_v2" src/f5_tts
```

然后按以下顺序处理：

1. 确认当前 `WavTTS_scale_*_16k.yaml` 是否还引用该分支。
2. 如果主线配置不使用，先标注为 legacy 或拆到单独文件。
3. 运行轻量 baseline。
4. 再删除相关 import / class / config 字段。
5. 每次只删除一个类别，避免难以定位回归。

---

### 阶段 4：正式改名 `f5_tts` -> `wavtts`

正式改名建议满足以下条件后开始：

- `debug_infer.sh` 已使用 `model_1000000.pt` 跑通。
- `run_main_train.sh` 至少能启动到 dataloader/model init。
- legacy 入口去留已经确定。
- 消融实验代码至少完成第一轮删减或迁移。

#### 改名范围

1. 包目录：

```text
src/f5_tts/ -> src/wavtts/
```

2. 保留兼容 shim：

```text
src/f5_tts/__init__.py
```

临时兼容旧 import：

```python
from wavtts import *
```

3. CLI entry points：

新增 WavTTS 命令：

```toml
wavtts-infer = "wavtts.infer.infer_cli:main"
wavtts-train = "wavtts.train.train:main"
```

旧命令可短期保留为兼容别名：

```toml
f5-tts_infer-cli = "wavtts.infer.infer_cli:main"
```

4. 配置路径和脚本路径：

- `configs/WavTTS_*.yaml` 保留。
- 所有训练/推理脚本改为 `wavtts` import 路径。
- checkpoint 加载需要确认 key 不受包名影响。

#### 暂不急着改的名称

以下名称可以最后处理：

- `CFM`
- `DiT`
- checkpoint 内部 key
- 历史论文 citation 中的 F5-TTS 名称

---

### 阶段 5：文档统一

最后统一修改：

- `README.md`
- `src/f5_tts/train/README.md` 或迁移后的 train README
- `src/f5_tts/infer/README.md` 或迁移后的 infer README
- `CLEANUP_PLAN.md`
- installation / environment docs
- acknowledgement / license / citation

要求：

- 明确 WavTTS 是 waveform-first fork。
- 说明仍兼容部分 F5-TTS checkpoint/import 的过渡策略。
- 不删除原 F5-TTS 合规信息。

---

## 阶段 3 只读扫描结果：消融/实验分支

当前 `WavTTS_scale_*_16k.yaml` 配置没有启用下列实验分支；主线已固定为 wav-only。

| 分支/模块 | 当前配置使用 | 建议 |
| --- | --- | --- |
| `use_aux_hubert_loss` | 否 | 已删除 |
| `use_aux_eres2net_loss` | 否 | 已删除 |
| `use_repa_ctc_loss` | 否 | 已删除 |
| `use_repa_ssl_feature_loss` | 否 | 已删除 |
| `loss_space="spec_scaled"` | 否 | 已删除 |
| SSL feature loading / `ssl_feature_*` | 否 | 已删除 |
| `speech_align_depth` / `text_align_depth` | 否 | 已删除 |
| wav frontend `conv/embed_v1/embed_v2` | 否，当前默认 `reshape` | 已删除，固定默认 reshape |
| `vocos` / `bigvgan` mel 推理 | 主线不用 | 主线已删除，README/旧脚本后续清理 |
| `speech_edit.py` | 主线不用 | 已删除 |
| `eval/` | 主线不用，但后续要用 | 保留，不改 |

阶段 3 当前进度：已删除 `HubertFeatureLoss` / `use_aux_hubert_loss`、`ERes2NetFeatureLoss` / `use_aux_eres2net_loss`、REPA 对齐损失相关代码、`loss_space="spec_scaled"` / `SpecScalingLoss`、dataset/collate 中的 SSL feature loading、time-weighted aux perceptual loss、`aux_mel_loss_start_t`、aux mel loss energy scaling、aux mel normalized/mag-log/align 开关，以及 wav frontend `conv/embed_v1/embed_v2` 消融分支；当前固定使用默认 reshape，`CFM.sample_rate` 默认 16000，aux mel 固定为 masked log-mel loss，主 flow loss 权重固定为 1，noise scale 固定为 1。


#### wav frontend 清理结果

已固定为默认 reshape，并处理：

- `src/f5_tts/model/cfm.py`：删除 `frontend_type` / `frontend_cfg` 参数和传递，只保留 `wav_frame_len`。
- `src/f5_tts/model/backbones/dit.py`：删除 `conv` / `embed_v1` / `embed_v2` 分支，`set_wav_frontend_config` 只配置 reshape。
- 删除未再引用的 `src/f5_tts/model/backbones/wav_frontend.py` 与 `src/f5_tts/model/backbones/wav_patch_embed.py`。
- 当前配置本身没有显式 frontend 字段，无需改 yaml。


#### 下一步候选：删除旧 mel/vocoder F5-TTS 路径

目标：WavTTS 主线只保留 raw waveform / `no_vocoder`，不再支持 mel 输入和外部 vocoder。

建议处理范围：

1. `src/f5_tts/model/cfm.py`
   - 固定 `wav_input_only=True`，删除非 wav 分支。
   - 删除 `MelSpec` 构建、`self.mel_spec`、`vocoder` 参数和 vocoder decode。
   - 保留 `MelSpectrogramLoss`，因为它仍作为 waveform aux loss 使用。

2. `src/f5_tts/model/dataset.py`
   - 已删除 `HFDataset` 与 preprocessed/mel dataset 路径。
   - `CustomDataset` / `collate_fn` 固定只返回 wav，不再返回 mel / mel_lengths。

3. `src/f5_tts/train/train.py` 与 `src/f5_tts/model/trainer.py`
   - 已删除 `wav_input` 开关、`mel_spec_type`、vocoder 初始化和 mel logging 分支。
   - 训练输入固定使用 `batch["wav"]` / `wav_lengths`。

4. `src/f5_tts/infer/utils_infer.py` / `infer_cli.py` / `src/f5_tts/eval/eval_infer_batch.py`
   - 已删除 `load_vocoder`、`vocoder_name`、`vocos/bigvgan` 分支。
   - 主推理与 batch eval 固定返回 waveform，CLI 不再暴露 vocoder 参数。

5. configs / deps
   - 已从 `WavTTS_scale_*_16k.yaml` 删除 `mel_spec_type: no_vocoder`、`return_wav_only` 和 `vocoder:` 配置；音频几何字段已收敛到 `model.waveform`。
   - 已从 `pyproject.toml` 删除 `vocos` 依赖；BigVGAN third_party 路径可后续清理。

`speech_edit.py` 已删除；eval batch 已改为 wav-only。


#### Config 命名清理结果

已将 WavTTS 主线配置从 mel-centric 命名收敛为 waveform-centric：

- `model.mel_spec` -> `model.waveform`。
- 配置中不再保留 `n_mel_channels`，统一用 `wav_frame_len` 表示 waveform frame 维度。
- `train.py` / `infer_cli.py` / `utils_infer.py` / `eval_infer_batch.py` / smoke 脚本已同步读取 `model.waveform`。
- 主线 `DiT` 内部和调用侧的 `mel_dim` 参数已改为 `wav_frame_len`。
- legacy `UNetT` 已删除；`MMDiT` 先保留，当前 WavTTS 主线仍使用 `DiT`。


#### MelSpec 清理结果

已删除 `modules.py` 中旧 `MelSpec`、`get_vocos_mel_spectrogram`、`get_bigvgan_mel_spectrogram` 以及 librosa mel cache；`eval/utils_eval.py` 已固定 wav-only prompt 处理。`MelSpectrogramLoss` 保留用于 waveform aux loss。
