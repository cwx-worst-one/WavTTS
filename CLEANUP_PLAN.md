# WavTTS 仓库清理与品牌迁移计划

本计划用于把当前基于 F5-TTS 的派生仓库，逐步整理为独立的 WavTTS 项目。目标不是“一次性重命名所有内容”，而是在每一步都可验证、可回滚、可审阅的前提下，完成品牌迁移、目录清理和兼容策略落地。

建议做法：
- 以“先审计、后迁移、再删除、最后重写文档”为主线推进。
- 每个阶段尽量单独成 commit，避免把命名迁移、功能删除、配置整理混在一起。
- 每完成一个阶段，都做一次最小验证并记录结果。

---

## 总体原则

- 先审计，再修改；先低风险改动，再高风险改动。
- 不要一开始做全局替换，尤其不要直接把所有 `F5` / `f5_tts` 批量替换掉。
- 文档名、CLI 名、Python 包名、模型类名要分阶段迁移，不要一次联动修改。
- 每完成一类改动后运行最小验证，确保 import、配置加载、推理入口或训练入口没有损坏。
- 如果需要兼容旧 checkpoint，应保留必要的旧类名、旧包名或旧配置别名。
- `LICENSE`、citation、acknowledgement 不能为了“去 F5 化”而直接删除，必须保留对原 F5-TTS 项目的合规说明。

---

## 建议的测试与 `.venv` 策略

这类仓库清理工作，**不要在阶段 0 就立刻安装整套依赖**。更合理的方式是：

- **阶段 1（结构审计）之前**：先不建 `.venv`，只做只读审计。
- **阶段 2（建立验证基线）开始前**：创建项目本地 `.venv`，用于记录“改造前基线”。
- **阶段 3 之后**：只要涉及 CLI、import、配置或入口脚本改动，就在 `.venv` 中重复做最小验证。
- **阶段 15（最终验证）**：使用同一个 `.venv` 做完整回归；如果依赖有明显变更，再额外新建一次干净环境复验安装流程。

推荐原因：
- 审计阶段不需要为大型依赖付出时间成本。
- 进入“验证基线”阶段后，需要一个固定环境来判断问题是“仓库原本就有”还是“改动引入的”。
- 把 `.venv` 建在仓库内，便于团队统一约定，也便于把 `.venv/` 明确加入 `.gitignore`。

建议在 **阶段 2 开始时** 执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

如果本项目依赖 PyTorch / CUDA 版本需要手动匹配，则建议改成：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# 先按机器环境安装匹配版本的 torch / torchaudio
# 再安装项目本体
pip install -e .
```

如果安装 `pip install -e .` 代价太高，也可以先做一个轻量基线：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m compileall .
```

但从当前仓库形态看，`pyproject.toml` 已存在，后续又一定会检查脚本入口和 import，因此**阶段 2 建 `.venv` 并尝试 editable install 是更合适的默认方案**。

---

## 阶段 0：确认目标边界

目标：先把“要保留什么、要兼容什么、要删除什么”定清楚，避免后续反复返工。

- [ ] 确认 WavTTS 是否需要兼容原 F5-TTS checkpoint。
- [ ] 确认是否保留旧的 `F5TTS` 类名别名。
- [ ] 确认是否保留旧的 `f5_tts` Python import 兼容层。
- [ ] 确认是否保留旧的 CLI 命令别名。
- [x] 已确认当前不保留或不优先维护的部分：Gradio App、Docker、Development 文档/流程（后续如有需要再补）。
- [x] 已确认当前训练主线配置文件：`src/f5_tts/configs/F5TTS_v1_Large_wav_x_pred_scale_9_aux_mel_w_0_05_noise_schedule_0_8_16k_dropout_0_joint_drop_0_1.yaml`。
- [ ] 确认 WavTTS 必须保留的功能：
  - [ ] 推理
  - [ ] 训练
  - [ ] 微调
  - [x] Gradio / Web demo（当前不作为维护优先级）
  - [ ] 数据预处理
  - [ ] 多说话人支持
  - [ ] vocoder 切换
  - [x] 评测 / benchmark（当前不作为维护优先级）
- [ ] 确认明确要删除的内容：
  - [ ] ablation 实验代码
  - [ ] ablation 配置
  - [ ] 旧 demo / notebook
  - [ ] 不需要的模型变体
  - [ ] 不需要的数据处理脚本
  - [ ] 不需要的 benchmark / eval

---

## 阶段 1：仓库结构审计

目标：只审计，不修改代码，也不急着安装环境。

- [ ] 梳理顶层文件：
  - [ ] `README.md`
  - [ ] `LICENSE` / `NOTICE` / `CITATION.cff`
  - [ ] `pyproject.toml` / `setup.py` / `setup.cfg`
  - [ ] `requirements.txt` / `environment.yml`
  - [ ] `scripts/`
  - [ ] `configs/`
  - [ ] `examples/`
  - [ ] `tests/`
- [ ] 梳理 Python 包结构。
- [ ] 找出所有训练入口。
- [ ] 找出所有推理入口。
- [ ] 找出所有 demo 入口。
- [ ] 找出所有配置文件入口。
- [ ] 找出模型 registry / 模型构造逻辑。
- [ ] 找出 checkpoint 加载逻辑。
- [ ] 找出所有包含以下关键词的位置并分类：
  - [ ] `F5`
  - [ ] `f5`
  - [ ] `F5-TTS`
  - [ ] `f5_tts`
  - [ ] `E2`
  - [ ] `e2`
  - [ ] `ablation`
  - [ ] `baseline`

建议命令：

```bash
rg "F5|f5|F5-TTS|f5_tts|E2|e2|ablation|baseline"
rg --files
```

如果机器上没有 `rg`，再退回使用 `find` / `grep`。

---

## 阶段 2：建立验证基线

目标：在正式修改前确认当前仓库哪些流程真实可运行，并从这里开始建立 `.venv`。

- [x] 创建仓库本地 `.venv`。
- [x] 将 `.venv/` 加入 `.gitignore`（如果尚未忽略）。
- [x] 记录当前可用的安装方式。
- [x] 记录当前可用的 import 测试。
- [x] 记录当前可用的最小推理命令。
- [ ] 记录当前可用的训练或模型初始化命令。
- [ ] 如果没有测试，新增最小 smoke tests：
  - [ ] import 测试
  - [ ] 配置加载测试
  - [ ] 模型初始化测试
- [ ] 确认 `python -m compileall .` 当前状态。
- [ ] 确认 `pytest` 当前状态（如果项目已有测试）。
- [x] 确认 `pip install -e .` 当前状态。
- [ ] 记录哪些失败是“仓库原有问题”，哪些是“环境缺失导致”。

建议顺序：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
python -m compileall .
pytest
```

说明：
- 如果 `pytest` 当前没有测试，可以记录为“无测试可跑”，但最好尽快补最小 smoke tests。
- 如果需要 GPU 特定依赖，先把 CPU 可做的 import / compile / 配置加载基线记录下来。

---

## 阶段 3：文档和展示层命名迁移

目标：先修改对外展示，不动 Python import 路径和核心类名。

- [x] 初步修改 `README.md` 中的项目名为 WavTTS。
- [ ] 修改 docs 中的标题、介绍、命令说明。
- [ ] 修改 examples 中的说明文本。
- [ ] 修改 Gradio / Web demo 页面标题和描述。
- [ ] 修改 CLI help 文本中的项目名。
- [ ] 保留必要的 acknowledgement，明确说明项目基于 F5-TTS 修改。
- [ ] 暂时不修改 `f5_tts` 包名。
- [ ] 暂时不修改核心模型类名。
- [ ] 在 `.venv` 中复跑最小 import / help 验证。

---

## 阶段 4：CLI 入口命名迁移

目标：将外部命令迁移到 WavTTS 命名，但优先保证旧命令兼容策略清晰。

- [ ] 确认新的命令命名规范，例如：
  - [ ] `wavtts-infer`
  - [ ] `wavtts-train`
  - [ ] `wavtts-finetune`
  - [ ] `wavtts-demo`
- [ ] 修改 `pyproject.toml` 或 `setup.py` 中的 entry points。
- [ ] 修改 shell scripts 中的旧命令。
- [ ] 修改 README / examples 中的旧命令。
- [ ] 决定是否保留旧的 `f5-tts_*` 命令别名。
- [ ] 如果保留旧别名，标记为 deprecated。
- [ ] 在 `.venv` 中重新安装或刷新 editable install。
- [ ] 运行 CLI help 测试。

---

## 阶段 5：Python 包名迁移

目标：将内部包名从 F5-TTS 相关命名迁移到 WavTTS；这是第一次高风险改动。

- [ ] 决定新包名：建议 `wavtts`。
- [ ] 将旧包目录迁移到新包目录。
- [ ] 修改所有 `from f5_tts...` import。
- [ ] 修改所有 `import f5_tts` import。
- [ ] 修改配置文件中的 `_target_` 或模块路径。
- [ ] 修改脚本中的模块路径。
- [ ] 修改 tests 中的模块路径。
- [ ] 修改打包配置中的 package discovery。
- [ ] 如需兼容，新增 `f5_tts` compatibility shim。
- [ ] 验证 `import wavtts`。
- [ ] 验证旧 import 是否按预期可用或不可用。
- [ ] 在 `.venv` 中重新执行 `pip install -e .` 后再验证。

---

## 阶段 6：模型类名和内部命名迁移

目标：将核心类名、变量名、注册名逐步迁移到 WavTTS。

- [ ] 梳理所有模型类名。
- [ ] 梳理所有 registry key。
- [ ] 梳理所有配置里引用的模型名。
- [ ] 新增 `WavTTS` / `WavTTSModel` 等新名字。
- [ ] 如需兼容，保留 `F5TTS = WavTTS` 这类别名。
- [ ] 修改 CLI、配置和文档优先使用 WavTTS 名字。
- [ ] 确认 checkpoint 加载不受影响，或明确不再兼容旧 checkpoint。
- [ ] 运行模型初始化测试。

---

## 阶段 7：删除低风险遗留内容

目标：先删除明显不参与核心流程的内容。

- [ ] 删除过时文档。
- [ ] 删除无用图片、旧 demo 资源。
- [ ] 删除不用的 notebook。
- [ ] 删除旧实验说明。
- [ ] 删除未被引用的示例脚本。
- [ ] 删除前使用 `rg` 确认引用关系。
- [ ] 删除后运行 import / compile 检查。

---

## 阶段 8：删除 ablation 配置和脚本

目标：清理多余消融实验入口。

- [ ] 找出所有 ablation 相关配置。
- [ ] 找出所有 ablation 相关 shell scripts。
- [ ] 找出所有 ablation 相关 README / docs。
- [ ] 确认它们没有被主训练 / 主推理路径引用。
- [ ] 删除不用的 ablation 配置。
- [ ] 删除不用的 ablation 脚本。
- [ ] 删除或更新相关文档引用。
- [ ] 运行配置加载测试。

---

## 阶段 9：删除不用的模型变体和实验分支

目标：清理不属于 WavTTS 主线的模型代码。

- [ ] 梳理所有模型变体。
- [ ] 标记必须保留的模型。
- [ ] 标记可删除的模型。
- [ ] 检查可删除模型是否被配置引用。
- [ ] 检查可删除模型是否被 checkpoint loader / registry 引用。
- [ ] 删除不用的模型文件或类。
- [ ] 删除对应 registry entry。
- [ ] 删除对应配置。
- [ ] 删除对应测试或示例。
- [ ] 每删除一批运行一次 smoke test。

---

## 阶段 10：整理配置系统

目标：让配置结构服务于 WavTTS 主线。

- [ ] 统一模型配置命名。
- [ ] 统一训练配置命名。
- [ ] 统一推理配置命名。
- [ ] 统一 checkpoint 路径字段。
- [ ] 统一 sample rate / hop length / mel 参数。
- [ ] 统一 vocoder 配置。
- [ ] 删除旧项目残留配置字段。
- [ ] 删除不再使用的默认值。
- [ ] 确认 README 中的配置示例可用。

可选目标结构：

```text
configs/
  model/
    wavtts.yaml
  train/
    base.yaml
  infer/
    default.yaml
  dataset/
    default.yaml
```

---

## 阶段 11：整理训练、推理和 demo 主路径

目标：让用户入口清晰、稳定。

- [ ] 推理入口统一为 WavTTS 命名。
- [ ] 训练入口统一为 WavTTS 命名。
- [ ] 微调入口统一为 WavTTS 命名。
- [ ] demo 入口统一为 WavTTS 命名。
- [ ] 删除入口中不必要的 F5-TTS 特殊分支。
- [ ] 删除不需要的实验参数。
- [ ] 保留必要的错误提示和兼容提示。
- [ ] 运行最小推理测试。
- [ ] 如保留训练，运行最小训练或 dry-run 测试。

---

## 阶段 12：正式重写 README

目标：代码结构稳定后重写最终文档，而不是在中途反复大改。

- [ ] 重写项目介绍。
- [ ] 重写安装说明。
- [ ] 重写快速开始。
- [ ] 重写推理说明。
- [ ] 重写训练 / 微调说明。
- [ ] 重写 checkpoint 说明。
- [ ] 重写数据准备说明。
- [ ] 重写项目结构说明。
- [ ] 更新 citation。
- [ ] 更新 acknowledgement。
- [ ] 更新 license 说明。
- [ ] 确认所有 README 命令都能运行或明确标注条件。

建议 README 结构：

```text
# WavTTS

## Introduction
## Highlights
## Installation
## Quick Start
## Inference
## Training / Fine-tuning
## Checkpoints
## Data Preparation
## Project Structure
## Citation
## Acknowledgements
## License
```

---

## 阶段 13：License、Citation 和合规检查

目标：确保派生项目合规。

- [ ] 检查原项目 license 要求。
- [ ] 保留必要版权声明。
- [ ] 保留必要 F5-TTS acknowledgement。
- [ ] 保留或更新 citation。
- [ ] 检查第三方模型 / vocoder / 数据处理代码的 license。
- [ ] 检查 README 是否准确说明本项目与 F5-TTS 的关系。

---

## 阶段 14：最终残留清理

目标：确认旧命名和旧功能没有误留；允许保留那些出于兼容或合规目的必须存在的痕迹。

- [ ] 搜索并分类处理 `F5`。
- [ ] 搜索并分类处理 `f5`。
- [ ] 搜索并分类处理 `F5-TTS`。
- [ ] 搜索并分类处理 `f5_tts`。
- [ ] 搜索并分类处理 `E2`。
- [ ] 搜索并分类处理 `e2`。
- [ ] 搜索并分类处理 `ablation`。
- [ ] 搜索并分类处理 `baseline`。
- [ ] 确认保留下来的旧关键词都属于 acknowledgement、citation、兼容层或历史说明。

建议命令：

```bash
rg "F5|f5|F5-TTS|f5_tts|E2|e2|ablation|baseline"
```

---

## 阶段 15：最终验证

- [ ] 运行 `python -m compileall .`。
- [ ] 运行 `pytest`（如果项目有测试）。
- [ ] 运行安装测试，例如 `pip install -e .`。
- [ ] 运行 CLI help 测试。
- [ ] 运行最小推理测试。
- [ ] 如保留训练，运行训练 dry-run 或最小训练测试。
- [ ] 检查 README 中所有命令。
- [ ] 检查打包元信息。
- [ ] 检查最终 git diff。
- [ ] 如依赖或安装流程已明显变更，在干净环境复验一次安装流程。

---

## 推荐 commit 拆分

- [ ] Commit 1：添加本计划和仓库审计记录。
- [ ] Commit 2：README / docs 初步 WavTTS 命名。
- [ ] Commit 3：建立 `.venv`、补充 smoke tests、记录验证基线。
- [ ] Commit 4：CLI 命令迁移。
- [ ] Commit 5：Python 包名迁移。
- [ ] Commit 6：模型类名和内部命名迁移。
- [ ] Commit 7：删除低风险旧文档、旧示例、旧资源。
- [ ] Commit 8：删除 ablation 配置和脚本。
- [ ] Commit 9：删除不用的模型变体和实验分支。
- [ ] Commit 10：整理训练、推理和 demo 主路径。
- [ ] Commit 11：重写最终 README。
- [ ] Commit 12：license / citation / acknowledgement 合规整理。
- [ ] Commit 13：最终残留清理和验证。

---

## 接下来 3 个直接执行动作

1. **收紧训练主线路径**
   - 检查并整理 `src/f5_tts/train/train.py`、主线 shell 脚本、README 中训练命令是否一致。
   - 去掉主线脚本里的个人环境变量、私有路径、硬编码代理和敏感信息。
   - 把当前主线 config 对应的训练脚本整理成可公开复用的版本。

2. **补最小 smoke tests**
   - 增加至少 3 个最小测试：`import f5_tts`、配置文件可加载、训练入口可完成参数解析/模型初始化的最小检查。
   - 目标不是跑完整训练，而是让后续命名迁移有自动回归点。

3. **开始第一轮低风险清理**
   - 优先处理 README、训练脚本、pyproject 中已经确认不再主推的 Gradio / Docker / Development 残留。
   - eval 和 finetune CLI 先降级优先级，不作为当前主线。

## 当前进度记录

| 日期 | 阶段 | 变更摘要 | 验证结果 | 备注 |
| --- | --- | --- | --- | --- |
| 2026-05-20 | 阶段 0-3 / 阶段 2 基线 | 收紧 README、移除 Gradio 入口、重建 `.venv`、预装 torch 2.9.1 / torchaudio 2.9.1、完成 `pip install -e .`、补 `src/f5_tts/__init__.py` | `import f5_tts` 通过，editable install 通过，CLI help 可启动（有 matplotlib 缓存目录警告） | 下一步聚焦训练主线路径与最小 smoke tests |
