# 新增 Commit 审查记录

日期：2026-03-29  
分支：`mel_dev`

## 审查范围

本次主要检查了相对 `origin/HEAD` 新增的、与近期 loss 改动相关的提交：

- `74ba6d3e6a`：修复 aux mel loss 只在 masked span 上计算，并对齐 mask 单位
- `56743ec371`：新增 no-Hubert 的 fix_mel_loss 配置，并更新本地脚本
- `eefe60f959`：新增 aux perceptual loss 时间加权能力，以及 `flow_loss_weight`

## 变更总结

近期改动主要包括：

1. 修复 aux mel loss，使其只在 masked span 上生效。  
2. 对齐随机 mask 与 mel/spec hop 单位，减少跨尺度映射误差。  
3. 新增 `flow_loss_weight`，可调主 flow/v loss 权重。  
4. 新增 aux perceptual loss 的时间加权能力。  
5. 增加了多组实验配置，用于比较 mel_only / mel_dominant / 不同 power 等设置。

## 可能存在的问题

### 1. 时间加权的 aux loss 没有做归一化

当前实现是直接使用：

- `(per_sample_loss * time_weight).mean()`

风险：

- loss 的整体量级会随 time 分布变化，不只取决于样本误差本身；
- 改动 `P_mean`、`P_std`、`t_eps` 或 power 后，aux loss 的绝对尺度会明显变化；
- 不同实验之间不太容易直接比较。

### 2. 时间权重在 `t -> 1` 附近可能过大

当前权重大致形式为：

- `((1 - t).clamp_min(t_eps))^(-power)`

以 `t_eps = 0.02` 为例：

- power=1 时，最大权重大约是 50；
- power=2 时，最大权重大约是 2500。

风险：

- 少量接近 `t=1` 的样本可能主导 aux loss；
- 在 `power=2` 的配置下，训练可能更不稳定。

### 3. `flow_loss_weight = 0.0` 会完全关掉主 flow/v loss

当前本地新增配置里有 mel-only 版本使用了：

- `flow_loss_weight: 0.0`

风险：

- 如果这是故意做 ablation，可以接受；
- 但如果本意只是“弱化” flow loss，那么这个设置过于激进；
- 训练会完全依赖 aux perceptual loss，可能偏离主目标。

### 4. 低 flow 权重和时间加权 aux loss 叠加后，可能改变优化主目标

风险：

- `flow_loss_weight` 降低后，aux loss 本来就更容易占主导；
- 如果再叠加强时间加权，实际训练目标可能会明显偏向 aux loss；
- 即使表面上的 aux loss weight 不大，实际影响也可能很强。

## 简要判断

- `74ba6d3e6a` 这次修复整体上是合理且有帮助的；
- `56743ec371` 主要是配置补充，本身没有看到明显代码问题；
- `eefe60f959` 提供了更灵活的 loss 控制能力，但也引入了上面几个需要重点观察的风险。

## 备注

本文件仅用于记录当前审查结论，暂不修改代码。

## 未提交改动审查记录

### 本次未提交改动概览

当前未提交改动主要包括：

- `src/f5_tts/model/cfm.py`
- `src/f5_tts/model/modules.py`
- `src/f5_tts/train/debug_train.sh`
- 多个新的实验配置文件
- 删除了旧的 `mel_only_time_weighted.yaml`

主要新增内容：

1. 在 `loss_space` 中新增了 `x` 和 `spec_scaled` 两种主损失形式；
2. 新增 `SpecScalingLoss`；
3. 将 mask 对齐逻辑扩展为同时兼容 aux mel 和 `spec_scaled`；
4. 新增一组实验配置，用于比较 `x loss`、`spec_scaled loss`、是否开启 aux mel、以及不同时间加权 power。

### 未提交改动中需要关注的问题

#### 1. `SpecScalingLoss` 当前定义更像“缩放后谱值均值”，不太像标准 L1/L2 重建损失

相关位置：

- `src/f5_tts/model/modules.py`
- `src/f5_tts/model/cfm.py`

当前实现大致是：

- 先计算 `err = x_pred - x_true`
- 再计算 `err` 的 spectrogram
- 用 GT spectrogram 做缩放
- 最后直接对缩放结果取 mean

需要关注：

- 当前没有看到显式的 `abs()` 或 `square()`；
- 因此它和常见的 L1/L2 reconstruction loss 定义不完全一致；
- 数值尺度和梯度特性可能与原来的 `flow / v / x` loss 差别较大。

风险：

- 如果目标是实现某种“谱域缩放误差”，当前公式需要后续再确认是否与预期一致；
- `flow_loss_weight`、`aux_mel_loss_weight` 等已有经验值不一定能直接复用。

#### 2. `spec_scaled` 主损失当前没有使用 time-weight，而 aux perceptual loss 使用了

相关位置：

- `src/f5_tts/model/cfm.py`
- `src/f5_tts/model/modules.py`

当前实现中：

- `spec_scaled` 被当作主损失；
- `aux_time_weight` 仍然只作用于 aux mel / hubert / eres2net；
- `spec_scaled` 本身没有接入 time-weight。

需要关注：

- 这在代码上没有问题；
- 但实验语义上需要明确：到底是“只有 aux loss 做时间加权”，还是“所有 perceptual / reconstruction loss 都应该做时间加权”。

风险：

- 后续比较 `spec_scaled` 与 time-weighted aux loss 实验时，容易混淆不同损失是否被同样处理。

### 未提交改动的简要判断

- `x loss` 的接入方式整体合理；
- `prediction == "flow"` 时补出 `x_pred` 的实现是合理的；
- `mask_align_to` 扩展后，整体逻辑比之前更通用；
- 当前最值得重点确认的是 `SpecScalingLoss` 的公式定义是否符合预期。
