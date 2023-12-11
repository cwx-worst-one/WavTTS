https://bytedance.larkoffice.com/docx/LFnidM5VUoV7v6xcy6HcAB5Cneb

1. BigTTS & BigMusic 代码整合目标

当前BigTTS核心任务包含如下7个任务：
高度关联的三组实验，作为一组进行处理
1. umm.stage1: 0.7B, https://ml.bytedance.net/development/instance/jobs/2fb7b456ae6553ec  @李唐 
2. umm.stage2: 0.7B, https://ml.bytedance.net/development/instance/jobs/77aed56c095853f6  @李唐 
3. umm.stage3: 0.7B, https://ml.bytedance.net/development/instance/jobs/b30e2dd2d4d07409  @李唐 

---
backbone 均为 ctiga，可以作为一组进行处理
4. umm.diffusion: 0.4B, https://ml.bytedance.net/development/instance/jobs/a17b558ea35ad148  @贾东亚 @从坚 
5. umm.ar: 0.7B, https://ml.bytedance.net/development/instance/jobs/bb4e48f592758f84  @陈家炜 @Kainan Peng 
6. wvae.ar: 0.9B, https://ml.bytedance.net/development/instance/jobs/8d5975ef8c9c7fe3  
https://ml.bytedance.net/development/instance/jobs/6ca159fab15fc4cd@黄智颖 @Mingbo Ma @潘俊杰 @李佳鑫 @李乃寒 

---
7. wvae.vocoder: 0.3B, https://ml.bytedance.net/development/instance/jobs/dd98640acfc16882  @刘正曦  @李乃寒 

---

当前BigMusic核心任务包含如下6个任务：
对应于tts的umm的三个stage，应该只有数据部分不一致
1. umm.stage1: https://ml.bytedance.net/development/instance/jobs/d684cbfbad8b2de8 @Zongyu Yin 
2. umm.stage2: https://ml.bytedance.net/development/instance/jobs/0952365a5c7d9141 @Zongyu Yin 
3. umm.stage3: https://ml.bytedance.net/development/instance/jobs/95dc633c7bfc555f @Zongyu Yin 

---
对应于umm.ar, 大体结构一致，算法还没有开始做合并相关的工作；
@Andrew Shaw  @Qingqing Huang 
计划添加这些任务，大体一致，主要是dependency，dataloader和模型input/output差异
4. P0 umm.ar.Lyrics2Song En base model
5. P0 umm.ar.Instrumental https://ml.byteintl.net/development/instance/jobs/7f27af1a49fab84c  metrics:
6. P0 mixed language training. (sami-tts-api phonemizer)
7. P1 umm.ar.voice related 
8. P1 leadsheet related 

---
对应于umm.diffusion, 跟tts的结构不一致；短期内不会跟tts任务合并
9. umm.diffusion: https://ml.bytedance.net/development/instance/jobs/aa0d48c6f7efd022 @Wei Tsung Lu 

---
10. umm.vocoder https://ml.bytedance.net/development/instance/jobs/029697361ee7b869 @Wei Tsung Lu 

---
Mulan
11. @Xuchen Song 

本次BigTTS & BigMusic 代码整合的目标是：
1. 实现当前BigTTS & BigMusic核心任务在主线master branch收敛；
2. 实现当前BigTTS & BigMusic核心任务在主线master branch进行稳定性维护和统一优化，实现一次优化，多处受益；
3. 探索工程和算法之间高效合作的方式；