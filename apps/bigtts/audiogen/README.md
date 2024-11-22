# soundify (video2audio)

ENV: https://ml.bytedance.net/development/instance/jobs/1dfb712722399ed1?trialId=34685182

Proxy:

```
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=*.byted.org
```

CODE:

```
mkdir .deploy_cache
hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/zhuangxiaobin/v2a_deploy/models--apple--DFN2B-CLIP-ViT-B-16 .deploy_cache
<<<<<<< HEAD


v3:
```
hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/zhuangxiaobin/v2a_deploy/v2a_0.7b_0.3_v2_20k_encoder.ckpt .deploy_cache
hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/zhuangxiaobin/v2a_deploy/v2a_0.7b_0.3_v2_20k_diffusion.ckpt .deploy_cache
hdfs dfs -get hdfs://haruna/home/byte_data_seed/lf_lq/speech/checkpoints/user/zhuangxiaobin/v2a_deploy/v2a_0.7b_0.3_v2_20k_vocoder.ckpt .deploy_cache
```
=======
>>>>>>> 73636a3c66d820f70b083bed41227b825d35b7e1
python3 apps/bigtts/audiogen/soundify/v2a_infer.py 
```

```
frames = v2a_model.read_video(in_video, target_fps=8, remove_caption=True) # remove_caption设置为 true，表示去除画面底部的 20%，对应字幕，最终还是用ocr来识别更合理
frames = frames.unsqueeze(0).to(device)
wave = v2a_model.inference(frames) # 默认 cfg = 4.5， steps = 25
save_audio(wave, out_audio)
save_video(in_video, out_audio, out_video)
```