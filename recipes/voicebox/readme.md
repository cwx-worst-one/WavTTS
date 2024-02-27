重构测试流程
1. 克隆测试集合；git clone git@code.byted.org:seed/bigtts_testset.git
2. 克隆测试脚本；git clone git@code.byted.org:seed/bigtts-eval.git
3. 修改 'recipes/voicebox/scripts/recons_umm_v2.sh' 中的输出地址、测试文件meta的路径、模型地址; 
4. 修改recipes/voicebox/scripts/eval.sh 中 测试脚本的路径;
5. 运行 bash recipes/voicebox/scripts/recons_umm_v2.sh，对应的 wer文件和asr文件 会存在 recons_umm_v2.sh 中指定的输出文件夹中；