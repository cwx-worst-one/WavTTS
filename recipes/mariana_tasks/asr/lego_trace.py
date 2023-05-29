from absl import logging, flags, app
import yaml
import time
import torch
import lego
import tempfile
import os
import json
from cruise.utilities.hdfs_io import hcopy, hopen, hexists
from cruise import CruiseConfig

import torch
from torch import nn
from transformers import AutoTokenizer
from mariana.data.gpt.tokenization import CasterTokenizer
from cruise import CruiseModule, CruiseCLI, CruiseConfig, last_cli
from cruise.utilities.distributed import DIST_ENV
from cruise.utilities.hdfs_io import hcopy, hisdir, hopen, hexists


from rich.console import Console
from rich.markdown import Markdown
from rich import print as rprint

from seed_finetune.utils.text import recover_text, preprocess, postprocess
from seed_finetune.models.generator import Generator
from seed_finetune.utils.common import parse_model_dir, disable_xperf, setup_generation_config

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_dir", None, "The trained output directory which contains checkpoints."
)

flags.DEFINE_string(
    "step", None, "The checkpoint step to be traced."
)

flags.DEFINE_string(
    "save_path", None, "The path traced mdoel."
)

flags.DEFINE_integer(
    "max_length", 2048, "The maximum sequence length." 
)

flags.DEFINE_integer(
    "max_new_tokens", 896, "The maximum new tokens."
)

flags.DEFINE_integer(
    "max_batch_size", 16, "The maximum batch size."
)

# ref: https://bytedance.feishu.cn/docx/SadJd6k2moiNZXxIB7ic8ox7nSh
import xperf_gpt
xperf_gpt.load_xperf_gpt()

class LegoTracer:
    def __init__(self):
        model_config, checkpoint, tokenizer= parse_model_dir(
            FLAGS.model_dir, FLAGS.step)

        self.tokenizer = tokenizer
        model_config = disable_xperf(model_config)
        generation_config = setup_generation_config(self.tokenizer)

        self.model = Generator(model_config=model_config, generation_config=generation_config)
        self.model.eval()
        self.model.to(torch.bfloat16)
        self.model.load(checkpoint)

        self.generate_kwargs = dict(
            max_length=FLAGS.max_length, max_new_tokens=FLAGS.max_new_tokens,
            do_sample=True, top_p=0.7, max_batch_size=FLAGS.max_batch_size, use_xperf_gpt=True)

    def preprocess(self, history):
        input_ids, prompt_len = preprocess(
            history=history, tokenizer=self.tokenizer, max_length=FLAGS.max_length,
            max_length_per_turn=FLAGS.max_length)
        return input_ids, prompt_len
    
    def postprocess(self, token_ids, prompt_len):
        length = len(token_ids)
        while length > 0 and token_ids[length - 1] == self.tokenizer.eos_token_id:
            length -= 1

        return postprocess(token_ids, prompt_len, self.tokenizer)

    def trace(self, save_path):
        input_prompt = ["小炒肉怎么做"]
        input_ids, prompt_len = self.preprocess(input_prompt)

        # 进行Lego 优化
        print("start tracing...")
        lego_opt_model = lego.seq2seq.optimize(
            self.model.gpt,
            **self.generate_kwargs,
            do_optimize=True)
        lego_opt_model.save(save_path)
        print(f"trace succese to {save_path}!")

        time1 = time.perf_counter()
        out = lego_opt_model(input_ids)
        response = self.postprocess(out[0], prompt_len)

        print(f"dela time: {time.perf_counter()-time1}")
        print(f"out: {out}\nresponse: {response}")

    def test(self, save_path):
        print("start testing...")
        model = torch.jit.load(save_path)

        input_prompt = ["怎么优雅地吃屎？"]
        time1 = time.perf_counter()
        input_ids, prompt_len = self.preprocess(input_prompt)
        out = model(input_ids)
        print(out)
        response = self.postprocess(out[0], prompt_len)
        print(f"dela time: {time.perf_counter()-time1}")
        print(f"test case: \npromot: {input_prompt}\nresponse: {response}")


        input_prompt = ["""作者讲了什么：
我第二次到仙岩的时候，我惊诧于梅雨潭的绿了。
梅雨潭是一个瀑布潭。仙岩有三个瀑布，梅雨瀑最低。走到山边，便听见哗哗哗哗的声音;抬起头，镶在两条湿湿的黑边儿里的，一带白而发亮的水便呈现于眼前了。我们先到梅雨亭。梅雨亭正对着那条瀑布;坐在亭边，不必仰头，便可见它的全体了。亭下深深的便是梅雨潭。这个亭踞在突出的一角的岩石上，上下都空空儿的;仿佛一只苍鹰展着翼翅浮在天宇中一般。三面都是山，像半个环儿拥着;人如在井底了。这是一个秋季的薄-阴-的天气。微微的云在我们顶上流着;岩面与草丛都从润湿中透出几分油油的绿意。而瀑布也似乎分外的响了。那瀑布从上面冲下，仿佛已被扯成大小的几绺;不复是一幅整齐而平滑的布。岩上有许多棱角;瀑流经过时，作急剧的撞击，便飞花碎玉般乱溅着了。那溅着的水花。晶莹而多芒;远望去，像一朵朵小小的白梅。微雨似的纷纷落着。据说，这就是梅雨潭之所以得名了。但我觉得像杨花，格外确切些。轻风起来时，点点随风飘散，那更是杨花了。——这时偶然有几点送入我们温暖的怀里，便倏的钻了进去，再也寻它不着。
梅雨潭闪闪的绿色招引着我们;我们开始追捉她那离合的神光了。揪着草，攀着乱石，小心探身下去，又鞠躬过了一个石穹门，便到了汪汪一碧的潭边了。瀑布在襟袖之间;但我的心中已没有瀑布了。我的心随潭水的绿而摇荡。那醉人的绿呀!仿佛一张极大极大的荷叶铺着，满是奇异的绿呀。我想张开两臂抱住她;但这是怎样一个妄想呀。——站在水边，望到那面，居然觉着有些远呢!这平铺着，厚积着的绿，着实可爱。她松松的皱缬着，像少妇拖着的 裙 幅;她轻轻的摆弄着，像跳动的初恋的处女的心;她滑滑的明亮着，像涂了“明油”一般，有鸡蛋清那样软，那样嫩，令人想着所曾触过的最嫩的皮肤;她又不杂些儿尘滓，宛然一块温润的碧玉，只清清的一色——但你却看不透她!我曾见过北京什刹海拂地的绿柳，脱不了鹅黄的底子，似乎太淡了。我又曾见过杭州虎跑寺近旁高峻而深密的“绿壁”，丛叠着无穷的碧草与绿叶的，那又似乎太浓了。其余呢，西湖的波太明了，秦淮河的也太暗了。可爱的，我将什么来比拟你呢?我怎么比拟得出呢?大约潭是很深的，故能蕴蓄着这样奇异的绿;仿佛蔚蓝的天融了一块在里面似的，这才这般的鲜润呀。——那醉人的绿呀!我若能裁你以为带，我将赠给那轻盈的舞女;她必能临风飘举了。我若能挹你以为眼，我将赠给那善歌的盲妹;她必明眸善睐了。我舍不得你;我怎舍得你呢?我用手拍着你，抚摩着你，如同一个十二三岁的小姑娘。我又掬你入口，便是吻着她了。我送你一个名字，我从此叫你“女儿绿”，好么?
我第二次到仙岩的时候，我不禁惊诧于梅雨潭的绿了。"""]
        time1 = time.perf_counter()
        input_ids, prompt_len = self.preprocess(input_prompt)
        out = model(input_ids)
        print(out)
        response = self.postprocess(out[0], prompt_len)
        print(f"dela time: {time.perf_counter()-time1}")
        print(f"test case: \npromot: {input_prompt}\nresponse: {response}")
        print("test success!")

def main(_):

    tracer = LegoTracer()
    tracer.trace(FLAGS.save_path)
    tracer.test(FLAGS.save_path)

if __name__ == "__main__":
    app.run(main)
