import configargparse
import os
import importlib
import re
import copy


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


BASE_KEY = '_base_'
EXPORT_RELATED_ARGS = "--export-onnx 1 --solution.onnx_with_mask False --solution.encoder_convert_stream False --solution.predictor_convert_stream True --solution.use_lm_jointer 0 --solution.panther_use_gpu 0 --solution.use_symbolic_export 1 --solution.skip_export_test_error 1 --solution.use_non_combined_adaptive_softmax 1"
DATA_RELATED_ARGS = "--data.max_batch_size 0 --data.max_batch_scale 0"
RESUME_RELATED_ARGS = "--train.resume resume_ckpt-0.pth --train.resume_optimizer 0 --train.resume_lr_scheduler 0 --train.resume_progress 0"


class QuantConfig:
    """
    详细使用方式见doc: https://bytedance.feishu.cn/docx/Hg64d5QdioCgz1xJvxicdAdbnVd
    """

    def __init__(
        self,
        train_content,
        infer_content,
        remote_save_root,
        use_lm_jointer,
        use_stream,
        enable_debug,
    ):
        self.use_lm_jointer = str2bool(use_lm_jointer)
        self.use_stream = str2bool(use_stream)
        self.remote_save_root = remote_save_root
        self.ckpt_path = self.get_checkpoint(infer_content)
        self.train_args, self.train_cfg = self.extract_args_config(train_content)
        self.enable_debug = str2bool(enable_debug)
        self.gen_components()

    def extract_args_config(self, content):
        args = re.search(r'"args":"(.*?)"', content)
        assert args, "no args"

        config_path = re.search(r'--config (\S+)', args.group(1))
        assert config_path, "no config file"

        if os.system("doas hdfs dfs -get {}".format(config_path.group(1))) != 0:
            assert False, "download config fail"

        return args.group(1), config_path.group(1)

    def parse_args(self, input_str):
        parts = input_str.split()
        assert len(parts) % 2 == 0
        ret = []
        for i in range(len(parts) // 2):
            [arg, val] = parts[i * 2 : i * 2 + 2]
            assert arg.startswith("--")
            ret.append([arg, val])
        return ret

    def parse_config(self, config_path):
        basename = os.path.basename(config_path)
        assert basename[-3:] == ".py"
        mod = importlib.import_module(basename[:-3])
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }
        if BASE_KEY in cfg_dict:
            assert False, "please operate manually"
        return cfg_dict

    def get_checkpoint(self, content):
        args, config_path = self.extract_args_config(content)
        argv = self.parse_args(args)
        cfg_dict_infer = copy.deepcopy(self.parse_config(config_path))
        assert "train" in cfg_dict_infer

        keys = [
            "--train.resume_hdfs_chkpt",
            "--train.remote_save_root",
            "--train.save_dir",
            "--train.save_name",
            "--train.resume",
            "--train.resume_pretrain_chkpt",
        ]

        for arg in argv:
            key = arg[0]
            val = arg[1]
            if key in keys:
                key1, key2 = key[2:].split(".")
                cfg_dict_infer[key1][key2] = val

        ckpt_path = ""
        train_cfg = cfg_dict_infer["train"]
        if "resume_hdfs_chkpt" in train_cfg:
            ckpt_path = train_cfg["resume_hdfs_chkpt"]
        elif (
            "remote_save_root" in train_cfg
            and "save_dir" in train_cfg
            and "save_name" in train_cfg
            and "resume" in train_cfg
        ):
            ckpt_path = os.path.join(
                train_cfg["remote_save_root"],
                train_cfg["save_dir"],
                train_cfg["save_name"],
                "checkpoints",
                train_cfg["resume"],
            )
        elif "resume_pretrain_chkpt" in train_cfg:
            ckpt_path = train_cfg["resume_pretrain_chkpt"]
        else:
            assert False, "no checkpoint"

        if os.system("rm {}".format(os.path.basename(config_path))) != 0:
            assert False, "rm tmp infer config fail"

        return ckpt_path

    def gen_components(self):
        # for args
        self.filtered_args = [
            "--train.resume_hdfs_chkpt",
            "--train.resume_pretrain_chkpt",
            "--solution.exclude_load_keys",
        ]

        # for config
        self.cfg1 = {"slim_init_config": dict(resume_pretrain_ckpt=self.ckpt_path)}
        self.cfg2 = {
            "slim_config": dict(
                Quantizer={
                    'num_iters_to_calc_scale': 256,  # Typically 128, 256, 512
                    'offline': True,
                    'expand_lstm': True,  # If you need to quantize LSTM, set this to True
                    # Op-wise config
                    'Linear': {'per_channel': True},
                    'work_scope': ['acoustic_backbone_module', 'predictor_module'],
                }
            )
        }

    def process(self):
        self.gen_args()
        self.gen_config()

    def config(self, input_str):
        argv = self.parse_args(input_str)
        ret = []
        for item in argv:
            arg = item[0]
            val = item[1]
            assert arg.startswith("--")
            if arg == "--config":
                val = os.path.join(
                    self.remote_save_root, os.path.basename(self.train_cfg)
                )
            if self.use_lm_jointer and arg == "--solution.use_lm_jointer":
                val = "1"
            if self.use_stream and arg == "--solution.encoder_convert_stream":
                val = "True"
            ret.extend([arg, val])
        return " ".join(ret)

    def filter_args(self, input_args):
        ret = []
        for arg in input_args:
            assert arg[0].startswith("--")
            if arg[0] not in self.filtered_args:
                ret.extend(arg)
        return " ".join(ret)

    def gen_args(self):
        args = (
            self.filter_args(self.parse_args(self.train_args))
            + " "
            + EXPORT_RELATED_ARGS
            + " "
            + DATA_RELATED_ARGS
            + " "
            + RESUME_RELATED_ARGS
        )
        print(self.config(args))

    def gen_config(self):
        train_cfg = os.path.basename(self.train_cfg)
        cfg_dict = self.parse_config(train_cfg)

        if self.enable_debug:
            with open("tmp_config.py", "w") as f:
                for key in cfg_dict:
                    f.write(key + "=" + repr(cfg_dict[key]) + "\n")

        with open(train_cfg, "w") as f:
            for key in cfg_dict:
                if key == "solution":
                    assert isinstance(cfg_dict[key], dict)
                    cfg_dict[key].update(self.cfg1)
                    cfg_dict[key].update(self.cfg2)
                if key == "train":
                    assert "remote_save_root" in cfg_dict[key].keys()
                    cfg_dict[key]["remote_save_root"] = self.remote_save_root
                f.write(key + "=" + repr(cfg_dict[key]) + "\n")
        if (
            os.system(
                "doas hdfs dfs -put {} {}".format(train_cfg, self.remote_save_root)
            )
            != 0
        ):
            assert False, "upload config fail"


if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add('--train_content', type=str, required=True)
    parser.add('--infer_content', type=str, required=True)
    parser.add('--remote_save_root', type=str, required=True)
    parser.add('--use_lm_jointer', type=str, required=True, default=False)
    parser.add('--use_stream', type=str, required=True, default=False)
    parser.add('--enable_debug', type=str, required=False, default=False)

    option = parser.parse_args()

    q = QuantConfig(
        option.train_content,
        option.infer_content,
        option.remote_save_root,
        option.use_lm_jointer,
        option.use_stream,
        option.enable_debug,
    )
    q.process()
