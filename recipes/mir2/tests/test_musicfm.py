import os
import torch
from hyperpyyaml import load_hyperpyyaml


def load_default_musicfm_model():
    yaml_str = """
seed: 0
__set_seed: !apply:pytorch_lightning.seed_everything [!ref <seed>]
__do_nothing: !apply:recipes.mir2.utils.musicfm_adapt.a []

# running parameters
run_opts:
  fast_dev_run: False
  tensorboard_dir: '/mnt/bn/mir-tasks/ju-chiang.wang/'
  log_dir: logs
  log_name: 'fm7'
  version: 'multi_7954'
  precision: 16-mixed
  accelerator: 'auto'
  num_workers: 4
  pin_memory: True
  val_dataset_split_size: 3000
  tasks: ['beat', 'chord', 'structure', 'key']
  tasks_weight: [0.7, 0.9, 0.5, 0.4]
  bar_sync_chord: True

################################################################################
##########                    TRAINING CONFIGURATION                  ##########
################################################################################
train_params:
  learning_rate: 0.001
  max_epochs: -1
  batch_size: 5
  scheduler_patience: 20
  scheduler_decay_factor: 0.9
  devices: 'auto'
  shuffle_buffer: 100
  num_iter: 25 # total number is num_workers * num_iter

val_params:
  num_sanity_val_steps: 2
  batch_size: 1

################################################################################
##########                    TASK CONFIGURATION                  ##########
################################################################################

audio:
  sampling_rate: 24000            # sampling rate for audio
  sample_len: 24.0                # data-loader duration in sec for input  #{24 or 6}
  task_sample_len: [6.0, 12.0, 24.0, 12.0]    # model duration in sec for task
  hop_factor: 6

labels:
  beat_label_hop: 0.02
  chord_label_hop: 0.1
  structure_label_hop: 0.2
  key_label_hop: 2
  task_label_hops: [0.02, 0.1, 0.2, 2.0]
  task_n_timesteps: [300, 120, 120, 6]  # 6 / 0.02, 12 / 0.1
  foundation_hz: 25

local_cache_dir: "/mnt/bn/audio-diffusion/pretrained_models/musicfm/"
stat_path: !new:recipes.mir2.utils.helper.HdfsFileWrapper
  file_path: "hdfs://haruna/home/byte_speech_sv/haonanchen/logs/musicfm/playlist_classic_stats.json"
  local_cache_dir: !ref <local_cache_dir>
model_path: !new:recipes.mir2.utils.helper.HdfsFileWrapper
  file_path: "hdfs://haruna/home/byte_speech_sv/haonanchen/logs/musicfm/FM7.pt"
  local_cache_dir: !ref <local_cache_dir>

frontend: !new:recipes.mir_benchmark.models.frontend.Frontend
  model:
    !new:recipes.icassp.models.musicfm_25Hz.MusicFM25Hz
    encoder_depth: 12
    is_flash: False
    stat_path: !ref <stat_path>
    model_path: !ref <model_path>
  layer_ix: 12
  is_update: True

backend: !new:recipes.mir_benchmark.models.multitask_probing.MultitaskProbingStage
  n_channel: 1024
  foundation_hz: !ref <labels[foundation_hz]>
  tasks: !ref <run_opts[tasks]>  # include all task names
  label_hops: !ref <labels[task_label_hops]> # include all label_hops 
  resamples: !ref <labels[task_n_timesteps]> # include all n_timesteps 

model: !new:sami_ai.core.BaseModel
  input_names: [audio]
  output_names: [output]
  stages:
      - !ref <frontend>
      - !ref <backend>

pl_module: !new:recipes.mir_benchmark.pl_modules.multitask_finetune_pl.LitFinetuneMultitask
  model: !ref <model>
  tasks: !ref <run_opts[tasks]> 
  tasks_weight: !ref <run_opts[tasks_weight]>
  sample_rate: !ref <audio[sampling_rate]>
  sample_len: !ref <audio[task_sample_len]>
  label_hop: !ref <labels[task_label_hops]>
  joint_hop_factor: !ref <audio[hop_factor]>
  lr: !ref <train_params[learning_rate]>
  scheduler_patience: !ref <train_params[scheduler_patience]>
  scheduler_decay_factor: !ref <train_params[scheduler_decay_factor]>
  val_dataset_split_size: !ref <run_opts[val_dataset_split_size]>
"""
    hparams = load_hyperpyyaml(yaml_str)
    return hparams["pl_module"]


def test_musicfm_adapter():
    import recipes.mir2.utils.musicfm_adapt

    from recipes.musiclm.datamodules.webdataset import ConcatDatasetsWithinBatch
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip


def test_beat_data():
    import recipes.mir2.utils.musicfm_adapt
    from recipes.mir_benchmark.utils.concat_beat_data import concatenate_dataset

    urls = concatenate_dataset("train")
    assert len(urls) > 0


def test_key_preprocessor():
    curr_dir = os.path.dirname(__file__)
    ## Try to find samantha root dir based on your current dir
    samantha_root_dir = os.path.join(curr_dir, "../../../")
    os.chdir(samantha_root_dir)
    from recipes.mir_benchmark.utils.key_processor import KeyPreprocessor

    key_preprocessor = KeyPreprocessor(24, 24000, 2)
    assert key_preprocessor is not None


def test_multi_finetune_yaml():
    ## musicfm can only be run from the parent folder of "recipes"
    import recipes.mir2.utils.musicfm_adapt

    assert torch.cuda.is_available(), "Model can only run on gpu."
    # os.environ["DISABLE_FLASH_ATTN"] = "True"
    overrides = {
        "train_params": {
            "batch_size": 2,
        },
        "frontend": {"model": {
            "stat_path": "/mnt/bn/music-llm-nas-lq/pretrained_musicfm/playlist_classic_stats.json",
            "model_path": "/mnt/bn/music-llm-nas-lq/pretrained_musicfm/FM7.pt",
        }},
    }
    yaml_path = os.path.join(
        os.environ["DEPS_DIR"],
        "samantha/recipes/mir_benchmark/conf/multitask/multi_finetune_hdfs.yaml"
    )
    with open(yaml_path, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides=overrides)

    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip

    pl_module = hparams["pl_module"]
    pl_datamodule = hparams["pl_datamodule"]

    di = iter(pl_datamodule.train_dataloader())
    pl_module.to("cuda:0")
    with torch.cuda.amp.autocast(enabled=True):
        ## Test model
        batch = next(di)
        outputs = pl_module.training_step(batch, 0)

        ## Test dataloader
        # for i, batch in enumerate(di):
        #     print(i)
        #     if i > 100:
        #         break
        # from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip


def test_umm2_tag():
    os.environ["BYTED_RAY_CLUSTER"] = "a"
    overrides = {

    }
    yaml_path = os.path.join(
        os.environ["DEPS_DIR"],
        "samantha/recipes/mir2/conf/bigmusic/bigmusic_ft_umm2.yaml"
    )
    with open(yaml_path, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides=overrides)
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
    pl_module = hparams["pl_module"]


def test_musicfm_umm2_frontend():
    """An example of running both musicfm and umm2 frontends with the same
    audio input.
    """
    pl_module_musicfm = load_default_musicfm_model()
    dataloader_yaml_path = os.path.join(
        os.environ["DEPS_DIR"],
        "samantha/recipes/mir2/conf/bigmusic/bigmusic_ft_umm2.yaml"
    )
    os.environ["BYTED_RAY_CLUSTER"] = "a"
    overrides = {
        "val_params": {"batch_size": 2},
        "frontend": {"bypasses": []},
    }
    with open(dataloader_yaml_path, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides=overrides)
    pl_module_umm = hparams["pl_module"]

    pl_datamodule = hparams["pl_datamodule"]
    di = iter(pl_datamodule.val_dataloader())

    batch = next(di)
    audio = batch[0].to("cuda:0")

    # pl_module._compute(batch)
    front_umm = pl_module_umm.model.stages[0].to("cuda:0")
    output_umm = front_umm({"audio": audio})
    front_musicfm = pl_module_musicfm.model.stages[0].to("cuda:0")
    output_musicfm = front_musicfm({"audio": audio})

    # batch_train = next(iter(pl_datamodule.train_dataloader()))


def test_musicfm_umm2_yaml():
    """Run musicfm with umm2
    """
    yaml_path = os.path.join(
        os.environ["DEPS_DIR"],
        "samantha/recipes/mir2/conf/musicfm/multi_ft_umm2.yaml"
    )
    os.environ["BYTED_RAY_CLUSTER"] = "a"
    local_cache_dir = os.path.join(
        os.environ["AI_MUSIC_DIR"],
        "../../dump/ai_music/20240312.symbolic.dump/module_cache/umm"
    )
    overrides = {
        "val_params": {"batch_size": 2},
        "train_params": {"batch_size": 2},
        # "required_modules": None,
        "run_opts": {
            "cache_dir": local_cache_dir,
            "fast_dev_run": 3,
        },
        "config": {
            "num_hidden_layers": 12,
        }
    }
    with open(yaml_path, "r", encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides=overrides)

    pl_module = hparams["pl_module"]
    if pl_module.required_modules is not None:
        pl_module.load_required_modules()
    pl_module.model.to("cuda:0")

    pl_datamodule = hparams["pl_datamodule"]
    di = iter(pl_datamodule.train_dataloader())
    batch = next(di)

    for i, item in enumerate(batch[0]):
        batch[0][i] = item.to("cuda:0")

    # from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip
    # pl_module.training_step(batch, 0)
    # from recipes.mir_benchmark.pl_modules.multitask_finetune_pl import LitFinetuneMultitask
    output = pl_module.training_step(batch, 0)

    di_val = iter(pl_datamodule.val_dataloader())
    batch_val = next(di_val)
    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip