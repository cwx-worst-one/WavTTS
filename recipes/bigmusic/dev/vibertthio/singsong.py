import sys
import os
import glob
import json
import pickle
from pathlib import Path
import torchaudio
import torch.nn.functional as F

TMP_DATA_FP = "./tmp/raw_data.pkl"
EXAMPLE_BATCH_FP = "./tmp/example_batch.pkl"


def print_parent_classes(obj):
    def recursive_bases(cls):
        for base in cls.__bases__:
            print(base)
            recursive_bases(base)

    # Start with the direct parent classes of the object's class
    print(obj.__class__)
    recursive_bases(obj.__class__)


def print_types(x, indent=0, key=""):
    """
    Print types by Vibert.

    print_types(1)
    >>> int

    print_types("a")
    >>> str

    print_types([1,2])
    >>> list:[
    >>>   int
    >>> ]

    print_types({'a': 1, 'list': ['a', 'b']})
    >>> dict:{
    >>>   "a":int
    >>>   "list":list:[
    >>>     str
    >>>   ]
    >>> }
    """

    blanks = "  "
    if isinstance(x, list):
        print(blanks * indent + key + f"list({len(x)}): [")
        print_types(x[0], indent + 1)
        print(blanks * indent + "]")
    elif isinstance(x, dict):
        print(blanks * indent + key + "dict: {")
        for key, value in x.items():
            if isinstance(value, dict):
                print_types(value, indent + 1, f'"{key}": ')
            else:
                print_types(value, indent + 1, f'"{key}": ')
        print(blanks * indent + "}")
    else:
        # If the object is not a list or a dict, it's a primitive type
        # Print the type of the object
        print_this = f"{type(x).__name__}"
        if hasattr(x, "shape"):
            print_this += f"({tuple(x.shape)})"
        if hasattr(x, "dtype"):
            print_this += f"({x.dtype})"

        if len(str(x)) < 30:
            print_this = f"{print_this} -> {str(x)}"
        else:
            print_this = f"{print_this} -> {str(x)[:30]}..."
        print_this = print_this.replace("\n", "<newline>")

        print(blanks * indent + key + print_this)


def load_vocal_app_parquet_dataset(data_id=1096, tmp_data_fp=TMP_DATA_FP):
    from samantha.dataio.parquet import ParquetDataset

    dataset = ParquetDataset(
        data_id=data_id,
        extra_fields_in_data=["vocal", "acc"],
        detshuffle=True,
        # resampled=True,
        # shardshuffle=True,
        # use_pipe=True,
    )

    it = iter(dataset)
    first_data = next(it)
    print_types(first_data)

    # meta = first_data["meta"]
    # meta = json.loads(meta)
    # print_types(meta)
    # with open(tmp_data_fp, "wb") as pkf:
    #     pickle.dump(first_data, pkf)

    # breakpoint()
    return dataset, first_data


def load_transform():
    from transformers import BertTokenizer
    from recipes.bigmusic.datasets.vc_mix import VocalAppZhTransforms

    return VocalAppZhTransforms(
        # sample_rate=24000,
        # audio_key="audio",
        index_key="meta",
        min_duration=10,
        # max_duration=30,
        # min_volume_threshold=min_volume_threshold,
        # loudness_ratio_threshold=loudness_ratio_threshold,
        lyrics_confidence=0.6,
        # normalize_audio=False,
        tokenizer=BertTokenizer.from_pretrained("bert-base-chinese"),
        # frame_rate=frame_rate,
        # segment_method=segment_method,
        # include_intro=include_intro,
        # max_seg_per_track=max_seg_per_track,
        # segment_max_phone_len=segment_max_phone_len,
        # use_soda_gt_lyrics=True,
        # vocal_key=vocal_key,
        # style_key=style_key,
        # voice_clone_duration=voice_clone_duration,
        app_type="singsong",
        singsong_predict_type="BGM",
    )


def use_transform_on_data():
    dataset, first_data = load_vocal_app_parquet_dataset()
    transform = load_transform()
    transform(first_data)


def load_vocal_app_parquet_dataset_webpipeline():
    """
    Load a WebPipeline (ParquetDataset + a Transform)
    """
    from recipes.bigmusic.datasets.vc_mix import VppParquetDataset

    vp_dataset = VppParquetDataset(data_id=1096)
    breakpoint()

    return vp_dataset


def load_vocal_app_data_module(data_id=1096):
    """
    Load Vocal App DataModule, which loads multiple WebPipeline
    """
    from recipes.bigmusic.datasets.vc_mix import (
        MixVocalAppZhWebDataModule,
        vpp_collate_fn,
    )
    import functools

    app_type = "singsong"
    sample_rate = 24000
    datamodule = MixVocalAppZhWebDataModule(
        sample_rate=sample_rate,
        batch_size=12,
        shuffle_buffer_size=10,
        num_workers=0,
        pin_memory=True,
        collate_fn=functools.partial(vpp_collate_fn, app_type=app_type),
        use_pipe=True,
        tokenizer="phoneme",
        frame_rate=25,
        normalize_audio=False,
        region="CN",
        wds_dataset_names=[],
        wds_dataset_weights=[1],
        parquet_dataset_ids=[data_id],
        parquet_dataset_weights=[],
        use_dynamic_batch=False,
        segment_method="random",
        segment_max_phone_len=400,
        include_intro=False,
        max_seg_per_track=-1,
        buckets_in_sec=[20, 25, 30],
        vocal_key="vocal",
        style_key="acc",
        voice_clone_duration=-1,
        app_type=app_type,
        singsong_predict_type="BGM",  # FULL
    )
    
    return datamodule

    dataloader = datamodule.train_dataloader()
    # breakpoint()
    
    for batch in dataloader:
        print(batch.keys())
        with open(EXAMPLE_BATCH_FP, "wb") as pkf:
            pickle.dump(batch, pkf)
        break
    
    # it = iter(dataloader)
    # count = 100
    # while count > 0:
    #     batch = next(it)
    #     print(".", end="")
    #     count = count - 1


def test_resolve_data_urls():
    from samantha.dataio.utils import resolve_data_urls

    resolve_data_urls(data_id=1096)


def test_sami_tts_api():
    from sami_tts_api.engine import TtsEngine, generate_tts_config
    from sami_tts_api.sail import download_model

    fe_version = "42.0"
    fe_task = "tts_chinese_frontend_model"
    fe = download_model(fe_task, "/opt/tiger/sami_tts_api/models", fe_version)

    from recipes.datasets.mcc.sami_tokenizer import (
        SamiTokenizer,
        Phrase,
        convert_labels_to_text_id,
    )
    from recipes.datasets.mcc.sami_tokenizer import all_tones

    tk = SamiTokenizer()
    label = tk(["abc <n> abc"])
    print(label)

    phrase = Phrase(text="hi")
    tk.fill_phonemes(phrase)
    labels = (
        list(filter(lambda x: x != "", phrase.phonemes.split("\n")))
        if phrase.phonemes
        else None
    )
    text_id, _, _ = convert_labels_to_text_id(labels, phrase.prefix_tags)


def load_tmp_data(fp=TMP_DATA_FP):
    with open(fp, "rb") as pkf:
        tmp_data = pickle.load(pkf)
    return tmp_data


def test_group_utterances():
    from recipes.bigmusic.datasets.vc_mix import group_utterances
    from recipes.bigmusic.datasets.transforms.lyrics_segment import (
        Segment,
        group_by_fixed_length,
        group_by_variable_length,
        WordSegment,
    )
            
    
    # if not os.path.exists(TMP_DATA_FP):
    #     load_vocal_app_parquet_dataset(tmp_data_fp=TMP_DATA_FP)

    # data = load_tmp_data()
    # meta = data["meta"]
    # meta = json.loads(meta)
    
    with open ("./tmp/meta.json", "r") as f:
        meta = json.load(f) 
    utterances = meta["lyrics"]["result"][0]["utterances"]
    
    breakpoint()
    
    ss = []
    for u in utterances:
        ss.append(Segment.from_utt(u))
    print(ss)
    
    # this doesn't work really well
    # group_utterances(utterances, 0, 10)


def load_mulan_chinese():
    from recipes.musiclm.requires.mulan.mulan_infer_chinese import LitMuLanModule, mulan_inference
    ckpt_path = "/mnt/bn/ashaw-cn/vibertthio/repos/samantha/.module_cache/musiclm/mulan-step=024600-kaggle.ckpt"
    model = LitMuLanModule.load_from_checkpoint(ckpt_path) 
    test_input_texts = [
        "rock and roll",
        "pop",
        "this is mulan.",
    ]
    result = mulan_inference(model, test_input_texts[0])
    breakpoint()


def load_umm_tokenizer():
    import torchaudio                               
    from io import BytesIO  
    from recipes.umm.requires.model_initializer import init_stage3
    hpath_cn_speech = "hdfs://haruna/home/byte_speech_sv/zongyu.yin/logs/umm/stage3_music_chroma_vq32768x32/checkpoints/step=0030000.ckpt"
    result = init_stage3(hpath_cn_speech, local_rank=0, cache_dir=".module_cache")
    
    # # Check umm ouptuts
    # lit_module = result["Stage3"]
    # tmp_data = load_tmp_data()
    # wav_f = BytesIO(tmp_data['audio'])
    # wav, sr = torchaudio.load(wav_f)
    # wav = wav.to("cuda:0")
    # tokens = lit_module.wav2token(wav)
    # print(tokens.shape, tokens[:10])
    
    return result


def derive_umm_tokenizer_frequency():
    pass     


def load_semantic_module():
    from torch.optim import AdamW
    from transformers import LlamaConfig

    from samantha.models.flash_llama import LlamaForCausalLM
    from samantha.criterion.masked_loss import sequence_mask

    from recipes.bigmusic.lightning.semantic_modules import SemanticModule
    from recipes.musiclm.lightning.modules import MaskedCrossEntropy
    from recipes.musiclm.optim.lr_scheduler.warmup_cosine_lr import WarmupCosine

    import functools
    
    run_opts = {
        "learning_rate": 3.0e-4
    }
    extra_params = {
        "sample_rate": 24000,
        "lyrics_codebook_size": 400,
        "semantic_codebook_size": 32_768,
        "semantic_frame_rate": 25,
        "semantic_type": "bestrq",
        "mulan_embed_dim": 512,
        "hidden_size": 72,
        "intermediate_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 4,
        "pretrained_path": None,
        "input_embedders": ["vocal_audio"], # for singsong switch to ["mulan", "vocal_audio", "acc_audio"]
        "app_type": "singsong",  #or singsong singsong_inverse, singsong_inverse_vc
        "voice_clone_duration": -1,
        "singsong_predict_type": "BGM", # or "FULL"
        "tokenizer": "phoneme", # or wordpiece
        "parquet_dataset_ids": [1096],  # groupA: 96, karaoke: 95, karaoke val: 94
        "parquet_dataset_weights": [1],
    }
    
    model_config = LlamaConfig(
        vocab_size=1,
        hidden_size=72,
        intermediate_size=1024,
        num_hidden_layers=24,
        num_attention_heads=4,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        num_logits=32_768+2,
    )
    
    model_cls = functools.partial(
        LlamaForCausalLM,
        config=model_config,
    )
    
    
    optimizer_cls = functools.partial(
        AdamW,
        lr=run_opts.get("learning_rate"),
        weight_decay=0.05,
        betas=[0.9, 0.96],
        eps=0.00000001
    )
    
    scheduler_cls = functools.partial(
        WarmupCosine,
        init_lr=run_opts.get("learning_rate"),
        warmup_steps=2000,
        cycle_steps=30000,
        min_lr=run_opts.get("learning_rate") * 0.1,
    )

    sm = SemanticModule(
        model_cls=model_cls,
        optimizer_cls=optimizer_cls,
        scheduler_cls=scheduler_cls,
        criterion_cls=MaskedCrossEntropy,
        extra_params=extra_params,
        required_modules=[],
    )
    
    sm.requires = load_umm_tokenizer()
    
    # return sm
    
    sm.cuda()
    
    # TODO: load a batch of training data with "target_audio" and "vocal_audio" (with transforms above easily)
    # TODO: create another function to save the transformed batch into a pickle file
    with open("./tmp/example_batch.pkl", "rb") as pkf:
        batch = pickle.load(pkf)
    
    # TODO: use sm.prepare_training_inputs to create tokens for training
    batch["target_audio"] = batch["target_audio"].cuda()
    batch["vocal_audio"] = batch["vocal_audio"].cuda()
    batch["acc_audio"] = batch["acc_audio"].cuda()
    
    # for _, embedder in sm.input_embedders.items():
    #     embedder.cuda()
    
    # 1/ get target_values:
    # ```
    # dict: {
    #     "token_embeds": Tensor((2, 493, 72))(torch.float32) -> tensor([[[-0.0096, -0.0088,  0...
    #     "token_ids": Tensor((2, 493))(torch.int64) -> tensor([[32768, 18254, 18254, ...
    #     "token_seq_lengths": Tensor((2,))(torch.int64) -> tensor([493, 493], device='cud...
    # }
    # ```
    
    # target_values = sm.prepare_target_inputs(batch)
    
    
    # 2/ get inputs_embds: Tensor((2, 493, 72))(torch.float32) -> tensor([[[-7.1291e-03, -7.2208...
    
    # inputs_embeds = sm.prepare_inputs_embeddings(batch)
    
    
    # 3/ get training inputs:
    # hidden_size -> 72
    # ```
    # dict: {
    #     "model_inputs": dict: {
    #         "inputs_embeds": Tensor((2, 985, 72))(torch.float32) -> tensor([[[-7.1291e-03, -7.2208...
    #     }
    #     "target_lengths": Tensor((2,))(torch.int64) -> tensor([492, 492], device='cud...
    #     "target_ids": Tensor((2, 492))(torch.int64) -> tensor([[18254, 18254, 18254, ...
    #     "inputs_embeds": Tensor((2, 493, 72))(torch.float32) -> tensor([[[-7.1291e-03, -7.2208...
    # }
    # ```
    # Note that the "inputs_embeds" in "model_inputs" is actually "inputs_embeds" concat with "target_embeds"
    
    training_inputs = sm.prepare_training_inputs(batch)
    
    
    # forward pass -> model output:
    # """
    # dict: {
    #     "logits": Tensor((2, 985, 32770))(torch.float32) -> tensor([[[-0.0922, -0.1294,  0...
    #     "hidden_states": tuple -> (tensor([[[-0.0162,  0.0031, -...
    #          -> (2, 985, 72)
    # }
    # """
    model_inputs = training_inputs['model_inputs']
    model_output = sm.model(**model_inputs, output_hidden_states=True)
    logits = model_output["logits"]
    hidden_states = model_output['hidden_states']  # len(hidden_states) -> num_hidden_layers + 1
    last_hidden_states = hidden_states[-1]
    
    # get loss
    target_ids = training_inputs['target_ids']
    target_length = target_ids.size(1)
    target_logits = logits[:, -target_length:, :]
    loss_mask = sequence_mask(training_inputs['target_lengths'], max_len=target_ids.shape[1], device=target_ids.device)
    breakpoint()
    
    accu = ((target_logits.argmax(dim=-1) == target_ids).float() * loss_mask).sum() / loss_mask.sum() * 100
    loss = sm.criterion(target_logits, target_ids, loss_mask)
    
    # # Back prop
    loss.backward()
    
    return
    
    # Plot gradients or weights
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    vocal_embedder = sm.input_embedders['vocal_audio']
    vocal_embedder_para = next(vocal_embedder.embedder.parameters())
    print(vocal_embedder_para.shape)
    
    plt.figure(figsize=(10, 10))  # You can adjust the figure size as needed
    # weights
    plt.imshow(vocal_embedder_para.detach().cpu().numpy()[::150], cmap='viridis', aspect='auto')
    
    # or, grad
    # plt.imshow(vocal_embedder_para.grad.cpu().numpy()[::150], cmap='cividis', aspect='auto')
    
    plt.colorbar()  # Show the color scale
    plt.title('Heatmap of Small Floating Point Values')
    plt.xlabel('Column Index')
    plt.ylabel('Sampled Row Index (every 100th row)')
    plt.savefig('./tmp/vocal_embedder_weights.png', dpi=300, bbox_inches='tight')


def load_validation_dataset():
    pass


def test_sequence_mask():
    import torch
    from samantha.criterion.masked_loss import sequence_mask
    t = torch.tensor([2,3,4,1,0])
    sequence_mask(t, max_len=10)


def create_trainer():
    import os
    import pytorch_lightning.strategies
    from pytorch_lightning import Trainer
    trainer = Trainer(
        # strategy="ddp_find_unused_parameters_false",
        strategy="auto",
        fast_dev_run=True,
        logger=None,
        accelerator="auto",
        devices="auto",
        num_nodes=os.getenv("ARNOLD_WORKER_NUM", 1),
        deterministic=False,
        enable_model_summary=True,
        log_every_n_steps=10,
        max_steps=20,
        precision="16-mixed",
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        val_check_interval=5,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=2,
    )
    
    dm = load_vocal_app_data_module()
    sm = load_semantic_module()
    breakpoint()
    
    trainer.fit(sm, dm.train_dataloader(), dm.val_dataloader())


def find_all_checkpoints():
    import samantha.utils.hdfs_tools

    # list checkpoints available
    # find the relevant ones
    # return a structured data about these checkpoints
    pass


def test_save_audio():
    import torch
    from recipes.bigmusic.utils.upload import audio_tensor_to_bytes, upload_to_easycycle
    from recipes.bigmusic.callbacks.save_outputs import mix_two_audio_tensors

    t1 = torch.tensor([1,2])
    t2 = torch.tensor([1,2,3])
    print(mix_two_audio_tensors(t1, t2))

    t1 = torch.tensor([[1,2],[3,4]])
    t2 = torch.tensor([[1,2,3], [5,6,7]])
    print(mix_two_audio_tensors(t1, t2))

    # audio_bytes = audio_tensor_to_bytes(wav.cpu().float(), sample_rate)
    # metadata["audio_url"] = upload_to_easycycle(audio_bytes, f"{wav_file_name}.generated")

    breakpoint()


def simple_sine_wave():
    # only works in interact jupyter
    import IPython.display as ipd
    import matplotlib.pyplot as plt

    import torch
    from torchaudio.prototype.functional import adsr_envelope, oscillator_bank
    F0 = 344.0  # fundamental frequency
    DURATION = 1.1  # [seconds]
    SAMPLE_RATE = 16_000  # [Hz]

    NUM_FRAMES = int(DURATION * SAMPLE_RATE)
    def show(freq, amp, waveform, sample_rate, zoom=None, vol=0.3):
        t = (torch.arange(waveform.size(0)) / sample_rate).numpy()

        fig, axes = plt.subplots(4, 1, sharex=True)
        axes[0].plot(t, freq.numpy())
        axes[0].set(title=f"Oscillator bank (bank size: {amp.size(-1)})", ylabel="Frequency [Hz]", ylim=[-0.03, None])
        axes[1].plot(t, amp.numpy())
        axes[1].set(ylabel="Amplitude", ylim=[-0.03 if torch.all(amp >= 0.0) else None, None])
        axes[2].plot(t, waveform.numpy())
        axes[2].set(ylabel="Waveform")
        axes[3].specgram(waveform, Fs=sample_rate)
        axes[3].set(ylabel="Spectrogram", xlabel="Time [s]", xlim=[-0.01, t[-1] + 0.01])

        for i in range(4):
            axes[i].grid(True)
        pos = axes[2].get_position()
        plt.tight_layout()

        if zoom is not None:
            ax = fig.add_axes([pos.x0 + 0.01, pos.y0 + 0.03, pos.width / 2.5, pos.height / 2.0])
            ax.plot(t, waveform)
            ax.set(xlim=zoom, xticks=[], yticks=[])

        waveform /= waveform.abs().max()
        return ipd.Audio(vol * waveform, rate=sample_rate, normalize=False)
    freq = torch.full((NUM_FRAMES, 1), F0)
    amp = torch.ones((NUM_FRAMES, 1))

    waveform = oscillator_bank(freq, amp, sample_rate=SAMPLE_RATE)

    return show(freq, amp, waveform, SAMPLE_RATE, zoom=(1 / F0, 3 / F0))


def download_data_from_parquet_dataset(data_id=1096):

    def write_audio_in_batch_to_file(dump_dir: Path, batch: dict):
        uttid = batch["uttid"]
        audio_keys = ["wav", "vocal", "acc"]
        print(f"Writing audio in {audio_keys} in batch (uttid: {uttid}) to files")
        for audio_key in audio_keys:
            with open(dump_dir / f"{uttid}.{audio_key}.wav", "wb") as audio_file:
                audio_file.write(batch[audio_key])

    dump_dir = Path(f"./tmp/raw_data_{data_id}")
    dump_dir.mkdir(exist_ok=True, parents=True)
    dataset, first_data = load_vocal_app_parquet_dataset(data_id=data_id)
    it = iter(dataset)    
            
    count = 5
    for i in range(count):
        batch = next(it)
        write_audio_in_batch_to_file(dump_dir, batch)


def download_data_from_data_module(data_id=1096):
    
    def write_audio_in_batch_to_file(dump_dir, batch, sample_rate=24000):
        audio_keys = ["target_audio", "acc_audio", "vocal_audio"]    
        uttids = batch["uttid"]
        idx_segs = batch["idx_seg"]
        
        for idx, (uttid, idx_seg) in enumerate(zip(uttids, idx_segs)):
            
            vocal_audio = None
            target_audio = None
            
            for audio_key in audio_keys:
                batch_audio = batch[audio_key][idx]
                
                if batch_audio.ndim not in [1, 2]:
                    raise ValueError(f"Cannot handle audio_data.ndim {batch_audio.ndim}")
                
                if batch_audio.ndim == 1:
                    batch_audio = batch_audio.unsqueeze(0)
                    
                torchaudio.save(
                    filepath=str(dump_dir / f"{uttid}.{idx_seg}.{audio_key}.wav"),
                    src=batch_audio,
                    sample_rate=sample_rate
                )
                
                if audio_key == "target_audio":
                    target_audio = batch_audio
                elif audio_key == "vocal_audio":
                    vocal_audio = batch_audio
            
            
            # add target_audio and vocal_audio into a new file
            torchaudio.save(
                filepath=str(dump_dir / f"{uttid}.{idx_seg}.mixed_audio.wav"),
                src=mix_audio(vocal_audio, target_audio),
                sample_rate=sample_rate
            )
            
    
    dm = load_vocal_app_data_module(data_id=data_id)
    
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()[0]
    for stage, dataloader in zip(["train", "val"], [train_dataloader, val_dataloader]):
        dump_dir = Path(f"./tmp/singsong_datamodule_{data_id}_{stage}")
        dump_dir.mkdir(exist_ok=True, parents=True)
        
        it = iter(dataloader)
        count = 5
        for i in range(count):
            batch = next(it)
            write_audio_in_batch_to_file(dump_dir, batch)


def mix_audio(a_1, a_2, mode="max"):
    assert a_1.ndim == 2
    assert a_2.ndim == 2
    assert a_1.shape[0] == 1
    assert a_2.shape[0] == 1
    
    if mode == "max":
        max_length = max(a_1.shape[1], a_2.shape[1])
        longer_a = a_1 if a_1.shape[1] >= a_2.shape[1] else a_2
        shorter_a = a_1 if a_1.shape[1] < a_2.shape[1] else a_2
        shorter_a = F.pad(shorter_a, (0, max_length - shorter_a.shape[1]), "constant", 0)
        
        mixed = longer_a + shorter_a
        
        return mixed / mixed.max()
    
    if mode == "min":
        min_length = min(a_1.shape[1], a_2.shape[1])
        mixed = a_1[:, :min_length] + a_2[:, :min_length]
        return mixed / mixed.max()


def mix_audio_in(dir):
    
    def get_target_audio(vocal_fp):
        return vocal_fp.parent / vocal_fp.name.replace("vocal_audio", "target_audio")

    def get_mixed_audio(vocal_fp):
        return vocal_fp.parent / vocal_fp.name.replace("vocal_audio", "mixed_audio")
    
    assert Path(dir).is_dir()

    pattern = str(Path(dir) / "*.vocal_audio.wav")
    vocal_audio_fps = sorted(glob.glob(pattern))
    for vfp in vocal_audio_fps:
        vfp = Path(vfp)
        tfp = get_target_audio(vfp)

        vocal_audio, sr_vocal = torchaudio.load(vfp)
        target_audio, sr_target = torchaudio.load(tfp)
        assert sr_vocal == sr_target, "Sample rate not matched"
        mixed_audio = mix_audio(vocal_audio, target_audio, mode="max")
        torchaudio.save(
            filepath=str(get_mixed_audio(vfp)),
            src=mixed_audio,
            sample_rate=sr_vocal
        )


if __name__ == "__main__":
    
    # os.chdir("/mnt/bn/ashaw-cn/vibertthio/repos/samantha")
    # sys.path.append("/mnt/bn/ashaw-cn/vibertthio/repos/samantha")
    
    # load_vocal_app_parquet_dataset()
    # use_transform_on_data()
    # test_group_utterances()
    # load_mulan_chinese()
    # load_umm_tokenizer()
    # load_vocal_app_data_module()
    # breakpoint()
    # load_semantic_module()
    # create_trainer()
    # test_save_audio()
    
    # mix_audio_in("/mnt/bn/ashaw-lq/vibertthio/data/singsong_inputs/audio/singsong_datamodule_1096_train")
    # mix_audio_in("/mnt/bn/ashaw-lq/vibertthio/data/singsong_inputs/audio/singsong_datamodule_1096_val")
    
    for data_id in [1096, 959]:
        # download_data_from_parquet_dataset(data_id)
        download_data_from_data_module(data_id=data_id)
