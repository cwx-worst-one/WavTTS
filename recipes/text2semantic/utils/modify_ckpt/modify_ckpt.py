def fix_model_blacklist(orig_ckpt_path,fix_ckpt_path):
    import functools
    import torch
    ckpt=torch.load(orig_ckpt_path,map_location=torch.device("cuda"))

    from recipes.text2semantic.modules.llama.wfvae_ctiga_llama_lang_spk import VAELLaMaLangSpk
    from recipes.text2semantic.modules.llama.wfvae_ctiga_llama_lang_spk import ModelArgs
    from recipes.text2semantic.lit_modules.llama.lit_vae_t2s_ctiga_lang_spk import VAET2SLangSpkModule

    new_params = functools.partial(ModelArgs, 
            dim=1536, n_layers=30, n_heads=16, vocab_size=1024, out_dim=32, multiple_of=256, norm_eps=1e-06, max_batch_size=32, max_seq_len=6000, attn_pdrop=0.0, resid_pdrop=0.0, sparse=False, checkpointing=False, num_res=-1, num_coarse=-1, num_fine=-1, audio_tokens_num=1024, phone_tokens_num=200, lang_vocab_size=200, phone_embed_dim=512, tone_embed_dim=64, n_phone=1000, n_tone=30, spk_vocab_size=8192
    )

    new_model_cls = functools.partial(VAELLaMaLangSpk, params=new_params, use_spk_id=True)
    ckpt["hyper_parameters"]["model_cls"] = new_model_cls

    new_lit_module = functools.partial(VAET2SLangSpkModule, model_cls=new_model_cls, logits_criterion_cls=

    # model_cls: !ref <model_cls>
    # optimizer_cls: !ref <optimizer_cls>
    # scheduler_cls: !ref <scheduler_cls>
    # logits_criterion_cls: !name:recipes.text2semantic.modules.loss.MaskedCrossEntropy
    # dense_criterion_cls: !name:recipes.text2semantic.modules.loss.MaskedKLLossPP
    # checkpointing: !ref <run_opts[checkpointing]>
    # required_modules: {}
    # stop_token_loss_weight: !ref <run_opts[stop_token_loss_weight]>
    # use_lang_id: !ref <run_opts[use_lang_id]>
    # use_spk_id: !ref <run_opts[use_spk_id]>
    # resume_ckpt_path: !ref <run_opts[resume_ckpt_path]>


    print(ckpt["hyper_parameters"].keys())

    # torch.save(ckpt, fix_ckpt_path)

if __name__=="__main__":
    import os
    src="/mnt/bn/huangzhiying-nas-speech2speech-volume1/exp/samantha_bigtts_mix/text2semantic/pretrain_labv3_punc_ll-rp-fq-others_bt17500_32A100_langid_multidimv3/checkpoints/epoch=00-step=95000-kl_loss=0.66.ckpt"
    tgt=f"{os.path.splitext(os.path.basename(src))[0]}-fix.ckpt"
    dst = os.path.join(
        os.path.dirname(src),
        tgt
    )
    fix_model_blacklist(src, dst)
