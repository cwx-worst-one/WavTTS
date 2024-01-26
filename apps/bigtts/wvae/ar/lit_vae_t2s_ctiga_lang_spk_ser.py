import logging

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.profilers import PassThroughProfiler
from s3a.providers.ctiga.utils.generation import InferenceParams
from speechbrain.nnet.activations import Softmax as log_softmax
from tqdm import tqdm

from samantha.utils.model_metric import ModelMetric

logger = logging.getLogger(__name__)


def sequence_mask(seq_lens, max_len=None, device="cpu"):
    b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device)  # [1, t]
    mask = mask < (seq_lens.unsqueeze(1))  # [1, t] + [b, 1] = [b, t]
    mask = mask.float()
    return mask


class VAET2SLangSpkSerModule(pl.LightningModule):
    def __init__(
        self,
        model_cls,
        logits_criterion_cls,
        dense_criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=True,
        stop_token_loss_weight=1.0,
        use_lang_id=False,
        use_spk_id=False,
        spk_type="add",
        use_phoneme_loss=False,
        use_ctc_loss=False,
        freeze_text_encoder=False,
        resume_ckpt_path=None,
        use_lang_grloss=False,
        input_type="2dim",
        use_ser_tag=False,
        use_ser_tag_loss=False,
        use_ref_enc=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_cls()
        self.logits_criterion = logits_criterion_cls()
        self.dense_criterion = dense_criterion_cls()
        self.requires = {}
        self.use_lang_id = use_lang_id
        logger.info(f"use_lang_id: {self.use_lang_id}")
        self.use_spk_id = use_spk_id
        logger.info(f"use_spk_id: {self.use_spk_id}")
        self.use_phoneme_loss = use_phoneme_loss
        logger.info(f"use_phoneme_loss: {use_phoneme_loss}")
        self.use_ctc_loss = use_ctc_loss
        logger.info(f"use_ctc_loss: {use_ctc_loss}")
        self.freeze_text_encoder = freeze_text_encoder
        logger.info(f"freeze_text_encoder: {freeze_text_encoder}")
        self.spk_type = spk_type
        logger.info(f"spk_type: {spk_type}")
        self.use_lang_grloss = use_lang_grloss
        logger.info(f"use_lang_grloss: {use_lang_grloss}")
        self.input_type = input_type
        logger.info(f"input_type: {input_type}")
        self.use_ser_tag = use_ser_tag
        logger.info(f"use_ser_tag: {use_ser_tag}")
        self.use_ser_tag_loss = use_ser_tag_loss
        logger.info(f"use_ser_tag_loss: {use_ser_tag_loss}")
        self.use_ref_enc = use_ref_enc
        logger.info(f"use_ref_enc: {use_ref_enc}")

        if checkpointing:
            self.model.gradient_checkpointing_enable()
        self.stop_token_loss_weight = stop_token_loss_weight

        if self.use_ctc_loss:
            self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True)

        logger.info(f"resume_ckpt_path: {resume_ckpt_path}")
        if resume_ckpt_path:
            state_dict = torch.load(resume_ckpt_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
            new_state_dict = []
            new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict, strict=False)
            logger.info(
                "Loading state_dict from {} successfully".format(resume_ckpt_path)
            )

    def setup(self, stage: str) -> None:
        self.model_metric = ModelMetric(
            precision=self.trainer.precision, model_obj_or_objs=self.model
        )

        if stage == "fit" and not self.requires:
            self.load_required_modules()

    def load_required_modules(self):
        for name, (hpath, initializer) in self.hparams.required_modules.items():
            self.requires.update(initializer(hpath, local_rank=self.local_rank))

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, False)

    @property
    def profiler(self):
        return getattr(self.trainer, "profiler") or PassThroughProfiler()

    def training_step(self, batch, batch_idx):
        with self.profiler.profile("[LightningModule]CoarseModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                frontend_inputs = {
                    "phone": batch["phone"],
                    "tone": batch["tone"],
                    "phonetone": batch["phonetone"],
                }

                bns, stop_tokens = batch["bn"], batch["stop_token"]
                text_lens, bn_lens = batch["text_lens"], batch["bn_lens"]
                max_text_len = max(text_lens)

                if self.use_ser_tag:
                    ser_tags, _ = self.get_ser_tags(batch["wav"], batch["wav_lens"])
                else:
                    ser_tags = None

                if self.use_spk_id and self.spk_type == "concat":
                    seq_lens = text_lens + 1 + bn_lens
                    if self.use_ser_tag:
                        seq_lens += 1
                    seq_len = max(seq_lens)
                    text_spk_lens = text_lens + 1
                    if self.use_ser_tag:
                        text_spk_ser_lens = text_lens + 2
                else:
                    seq_lens = text_lens + bn_lens
                    if self.use_ser_tag:
                        seq_lens += 1
                    seq_len = max(seq_lens)
                    text_spk_lens = text_lens
                    if self.use_ser_tag:
                        text_spk_ser_lens = text_lens + 1

                loss_mask = sequence_mask(seq_lens, device="cuda")
                text_loss_mask = sequence_mask(text_lens, device="cuda")
                if self.use_ser_tag:
                    text_spk_ser_loss_mask = sequence_mask(
                        text_spk_ser_lens, device="cuda"
                    )
                    pad_text_spk_ser_loss_mask = F.pad(
                        text_spk_ser_loss_mask,
                        (0, seq_len - text_spk_ser_loss_mask.shape[1]),
                        "constant",
                        0,
                    )
                    z_loss_mask = loss_mask - pad_text_spk_ser_loss_mask
                else:
                    text_spk_loss_mask = sequence_mask(text_spk_lens, device="cuda")
                    pad_text_spk_loss_mask = F.pad(
                        text_spk_loss_mask,
                        (0, seq_len - text_spk_loss_mask.shape[1]),
                        "constant",
                        0,
                    )
                    z_loss_mask = loss_mask - pad_text_spk_loss_mask

        with self.profiler.profile("[LightningModule]CoarseModule.model_forward"):
            ret_dict = self.model(
                frontend_inputs,
                bns,
                text_lens,
                bn_lens,
                lang_seqs=batch["lang_seq"],
                spk_seqs=batch["spk_seq"],
                byt5_embeds=batch.get("byt5"),
                byt5_lens=batch.get("byt5_lens"),
                tag_ids=batch.get("tag_id"),
                ser_tags=ser_tags,
                crop_bn=batch.get("crop_bn"),
                spk_embd_masks=batch.get("spk_embd_masks"),
            )
            pred_stop_token = ret_dict["stop_token"]
            pred_dense = ret_dict["dense"]
            attn_weights = ret_dict["attn_weights"]

        pred_stop_token = pred_stop_token[:, 0 : seq_len - 1, :]
        pred_dense = pred_dense[:, 0 : seq_len - 1, :]

        bsz, bn_t, bn_c = bns.shape
        targets_dense = []
        for i in range(bsz):
            if self.use_ser_tag:
                targets_dense.append(
                    F.pad(
                        bns[i, : bn_lens[i], :],
                        (
                            0,
                            0,
                            text_spk_ser_lens[i],
                            seq_len - (bn_lens[i] + text_spk_ser_lens[i]),
                        ),
                        "constant",
                        0,
                    )
                )
            else:
                targets_dense.append(
                    F.pad(
                        bns[i, : bn_lens[i], :],
                        (
                            0,
                            0,
                            text_spk_lens[i],
                            seq_len - (bn_lens[i] + text_spk_lens[i]),
                        ),
                        "constant",
                        0,
                    )
                )
        targets_dense = torch.stack(targets_dense)[:, 1:seq_len, :]

        target_m, target_logs = torch.split(targets_dense, bn_c // 2, dim=-1)
        pred_m, pred_logs = torch.split(pred_dense, bn_c // 2, dim=-1)

        # kl_loss
        kl_loss = self.dense_criterion(
            pred_m,
            pred_logs,
            target_m.detach(),
            target_logs.detach(),
            z_mask=z_loss_mask[:, 1:],
        )

        # stop_token_loss
        if self.use_spk_id and self.spk_type == "concat":
            if self.use_ser_tag:
                target_stop_token = torch.cat(
                    (stop_tokens[:, 0:1], stop_tokens[:, :seq_len]), dim=-1
                )
            else:
                target_stop_token = stop_tokens[:, :seq_len]
        else:
            if self.use_ser_tag:
                target_stop_token = stop_tokens[:, :seq_len]
            else:
                target_stop_token = stop_tokens[:, 1:seq_len]

        stop_token_loss = self.logits_criterion(
            pred_stop_token.float(), target_stop_token, mask=z_loss_mask[:, 1:]
        )
        stop_token_accu = (
            (
                (pred_stop_token.argmax(dim=-1) == target_stop_token).float()
                * z_loss_mask[:, 1:]
            ).sum()
            / z_loss_mask[:, 1:].sum()
            * 100
        )

        # phoneme loss
        if self.input_type == "2dim":
            targets_logits = batch["phone"]
        elif self.input_type == "1dim":
            targets_logits = batch["phonetone"]
        else:
            raise NotImplementedError
        if self.use_phoneme_loss:
            # pred_logits = ret_dict["logits"][:, :max_text_len, :]
            # phoneme_loss = self.logits_criterion(pred_logits.float(), targets_logits, mask=text_loss_mask)
            pred_logits = ret_dict["logits"][:, : max_text_len - 1, :]
            targets_logits = targets_logits[:, 1:]
            text_loss_mask = text_loss_mask[:, 1:]
            phoneme_loss = self.logits_criterion(
                pred_logits.float(), targets_logits, mask=text_loss_mask
            )
            phoneme_accu = (
                (
                    (pred_logits.argmax(dim=-1) == targets_logits).float()
                    * text_loss_mask
                ).sum()
                / text_loss_mask.sum()
                * 100
            )
        else:
            phoneme_loss = torch.tensor(0.0)
            phoneme_accu = torch.tensor(0.0)

        # ctc loss
        if self.use_ctc_loss:
            ctc_loss = self.add_ctc_loss(
                attn_weights, text_lens, batch["bpe_lens"], bns.device
            )
        else:
            ctc_loss = torch.tensor(0.0)

        # ser_tag loss
        if self.use_ser_tag_loss:
            pred_ser_tag_logits = []
            for i in range(bsz):
                pred_ser_tag_logits.append(
                    ret_dict["ser_tag_logits"][i, text_spk_lens[i] - 1, :]
                )
            pred_ser_tag_logits = torch.stack(pred_ser_tag_logits)
            targets_ser_tag_logits = ser_tags

            ser_tag_loss = self.logits_criterion(
                pred_ser_tag_logits.float(), targets_ser_tag_logits
            )
            ser_tag_accu = (
                (
                    (
                        pred_ser_tag_logits.argmax(dim=-1) == targets_ser_tag_logits
                    ).float()
                ).sum()
                / targets_ser_tag_logits.shape[0]
                * 100
            )
        else:
            ser_tag_loss = torch.tensor(0.0)
            ser_tag_accu = torch.tensor(0.0)

        # lang_grloss
        if self.use_lang_grloss:
            lang_loss_mask = sequence_mask(bn_lens, device="cuda")
            target_lang = batch["lang_seq"]
            pred_lang = ret_dict["lang_output"]
            lang_token_loss = self.logits_criterion(
                pred_lang.float(), target_lang, mask=lang_loss_mask
            )
            lang_token_accu = (
                (
                    (pred_lang.argmax(dim=-1) == target_lang).float() * lang_loss_mask
                ).sum()
                / lang_loss_mask.sum()
                * 100
            )
        else:
            lang_token_loss = torch.tensor(0.0)
            lang_token_accu = torch.tensor(0.0)

        total_loss = (
            ctc_loss
            + kl_loss
            + stop_token_loss * self.stop_token_loss_weight
            + phoneme_loss
            + lang_token_loss
            + ser_tag_loss
        )

        batch_tokens = bsz * seq_len

        self.log_dict(
            {
                "ctc_loss": ctc_loss.item(),
                "kl_loss": kl_loss.item(),
                "phoneme_loss": phoneme_loss.item(),
                "phoneme_accu": phoneme_accu.item(),
                "stop_token_loss": stop_token_loss.item(),
                "stop_token_accu": stop_token_accu.item(),
                "loss": total_loss.item(),
                "bsz": bsz,
                "seqlen": seq_len,
                "batch_tokens": batch_tokens,
                "training/loss": total_loss.item(),
                "lang_token_loss": lang_token_loss.item(),
                "lang_token_accu": lang_token_accu.item(),
                "ser_tag_loss": ser_tag_loss.item(),
                "ser_tag_accu": ser_tag_accu.item(),
            },
            prog_bar=True,
            sync_dist=True,
        )

        self.model_metric.update(
            num_tokens=batch_tokens,
            stage=self.trainer.state.stage,
            model_kwargs=dict(batch_size=bsz, seqlen=seq_len),
        )
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            if self.trainer.global_rank == 0 and attn_weights is not None:
                self.log_attention(attn_weights, batch_idx)

            metric = self.model_metric.compute(self.trainer.global_step)
            self.log_dict(metric, sync_dist=True, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        params = []
        for name, p in self.model.named_parameters():
            if self.freeze_text_encoder and name.startswith("text_encoder"):
                p.requires_grad = False
                continue
            params.append({"params": [p]})

        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def add_ctc_loss(self, alignments, text_lens, bpe_lens, device):
        #### ctc loss ####
        neg_blank_loss, ctc_loss = 0.0, 0.0
        ctc_target_lens = bpe_lens - 1  # remove sep
        ctc_targets = (
            torch.arange(ctc_target_lens.max())
            .to(device)
            .unsqueeze(0)
            .repeat(alignments.shape[0], 1)
            + 1
        )  # remove bos
        ctc_input_lens = text_lens - 1  # remove eos
        blank_mask = sequence_mask(
            ctc_input_lens, max_len=ctc_input_lens.max(), device=device
        )
        sub_aligns = []
        # 将 wave 部分切片
        for i, sub_align in enumerate(alignments):
            align = sub_align[:, : ctc_input_lens[i], :]
            sub_aligns.append(align)  # [head, t_bn, t]
        # 补长度
        max_len = max([sub_align.size(1) for sub_align in sub_aligns])
        for i, sub_align in enumerate(sub_aligns):
            sub_align = (
                torch.where(
                    torch.isinf(sub_align), torch.zeros_like(sub_align) - 1e3, sub_align
                )
                .float()
                .log_softmax(dim=-1)
            )  # log_softmax
            sub_align = torch.where(
                torch.isinf(sub_align), torch.zeros_like(sub_align) - 15, sub_align
            ).clamp(
                -15
            )  # inf 处理
            sub_align = F.pad(sub_align, (0, 0, 0, max_len - sub_align.size(-2)))
            sub_aligns[i] = sub_align
        sub_aligns = torch.stack(sub_aligns, dim=0)  # [b, head, t_bn, t_seq]
        _b, _head, _t_bn, _t_q = sub_aligns.size()
        sub_aligns = sub_aligns.view(_b * _head, _t_bn, _t_q)  # [b*head, t_bn, t_seq]

        # head 匹配
        ctc_targets = (
            ctc_targets.unsqueeze(1).repeat(1, _head, 1).reshape(_b * _head, -1)
        )
        ctc_target_lens = ctc_target_lens.unsqueeze(1).repeat(1, _head).reshape(-1)
        ctc_input_lens = ctc_input_lens.unsqueeze(1).repeat(1, _head).reshape(-1)
        blank_mask = blank_mask.unsqueeze(1).repeat(1, _head, 1).reshape(_b * _head, -1)

        # blank loss to prevent blank prediction
        neg_blank_loss += (sub_aligns[:, :, 0] * blank_mask).sum() / blank_mask.sum()

        # 均衡长度 align loss
        guided_targets = ((ctc_target_lens - 1) / ctc_input_lens).unsqueeze(
            1
        ) * torch.arange(ctc_input_lens.max()).unsqueeze(0).to(
            device
        ) + 1  # [b*head, t]
        guided_targets = (
            torch.round(guided_targets).long().clamp(0, sub_aligns.size(-1) - 1)
        )
        guided_loss = self.logits_criterion(
            sub_aligns, guided_targets, blank_mask, log_softmax=False
        )
        sub_aligns = sub_aligns.permute(
            1, 0, 2
        )  # [b*head, t_bn, t_seq] -> [t_bn, b*head, t_seq]
        ctc_loss += self.ctc_loss(
            sub_aligns,
            ctc_targets,
            input_lengths=ctc_input_lens,
            target_lengths=ctc_target_lens,
        )

        ctc_scale = min(1, self.trainer.global_step / 10_000) * 0.1
        guided_scale = max(0, (1 - self.trainer.global_step / 20_000)) * 0.1
        ctc_total_loss = (
            ctc_scale * (ctc_loss + neg_blank_loss) + guided_scale * guided_loss
        )
        return ctc_total_loss

    @torch.no_grad()
    def log_attention(self, attn_weights, batch_idx):
        attn_weights = attn_weights.cpu().detach().numpy()[0]
        for i in range(attn_weights.shape[0]):
            # 取argmax操作，得到二值矩阵
            single_head_weight = attn_weights[i]
            len_src, len_tgt = single_head_weight.shape
            argmax_attn_weights = np.zeros_like(single_head_weight)
            if len_src >= len_tgt:
                argmax_attn_weights[
                    np.arange(len_src), single_head_weight.argmax(1)
                ] = 1
            else:
                argmax_attn_weights[
                    single_head_weight.argmax(0), np.arange(len_tgt)
                ] = 1

            # 绘制注意力权重图
            fig, ax = plt.subplots()
            # plt.imshow(argmax_attn_weights, cmap='Blues')
            im = ax.imshow(argmax_attn_weights, cmap="Blues")
            plt.title("Attention Alignment Matrix")
            plt.xlabel("Source Tokens")
            plt.ylabel("Target Tokens")
            # plt.colorbar(im)
            # 将图像数据记录到WandB
            wandb.log({f"Attention Alignment Matrix {i}": wandb.Image(fig)})
            plt.clf()

    @torch.no_grad()
    def get_semantic_codes(self, wavs_16k, wavs_16k_len):
        infer_fn = self.requires["semantic_infer_fn"]
        semantic_codes, semantic_length = infer_fn(
            self.requires["semantic"],
            self.requires["centroids"],
            wavs_16k,
            wavs_16k_len,
            wavs_16k.device,
        )
        return semantic_codes, semantic_length

    @torch.no_grad()
    def get_codec_codes(self, x):
        output = self.requires["ss"](x)[2]
        output = torch.stack(output, dim=2)  # [b, t, n_code]
        return output

    @torch.no_grad()
    def get_ser_tags(self, wavs, wav_lens):
        # wav2vec2
        raw_outputs = self.requires["wav2vec2"].forward(wavs)

        # avg_pool
        outputs = []
        for snt_id in range(raw_outputs.shape[0]):
            actual_size = int(torch.round(wav_lens[snt_id] * raw_outputs.shape[1]))
            outputs.append(torch.mean(raw_outputs[snt_id, 0:actual_size, ...], dim=0))
        outputs = torch.stack(outputs)
        outputs = outputs.view(outputs.shape[0], -1)

        # output_mlp
        out_tag_orig = self.requires["output_mlp"].forward(outputs)
        out_deg_orig = self.requires["output_deg"].forward(outputs)

        classifier = log_softmax(apply_log=True)
        out_tag = classifier(out_tag_orig).exp()
        out_deg = classifier(out_deg_orig).exp()

        ser_tags = torch.argmax(out_tag, dim=-1)
        ser_degrees = torch.argmax(out_deg, dim=-1)

        return ser_tags, ser_degrees

    @torch.no_grad()
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
        return e_x / e_x.sum()

    @torch.no_grad()
    def inference_from_text(self, batch, tokenizer):
        bns = batch["bn"]
        text_lens, bn_lens = batch["text_lens"], batch["bn_lens"]
        lang_seq, infer_lang_id = batch["lang_seq"], batch["infer_lang_id"]
        spk_seq, infer_spk_id = batch["spk_seq"], batch["infer_spk_id"]
        if (
            self.use_spk_id
            and (self.spk_type == "concat" or self.spk_type == "cln")
            and spk_seq is not None
        ):
            spk_seq = torch.from_numpy(np.asarray([infer_spk_id]))
            spk_seq = spk_seq.unsqueeze(0)
            spk_seq = spk_seq.to(bns.device)

        frontend_inputs = {
            "phone": batch["phone"],
            "tone": batch["tone"],
            "phonetone": batch["phonetone"],
        }

        b = bns.shape[0]

        # llama style inference
        self.model.params.use_cache = True
        start_pos = 0

        inference_params = InferenceParams(
            max_sequence_len=8000, max_batch_size=b, fused_ft_kernel=False
        )
        semantic_outputs = []
        z_list = []

        max_step_ = text_lens[0] * 10 - bn_lens[0]
        max_step = 6000

        with torch.autocast(device_type="cuda", enabled=True):
            for i in tqdm(range(max_step)):
                if i == 0:
                    model_outputs = self.model(
                        frontend_inputs,
                        bns,
                        text_lens,
                        bn_lens,
                        start_pos=start_pos,
                        inference_params=inference_params,
                        lang_seqs=lang_seq,
                        spk_seqs=spk_seq,
                        bpe_seqs=batch.get("bpe_seq", None),
                        bpe_lens=batch.get("bpe_lens", None),
                        tag_ids=batch.get("tag_id", None),
                        crop_bn=batch.get("bn", None),
                    )
                    input_len = text_lens[0] + bn_lens[0]
                    if self.use_spk_id and self.spk_type == "concat":
                        input_len += 1
                else:
                    model_outputs = self.model(
                        frontend_inputs,
                        bns,
                        text_lens,
                        bn_lens,
                        start_pos=start_pos,
                        use_cache=True,
                        inference_params=inference_params,
                        lang_seqs=lang_seq,
                        spk_seqs=spk_seq,
                        bpe_seqs=batch.get("bpe_seq", None),
                        bpe_lens=batch.get("bpe_lens", None),
                        tag_ids=batch.get("tag_id", None),
                        crop_bn=batch.get("bn", None),
                    )
                    input_len = 1
                    z_list.append(model_outputs["bn_in_z"])

                pred_stop_token = model_outputs["stop_token"][:, -1, :]
                samples = torch.argmax(pred_stop_token)
                if (i > 10 and samples.item() == 1) or (i == max_step_):
                    break

                pred_dense = model_outputs["dense"][:, -1:, :]

                start_pos += input_len
                inference_params.sequence_len_offset = start_pos

                # next infer
                semantic_outputs.append(pred_dense)
                bns = pred_dense
                bn_lens[0] = 1

                if self.use_lang_id:
                    lang_id = torch.from_numpy(np.asarray([infer_lang_id]))
                    lang_id = lang_id.unsqueeze(0)
                    lang_id = lang_id.to(pred_dense.device)
                    lang_seq = lang_id

                spk_seq = None
                if self.use_spk_id and (
                    self.spk_type == "add" or self.spk_type == "cln"
                ):
                    spk_id = torch.from_numpy(np.asarray([infer_spk_id]))
                    spk_id = spk_id.unsqueeze(0)
                    spk_id = spk_id.to(pred_dense.device)
                    spk_seq = spk_id

        z_outputs = torch.cat(z_list, dim=1)  # [b, t, c]
        semantic_outputs = torch.cat(semantic_outputs, dim=1)  # [b, t, c]
        self.model.params.use_cache = False
        return z_outputs, semantic_outputs

    predict = inference_from_text
