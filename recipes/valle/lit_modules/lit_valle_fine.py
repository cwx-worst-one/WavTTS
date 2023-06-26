import torch

from .lit_valle_coarse import ValleCoarse, compute_loss
import torch.nn.functional as F

class ValleFine(ValleCoarse):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
        resume_ckpt_path=None,
    ):
        super().__init__(
            model_cls,
            criterion_cls,
            optimizer_cls,
            scheduler_cls,
            required_modules,
            checkpointing,
            extra_params,
            resume_ckpt_path,
        )

    def training_step(self, batch, batch_idx):
        with self.profiler.profile("[LightningModule]LLMModule.model_forward"):
            utt_ids, seqs, seq_lens, pos_ids, seq_sen_ids, full_seqs = batch
            b, t, _ = full_seqs.shape
            logits, target_x, target_mask = self.model(full_seqs.transpose(1,2), seq_sen_ids, pos_ids)
        # calculate loss
        loss = compute_loss(logits, target_x, mask=target_mask)
        # self.log("train_loss", loss, logger=True, sync_dist=True, prog_bar=True)
        self.log_dict({
                "train_loss": loss,
                "batch_size": b
            },
            logger=True, sync_dist=True, prog_bar=True, batch_size=b)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     with torch.autocast(device_type="cuda", enabled=False):
    #         wavs, text_ids, wav_lens, text_lens, utt_ids = batch
    #         _, _, pos_ids, seq_sen_ids, full_seqs = self.prepare_feature(
    #             wavs, text_ids, wav_lens, text_lens, "cuda")

    #     # print("pos_ids: ", pos_ids.shape, pos_ids)
    #     # print("seq_sen_ids: ", seq_sen_ids.shape, seq_sen_ids)
    #     # print("full_seqs: ", full_seqs.shape, full_seqs)
    #     logits, target_x, target_mask = self.model(full_seqs.transpose(1,2), seq_sen_ids, pos_ids)
    #     loss = compute_loss(logits, target_x, mask=target_mask)
    #     # print("loss: ", loss)
    #     # exit()
    #     self.log_dict( {"val_loss": loss}, logger=True, sync_dist=True, prog_bar=True)
    #     return loss

    # @torch.no_grad()
    # def prepare_feature(self, wavs, text_ids, wav_lens, text_lens, device):
    #     batch = len(wavs)
    #     text_ids = self.tokenizer.tokenize(text_ids, "inputs")
    #     wav_ids = self.tokenizer.tokenize(self.get_codec_codes(wavs), "targets")

    #     num_rvqs = wav_ids[0].shape[1]

    #     # wav_id_len = wav_len / (0.0125 * 24000) = wav_len / 300
    #     wav_id_lens = wav_lens // 300 # 300 = 0.0125 * 24000

    #     seqs = []
    #     seq_lens = []
    #     seq_sen_ids = []
    #     pos_ids = []
    #     full_seqs = []
    #     max_seq_len = max([w+t for w, t in zip(wav_id_lens, text_lens)])
    #     for i in range(batch):
    #         wav_id, text_id, wav_len, text_len = \
    #             wav_ids[i], text_ids[i], wav_id_lens[i], text_lens[i]
    #         cur_seq_len = wav_len + text_len
    #         seq = torch.cat([
    #             torch.tensor([self.tokenizer.bos]).to(device),
    #             text_id[:text_len],
    #             torch.tensor([self.tokenizer.sep]).to(device),
    #             wav_id[:wav_len, 0],
    #             torch.tensor([self.tokenizer.eos]).to(device)
    #             ], dim=0)
    #         seq_lens.append(seq.shape[0])
    #         seq = F.pad(
    #             seq,
    #             (0, max_seq_len - cur_seq_len),
    #             mode="constant",
    #             value=self.tokenizer.pad,
    #         )
    #         seqs.append(seq)

    #         full_text_seq = torch.stack([text_id[:text_len]] * num_rvqs, dim=1)
    #         sep = torch.stack([torch.tensor([self.tokenizer.sep]).to(device)] * num_rvqs, dim=1)
    #         bos = torch.stack([torch.tensor([self.tokenizer.bos]).to(device)] * num_rvqs, dim=1)
    #         eos = torch.stack([torch.tensor([self.tokenizer.eos]).to(device)] * num_rvqs, dim=1)
    #         full_seq = torch.cat(
    #             (bos, full_text_seq, sep, wav_id[:wav_len, :], eos), dim=0
    #         )
    #         full_seq = F.pad(
    #             full_seq,
    #             (0, 0, 0, max_seq_len - cur_seq_len),
    #             mode="constant",
    #             value=self.tokenizer.pad,
    #         )  # [t, n_codebook]
    #         full_seqs.append(full_seq)

    #          # 区分是text还是wav
    #         seq_sen_id = torch.tensor([1] * (text_len + 2) + [2] * (wav_len + 1))
    #         seq_sen_id = F.pad(
    #             seq_sen_id,
    #             (0, max_seq_len - cur_seq_len),
    #             mode="constant",
    #             value=self.tokenizer.pad,
    #         )
    #         seq_sen_ids.append(seq_sen_id)

    #         pos_id = torch.tensor(list(range(text_len + 2)) + list(range(wav_len + 1)))
    #         pos_id = F.pad(
    #             pos_id,
    #             (0, max_seq_len - cur_seq_len),
    #             mode="constant",
    #             value=self.tokenizer.pad,
    #         )
    #         pos_ids.append(pos_id)

    #     seqs = torch.stack(seqs, dim=0)
    #     seq_lens = torch.tensor(seq_lens).to(device)
    #     full_seqs = torch.stack(full_seqs, dim=0)
    #     pos_ids = torch.stack(pos_ids, dim=0).to(device)
    #     seq_sen_ids = torch.stack(seq_sen_ids, dim=0).to(device)

    #     return seqs, seq_lens, pos_ids, seq_sen_ids, full_seqs

    @torch.no_grad()
    def generate(self, full_seqs, unmask_len, seq_sen_ids, pos_ids, mask2, tokenizer, temperature=1.0):
        for layer_index in range(1, self.model.hp.num_res):
            logits = self.model.predict(
                full_seqs, unmask_len, seq_sen_ids, pos_ids, layer_index
            )
            logits = logits * temperature
            logits[:, :, -2:] = -1e5
            probs = logits.softmax(dim=2)  # [b, t, d]
            samples = (
                torch.argmax(logits, dim=-1)
                + tokenizer.phone_token_num
                + 1
            )
            samples = torch.where(mask2, full_seqs[:, layer_index, :], samples)
            full_seqs[:, layer_index, :] = samples
        
        return full_seqs