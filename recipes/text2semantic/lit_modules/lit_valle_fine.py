import torch

from .lit_valle_coarse import ValleCoarse, compute_loss

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
        with self.profiler.profile("[LightningModule]LLMModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                _, _, pos_ids, seq_sen_ids, full_seqs, _ = batch
        
        with self.profiler.profile("[LightningModule]LLMModule.model_forward"):
            logits, target_x, target_mask = self.model(full_seqs.transpose(1,2), seq_sen_ids, pos_ids)
        # calculate loss
        loss = compute_loss(logits, target_x, mask=target_mask)
        self.log("train_loss", loss, logger=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.autocast(device_type="cuda", enabled=False):
            _, _, pos_ids, seq_sen_ids, full_seqs, _ = batch

        logits, target_x, target_mask = self.model(full_seqs.transpose(1,2), seq_sen_ids, pos_ids)
        loss = compute_loss(logits, target_x, mask=target_mask)
        self.log_dict( {"val_loss": loss}, logger=True, sync_dist=True, prog_bar=True)
        return loss
    
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

