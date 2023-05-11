import torch
from tqdm import tqdm

from ...utils.utils import sample
from .lit_coarse_3ar import CoarseModule


class FineModule(CoarseModule):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules,
        checkpointing=False,
        extra_params=None,
        mulan_token_sep=False,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
            mulan_token_sep=mulan_token_sep,
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        device = batch.device

        num_coarse, num_fine, num_res, quant_token_num = (
            self.extra_params.num_coarse,
            self.extra_params.num_fine,
            self.extra_params.num_res,
            self.extra_params.quant_token_num,
        )
        with self.profiler.profile("[LightningModule]FineModule.prepare_feature"):
            with torch.autocast(device_type="cuda", enabled=False):
                wav_ids = self.get_quant(batch.float())
                b = wav_ids.size(0)
                # get coarse ids
                offset = torch.arange(num_coarse) * quant_token_num
                coarse_ids = (wav_ids[:, :, 0:num_coarse] + offset.to(device)).reshape(
                    (b, -1)
                )
                # get fine ids
                offset = (torch.arange(num_fine) + num_coarse) * quant_token_num
                fine_ids = (
                    wav_ids[:, :, num_coarse : num_coarse + num_fine]
                    + offset.to(device)
                ).reshape((b, -1))
                # get eos id
                eos_id = num_res * 1024
                eos_ids = (
                    torch.zeros([b, 1], dtype=wav_ids.dtype, device=device) + eos_id
                )
                # final input tokens
                input_tokens = torch.cat([coarse_ids, eos_ids, fine_ids], dim=1)

                org_len = input_tokens.size(1)
                if hasattr(self.model.config, "train_len"):
                    train_len = self.model.config.train_len
                else:
                    train_len = org_len
                assert train_len >= org_len
                input_tokens = torch.nn.functional.pad(
                    input_tokens, [0, train_len - org_len]
                )

        with self.profiler.profile("[LightningModule]FineModule.model_forward"):
            logits = self.model(input_ids=input_tokens)["logits"]

        if self.local_rank == 0:
            print("Fine Training, V4...")
        loss_offset = coarse_ids.size(1)

        x = logits[:, loss_offset : org_len - 1, :]
        targets = input_tokens[:, loss_offset + 1 : org_len]
        loss = self.criterion(x, targets)
        accu = (x.argmax(dim=-1) == targets).float().mean() * 100
        self.log_dict(
            {"tr_loss": loss.item(), "accu": accu.item()}, prog_bar=True, sync_dist=True
        )
        return loss

    @torch.no_grad()
    def predict(self, coarse_samples):
        num_coarse, num_fine, num_res = (
            self.extra_params.num_coarse,
            self.extra_params.num_fine,
            self.extra_params.num_res,
        )
        device = coarse_samples.device
        bs = coarse_samples.size(0)
        eos_ids = (
            torch.zeros(size=[bs, 1], dtype=torch.long, device=device) + num_res * 1024
        )  # [b, 1]

        slice_range = [[0, 4000]]
        # beg = 0
        # while True:
        #     end = beg + 10 * frequency * num_fine
        #     if end >= args.duration * frequency * num_fine:
        #         end = args.duration * frequency * num_fine
        #         beg = end - 10 * frequency * num_fine
        #         slice_range.append([beg, end])
        #         break
        #     else:
        #         slice_range.append([beg, end])
        #     beg += stride_in_sec * frequency * num_fine

        prev_end = 0
        fine_samples = None
        for cur_beg, cur_end in slice_range:
            cache_len = prev_end - cur_beg
            prev_end = cur_end
            coarse_beg = int(cur_beg / num_fine * num_coarse)
            coarse_end = coarse_beg + 2000
            coarse_slice = coarse_samples[:, coarse_beg:coarse_end]  # [b, coarse*t]
            if cache_len == 0:
                input_tokens = torch.cat([coarse_slice, eos_ids], dim=1)
            else:
                prefix_fine_samples = fine_samples[:, cur_beg : cur_beg + cache_len]
                input_tokens = torch.cat(
                    [coarse_slice, eos_ids, prefix_fine_samples], dim=1
                )
            past_key_values = None

            pbar = tqdm(range(cur_end - cur_beg - cache_len))
            for i in pbar:
                pbar.set_description(
                    f"Fine [{cur_beg} - {cur_end}] [{coarse_beg} - {coarse_end}]"
                )
                fine_outputs = self.model(
                    input_tokens, past_key_values=past_key_values, use_cache=True
                )
                logits = fine_outputs["logits"]  # [b, t, d]
                layer_idx = i % num_fine + num_coarse
                predict_logits = logits[
                    :, -1, layer_idx * 1024 : (layer_idx + 1) * 1024
                ]  # [b, d]
                samples = sample(predict_logits, temp=0.4, mode="gumbel").unsqueeze(1)
                samples = samples + layer_idx * 1024
                past_key_values = fine_outputs["past_key_values"]
                input_tokens = samples
                if fine_samples is None:
                    fine_samples = samples
                else:
                    fine_samples = torch.cat([fine_samples, samples], dim=1)
        return fine_samples

    def predict2(self, coarse_ids, temp=0.4, sample_len=4000, sample_mode="naive"):
        bs = coarse_ids.size(0)
        device = coarse_ids.device
        num_coarse = self.extra_params.num_coarse
        num_fine = self.extra_params.num_fine
        num_res = self.extra_params.num_res
        eos_ids = (
            torch.zeros(size=[bs, 1], dtype=torch.long, device=device) + num_res * 1024
        )  # [b, 1]

        fine_samples = None
        input_tokens = torch.cat([coarse_ids, eos_ids], dim=1)
        past_key_values = None

        pbar = tqdm(range(sample_len))
        for i in pbar:
            pbar.set_description("Fine")
            fine_outputs = self.model(
                input_tokens, past_key_values=past_key_values, use_cache=True
            )
            logits = fine_outputs["logits"]  # [b, t, d]
            layer_idx = i % num_fine + num_coarse
            predict_logits = logits[
                :, -1:, layer_idx * 1024 : (layer_idx + 1) * 1024
            ]  # [b, d]
            samples = sample(predict_logits, temp=temp, mode=sample_mode)
            samples = samples + layer_idx * 1024
            past_key_values = fine_outputs["past_key_values"]
            input_tokens = samples
            if fine_samples is None:
                fine_samples = samples
            else:
                fine_samples = torch.cat([fine_samples, samples], dim=1)
        return fine_samples
