from operator import concat
import random
from collections import defaultdict
from sklearn import metrics

from einops import rearrange
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed 
from torch import optim
from recipes.mir2.pl_modules.tagging_pl_umm2 import get_auc_scores, merge_chunk_probs
from recipes.umm2.modules.lr_scheduler import get_model_parameters_with_lr
from recipes.umm2.modules.utils import (
    get_nuc,
    get_task_loss_weights,
    get_task_losses,
    get_quant_rate,
)

from sklearn.metrics import accuracy_score
import warnings 
warnings.filterwarnings("ignore", category=UserWarning)   


class LitTagging(pl.LightningModule):
    def __init__(
        self,
        model,
        lr,
        scheduler_patience,
        scheduler_decay_factor,
        sample_rate,
        sample_len,
        infer_batch_size=32,
        required_modules=None,
        tag_types = None, 
        tag_map = None, # not used
        per_tag=False,
        final_pooling = True,
        valid_per_node_max_num=2000,
        test_per_node_max_num=5000,
        vq_codebook_size=1024,
    ):
        super().__init__()
        self.model = model
        self._lr=lr
        self._scheduler_patience=scheduler_patience
        self._scheduler_decay_factor=scheduler_decay_factor
        self._sample_rate = sample_rate
        self._sample_len = sample_len
        self._infer_batch_size = infer_batch_size
        self.tag_types = tag_types
        self.per_tag = per_tag
        self.final_pooling = final_pooling
        self.valid_per_node_max_num = valid_per_node_max_num
        self.test_per_node_max_num = test_per_node_max_num
        self.vq_codebook_size = vq_codebook_size
        self.required_modules = required_modules
        self.cached_log_dict = dict()
        self.val_outputs = []

    def setup(self, stage):
        if self.global_rank == 0:
            print(self.model)
        if stage == "fit" and self.required_modules is not None:
            self.load_required_modules()

    def load_required_modules(self):
        for module_name, loader_config in self.required_modules.items():
            print(f"loading module {module_name}...")
            _args = {k: v for k, v in loader_config.items() if k != "loader"}
            loader = loader_config["loader"](**_args)
            self = loader.load_model(pl_module=self)

    def configure_optimizers(self):
        # Config optimizer and scheduler
        # optimizer = optim.Adam(
        #    list(self.model.parameters()), lr=self._lr, weight_decay=0
        # )
        optimizer = optim.Adam([
            {"params": self.model.stages[0].parameters(), "lr": self._lr / 10},
            {"params": self.model.stages[1].parameters(), "lr": self._lr}
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self._scheduler_patience,
            factor=self._scheduler_decay_factor,
            verbose=True,
            mode="min",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def training_step(self, batch, batch_idx):
        prev_log_dict, pending_deletion = self.flush_log_dict()

        inputs = {}
        inputs["audio"] = batch[0]
        inputs["tags"] = batch[1]
        
        # model prediction
        output_dict = self.model(inputs)

        self.log_dict(prev_log_dict, prog_bar=True, sync_dist=False, rank_zero_only=True)
        del pending_deletion
        del prev_log_dict

        loss_dict = {"loss": output_dict["loss"]}
        #self.log("train_loss", loss_dict['loss'], prog_bar=True, on_step=True, sync_dist=True)

        loss_dict.update(get_task_loss_weights(self, prefix="aux/")) # update by adding f"aux/w_loss_{task}"
        loss_dict.update(get_task_losses(self, output_dict))    # update by adding f"loss_{task}"

        if self.trainer.global_step % 100 == 0:
            code_rate = get_nuc(output_dict["vq_ids"])
            loss_dict["aux/code_rate"] = code_rate
    
        loss_dict["aux/quant_rate"] = get_quant_rate(self,
            output_dict["vq_ids"].long(), self.vq_codebook_size
        )
        loss_dict["aux/vq_entropy"] = output_dict["vq_entropy"]

        mel = output_dict["mel"]
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()

        loss_dict["training/loss"] = loss_dict["loss"]
        self.log_dict_cached(loss_dict)

        return loss_dict

    def log_dict_cached(self, kvs):
        """Save `kvs` into cached log dict. The dict is flushed after each
        forward pass."""
        self.cached_log_dict.update(
            {k: v if v is not torch.Tensor else v.detach() for k, v in kvs.items()}
        )

    def flush_log_dict(self):
        orig_keys = []
        cpu_values = []
        cuda_values = []

        # Move all non-CUDA tensor in one go.
        for k, v in self.cached_log_dict.items():
            if v is not torch.Tensor:
                orig_keys.append(k)
                cpu_values.append(v)
            elif not v.is_cuda:
                orig_keys.append(k)
                cpu_values.append(v.item())
            else:
                assert v.is_cuda

        for k, v in self.cached_log_dict.items():
            if v is torch.Tensor and v.is_cuda:
                orig_keys.append(k)
                if len(v.size()) == 0:
                    v = torch.unsqueeze(v.float(), 0)
                cuda_values.append(v)

        values = torch.cat(
            [torch.tensor(cpu_values, dtype=torch.float, device="cuda")] + cuda_values
        )
        # Support different collectives is just a matter of gathering metrics
        # to rank 0 and reducing them locally.
        torch.distributed.reduce(values, 0, op=torch.distributed.ReduceOp.AVG)

        res_dict = {orig_keys[i]: values[i] for i in range(len(values))}
        self.cached_log_dict.clear()

        return res_dict, [orig_keys, cpu_values, cuda_values, values]
    
    def test_step(self, batch, batch_idx):
        inputs = {}
        inputs["audio"] = batch[0]
        inputs["tags"] = batch[1]
        uutid = batch[2]
        losses = []

        # model prediction
        output = self.model(inputs)

        for tag_type in self.tag_types: 
            predictions = torch.mean(output['tag_pred'][tag_type], 0).unsqueeze(0).detach().cpu().numpy()
            target_vectors = inputs["tags"][tag_type][0, :].unsqueeze(0).detach().cpu().numpy()

            self.val_outputs.append(
                {
                    "tag_type": tag_type,
                    "predictions": predictions,
                    "target_vectors": target_vectors,
                    "uutid": uutid[0, :],
                }
            )
    
        self.log(f"valid_loss", output["loss"], prog_bar=True, sync_dist=True)
        return {}

    def validation_step(self, batch, batch_idx):
        inputs = {}

        inputs["audio"] = batch[0]
        inputs["tags"]  = batch[1]
        uutid = batch[2]

        # model prediction
        output = self.model(inputs)

        for tag_type in self.tag_types: 
            # detach predictions
            predictions = [prd.detach().cpu().numpy() for prd in output['tag_pred'][tag_type]]
            target_vectors = [trg.detach().cpu().numpy() for trg in inputs["tags"][tag_type]]

            self.val_outputs.append(
                {
                    "tag_type": tag_type,
                    "predictions": predictions,
                    "target_vectors": target_vectors,
                    "uutid": uutid,
                }
            )

        self.log(f"valid_loss", output["loss"], prog_bar=True, sync_dist=False)
        return {}

    def get_ceil(self, i):
        l = len(str(i))
        ii = np.ceil(i/10**(l-1))
        return int(ii*10**(l-1))

    def get_results(self, is_validation=True):

        outputs_map = {}
        for outs in self.val_outputs:
            outputs_map.setdefault(outs['tag_type'], [])
            outputs_map[outs['tag_type']].append(outs)
        
        total_pr_auc = 0
        sample_nums = []
        for tag_type, val_outputs in outputs_map.items():
            predictions = [torch.tensor(outs['predictions']) for outs in val_outputs]
            targets = [torch.tensor(outs['target_vectors']) for outs in val_outputs]
            uutids = [torch.tensor(outs["uutid"]) for outs in val_outputs]
        
            prd = torch.cat(predictions)
            trg = torch.cat(targets)
            uutid = torch.cat(uutids)
            
            if is_validation:
                pad_len = self.valid_per_node_max_num - len(uutid) # self.get_ceil(len(uutid))
            else:
                pad_len = self.test_per_node_max_num - len(uutid) # self.get_ceil(len(uutid))
            if pad_len > 0:
                prd = torch.cat((prd, prd[-1, :].repeat(pad_len, 1)))
                trg = torch.cat((trg, trg[-1, :].repeat(pad_len, 1)))
                uutid = torch.cat((uutid, uutid[-1].repeat(pad_len)))
            # else:
            #     prd = prd[: self.valid_per_node_max_num, :]
            #     trg = trg[: self.valid_per_node_max_num, :]
            #     uutid = uutid[: self.valid_per_node_max_num]
            
            prd = self.all_gather(prd)
            trg = self.all_gather(trg)
            uutid = self.all_gather(uutid)
            
            prd = rearrange(prd, 'x y z -> (x y) z').cpu().numpy()
            trg = rearrange(trg, 'x y z -> (x y) z').cpu().numpy()
            uutid = rearrange(uutid, 'x y -> (x y)').cpu().numpy()

            _, idx = np.unique(uutid, return_index=True)
            prd = prd[idx, :]
            trg = trg[idx, :]
            
            #print(f'### len={len(idx)}, full_len={len(uutid)}')
            sample_nums.append(len(idx))

            aucs = get_auc_scores(trg, prd, self.per_tag)

            if self.per_tag:
                for k, v in aucs.items():
                    self.log(k, v)
            else:
                self.log(f"roc_auc_{tag_type}", aucs["roc_auc"])
                self.log(f"pr_auc_{tag_type}", aucs["pr_auc"])

            total_pr_auc += aucs["pr_auc"]
        self.log("num_valid", round(np.mean(sample_nums)))
        self.log(f"pr_auc", total_pr_auc / len(outputs_map))
        self.val_outputs = []

    def on_validation_epoch_end(self):
        self.get_results(is_validation=True)

    def on_test_epoch_end(self):
        self.get_results(is_validation=False)


    def predict_step(self, batch, batch_idx):
        audio = batch[0]
        duration = audio.shape[-1] / self._sample_rate
        chunk_len = int(self._sample_rate * self._sample_len)
        chunk_hop = chunk_len//2
        audio = torch.nn.functional.pad(
            audio, (0, chunk_hop)
        )
        audio = audio.unfold(1, chunk_len, chunk_hop)

        res = {tag_type: [] for tag_type in self.tag_types}
        for b in torch.split(audio.squeeze(0), self._infer_batch_size):
            inputs = {"audio": b}
            output = self.model(inputs)['tag_pred']
            for tag_type, tag_pred in output.items():
                for pred in tag_pred:
                    pred = torch.sigmoid(pred)
                    chunk_pred = pred.detach().cpu().numpy()
                    res[tag_type].append(np.expand_dims(chunk_pred, 0))

        res_prob = {}
        res_tag = {}
        tag_map = { tag: { idx: category for category, idx in val.items()} for tag, val in self.tag_map.items() }
        for tag_type, pred in res.items():
            song_pred = np.concatenate(pred, 0)
            if self.final_pooling:
                tag_prob = np.mean(song_pred, axis=0)
                res_prob[tag_type] = tag_prob
                idx = np.argsort(-tag_prob)
                if tag_type == 'instruments':
                    res_tag[tag_type] = {tag_map[tag_type][i]: round(tag_prob[i], 4) for i in idx if tag_prob[i]>0.099}
                else:
                    res_tag[tag_type] = {tag_map[tag_type][i]: round(tag_prob[i], 4) for i in idx[:3]}
            else:
                song_pred = merge_chunk_probs(song_pred, int(chunk_hop//self._sample_rate * 25))
                song_pred = song_pred[:, : int(np.ceil(duration * 25))] # seq in 25hz
                res_prob[tag_type] = song_pred
                res_tag[tag_type] = tag_map[tag_type]
                #print("######", duration, song_pred.shape)
              
        out_dict = {}
        for tag_type, prob in res_prob.items():
            if self.final_pooling:  # final_pooling result, so no time dimension
                out_dict[tag_type] = res_tag[tag_type]
            else:
                out_dict[tag_type] = prob
                out_dict[f"map_{tag_type}"] = res_tag[tag_type]

        return out_dict