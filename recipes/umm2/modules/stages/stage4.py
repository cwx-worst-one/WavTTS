import torch
from einops import rearrange
import numpy as np

from recipes.umm2.modules.stages.stage2 import Stage2
from recipes.umm2.modules.utils import (
    get_nuc,
    get_task_loss_weights,
    get_task_losses,
    get_quant_rate,
)
from recipes.mir2.pl_modules.tagging_pl_umm2 import get_auc_scores, merge_chunk_probs

def switch_keys(d, old_key, new_key):
    if old_key in d:
        d[new_key] = d[old_key]
        del d[old_key]
    return d

class Stage4(Stage2):
    def __init__(
        self,
        config,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
        tag_types=None,
        eval_padding=False,
    ):
        super().__init__(
            config=config,
            model_cls=model_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )

        self.config = config
        self.tag_types = tag_types
        self.eval_padding = eval_padding

    def _shared_step(self, batch, training=True):

        output_dict = self.model(batch)

        loss_dict = {"loss": output_dict["loss"]}

        mel = output_dict["mel"]
        loss_dict["bs"] = mel.shape[0]
        #loss_dict["flops"] = output_dict["flops"]
        
        if "text_ids" in output_dict:
            text_ids = output_dict["text_ids"]
            loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)

        loss_dict.update(get_task_loss_weights(self, prefix="aux/")) # update by adding f"aux/w_loss_{task}"
        loss_dict.update(get_task_losses(self, output_dict))    # update by adding f"loss_{task}"
        # print("####", "loss_dict", list(loss_dict.keys()))

        if self.trainer.global_step % 100 == 0:
            code_rate = get_nuc(output_dict["vq_ids"])
            loss_dict["aux/code_rate"] = code_rate
    
        loss_dict["aux/quant_rate"] = get_quant_rate(self,
            output_dict["vq_ids"].long(), self.config.vq_codebook_size
        )
        loss_dict["aux/vq_entropy"] = output_dict["vq_entropy"]
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/noise_scale"] = output_dict.get("noise_scale", 0)
        
        if not training:
            loss_dict[f"tags"] = output_dict["tags"]
            loss_dict[f"tag_pred"] = output_dict["tag_pred"]
    
        return loss_dict

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss_dict = self._shared_step(batch, training=False)
        loss_dict = {k: v for k, v in loss_dict.items() if "aux/" not in k} # remove aux items

        if dataloader_idx not in self.val_outputs:
            self.val_outputs[dataloader_idx] = []

        loss_dict = {k: v for k, v in loss_dict.items() if 'loss' in k or 'tag' in k}  # keep only loss related items
        self.val_outputs[dataloader_idx].append(loss_dict)

    def on_validation_epoch_end(self):
        dataloader_idx = 0
        per_node_max_num = 6000
        
        val_loss_dict = {}
        val_tag_dict = {}
        for output in self.val_outputs[dataloader_idx]:
            for k, v in output.items():
                if 'loss' in k:
                    k = f"val_0/{k}"
                    if k not in val_loss_dict:
                        val_loss_dict[k] = [v]
                    else:
                        val_loss_dict[k].append(v)
                elif 'tag' in k and isinstance(v, dict):  # tagging related
                    for tag_type in v:
                        tag_k = f"{k}_{tag_type}"
                        if tag_k not in val_tag_dict:
                            val_tag_dict[tag_k] = [v[tag_type]]
                        else:
                            val_tag_dict[tag_k].append(v[tag_type])

        #from IPython import embed; embed(using=False); os._exit(0)

        for k in val_loss_dict:
            if self.eval_padding:
                pad_len = per_node_max_num - len(val_loss_dict[k])
                to_add = torch.tensor(float('nan')).to(val_loss_dict[k][0].device)
                val_loss_dict[k] += [to_add] * pad_len
            val_loss_dict[k] = torch.stack(val_loss_dict[k])

        for k in val_tag_dict:
            val_tag_dict[k] = torch.cat(val_tag_dict[k])
            if self.eval_padding:
                pad_len = per_node_max_num - len(val_tag_dict[k])
                to_pad = torch.tensor(float('nan')).unsqueeze(-1).repeat(pad_len, val_tag_dict[k].shape[-1]).to(val_tag_dict[k].device)
                val_tag_dict[k] = torch.cat((val_tag_dict[k], to_pad))

        for k in val_loss_dict:
            #val_loss_dict[k] = self.all_gather(val_loss_dict[k])
            val_loss_dict[k] = torch.nanmean(val_loss_dict[k].flatten())
        #print({(k, v) for k, v in val_loss_dict.items()})

        for k in val_tag_dict:
            #val_tag_dict[k] = rearrange(self.all_gather(val_tag_dict[k]), 'x y z -> (x y) z')
            val_tag_dict[k] = val_tag_dict[k][~torch.any(val_tag_dict[k].isnan(), dim=1)]
        #print("tag:", {(k, v.shape) for k, v in val_tag_dict.items()})

        pr_aucs, roc_aucs = [], []
        num_valid_collect = set()
        for tag_type in self.tag_types:
            target = val_tag_dict[f"tags_{tag_type}"].cpu().numpy()
            logits = val_tag_dict[f"tag_pred_{tag_type}"].cpu().numpy()
            aucs = get_auc_scores(target, logits, per_tag=False)
            val_loss_dict[f"pr_auc_{tag_type}"] = aucs["pr_auc"]
            pr_aucs.append(aucs["pr_auc"])
            val_loss_dict[f"roc_auc_{tag_type}"] = aucs["roc_auc"]
            roc_aucs.append(aucs["roc_auc"])
            num_valid_collect.add(int(val_tag_dict[f"tag_pred_{tag_type}"].shape[0]))
        val_loss_dict[f"pr_auc"] = np.mean(pr_aucs)
        val_loss_dict[f"roc_auc"] = np.mean(roc_aucs)
        val_loss_dict = switch_keys(val_loss_dict, "val_0/loss_tagging", "valid_loss")
        val_loss_dict[f"num_valid"] = np.mean(list(num_valid_collect))
        # for i, n in enumerate(list(num_valid_collect)):
        #     val_loss_dict[f"num_valid_{i}"] = int(n)

        self.log_dict(val_loss_dict, prog_bar=True, sync_dist=True)
        self.val_outputs = {}
