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
        one_shot_map = None,
        infer_batch_size=16,
        manual_to_device=False,
        required_modules=None,
        tag_types = None,
        loss_weights = None,
        per_tag=False,
        final_pooling = True,
        valid_per_node_max_num=2000,
        test_per_node_max_num=5000,
    ):
        super().__init__()
        self.model = model
        self._lr=lr
        self._scheduler_patience=scheduler_patience
        self._scheduler_decay_factor=scheduler_decay_factor
        self._sample_rate = sample_rate
        self._sample_len = sample_len
        self.manual_to_device = manual_to_device
        self._infer_batch_size = infer_batch_size
        self._train_num_samples = int(self._sample_rate * self._sample_len)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.tag_types = tag_types
        if loss_weights is None:
            self.loss_weights = [1.0] * len(tag_types)
        else:
            self.loss_weights = loss_weights
        self.one_shot_map = one_shot_map
        self.tagging_params = { tag_type: one_shot_map[tag_type].keys() for tag_type in tag_types }
        self.per_tag = per_tag
        self.final_pooling = final_pooling
        self.valid_per_node_max_num = valid_per_node_max_num
        self.test_per_node_max_num = test_per_node_max_num
        self.required_modules = required_modules
        
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
        optimizer = optim.Adam(
           list(self.model.parameters()), lr=self._lr, weight_decay=0
        )
        # optimizer = optim.Adam([
        #     {"params": self.model.stages[0].parameters(), "lr": self._lr / 10},
        #     {"params": self.model.stages[1].parameters(), "lr": self._lr}
        # ])
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
        inputs = {}

        inputs["audio"] = batch[0]
        inputs["tags"] = batch[1]
        
        # model prediction
        tag_pred = self.model(inputs)[0]

        total_loss = None
        for weight, tag_type in zip(self.loss_weights, self.tag_types):
            loss = self.loss(tag_pred[tag_type], inputs["tags"][tag_type])
            if total_loss is None:
                total_loss = weight * loss
            else:
                total_loss += weight * loss

        # Logging to TensorBoard by default
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, sync_dist=True)
        return total_loss


    def on_fit_start(self):  
        if self.manual_to_device: # put stft funcs to gpu
            self.model.stages[0].model.manually_to_device(self.device)

    def on_validation_start(self):
        if self.manual_to_device: # put stft funcs to gpu
            self.model.stages[0].model.manually_to_device(self.device)

    def on_test_start(self):
        if self.manual_to_device: # put stft funcs to gpu
            self.model.stages[0].model.manually_to_device(self.device)

    def on_predict_start(self):
        if self.manual_to_device: # put stft funcs to gpu
            self.model.stages[0].model.manually_to_device(self.device)

    def test_step(self, batch, batch_idx):
        inputs = {}
        inputs["audio"] = batch[0]
        inputs["tags"] = batch[1]
        uutid = batch[2]
        losses = []

        # model prediction
        tag_pred = self.model(inputs)[0]

        for weight, tag_type in zip(self.loss_weights, self.tag_types):
            loss = self.loss(tag_pred[tag_type], inputs["tags"][tag_type])
            losses.append(loss)

            predictions = torch.mean(tag_pred[tag_type], 0).unsqueeze(0).detach().cpu().numpy()
            target_vectors = inputs["tags"][tag_type][0, :].unsqueeze(0).detach().cpu().numpy()

            self.val_outputs.append(
                {
                    "tag_type": tag_type,
                    "loss": loss,
                    "predictions": predictions,
                    "target_vectors": target_vectors,
                    "uutid": uutid[0, :],
                }
            )
        self.log(f"valid_loss", torch.mean(torch.tensor(losses)), prog_bar=True, sync_dist=True)
        return {}

    def validation_step(self, batch, batch_idx):
        inputs = {}

        inputs["audio"] = batch[0]
        inputs["tags"]  = batch[1]
        uutid = batch[2]

        # model prediction
        tag_pred = self.model(inputs)[0]

        losses = []
        for weight, tag_type in zip(self.loss_weights, self.tag_types): 
            loss = self.loss(tag_pred[tag_type], inputs["tags"][tag_type])
            losses.append(loss)

            # detach predictions
            predictions = [prd.detach().cpu().numpy() for prd in tag_pred[tag_type]]
            target_vectors = [trg.detach().cpu().numpy() for trg in inputs["tags"][tag_type]]

            self.val_outputs.append(
                {
                    "tag_type": tag_type,
                    "loss": loss,
                    "predictions": predictions,
                    "target_vectors": target_vectors,
                    "uutid": uutid,
                }
            )

        self.log(f"valid_loss", torch.mean(torch.tensor(losses)), prog_bar=True, sync_dist=False)
        return {}

    def get_ceil(self, i):
        l = len(str(i))
        ii = np.ceil(i/10**(l-1))
        return int(ii*10**(l-1))

    def get_results(self, is_validation=True):
        
        tagtype_outputs = {}
        for outs in self.val_outputs:
            tagtype_outputs.setdefault(outs['tag_type'], [])
            tagtype_outputs[outs['tag_type']].append(outs)
        
        total_pr_auc = 0
        sample_nums = []
        for tag_type, val_outputs in tagtype_outputs.items():
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

            aucs = self.get_auc_scores(trg, prd, self.tagging_params[tag_type], tag_type, self.per_tag)

            if self.per_tag:
                for k, v in aucs.items():
                    self.log(k, v)
            else:
                self.log(f"roc_auc_{tag_type}", aucs["roc_auc"])
                self.log(f"pr_auc_{tag_type}", aucs["pr_auc"])

            total_pr_auc += aucs["pr_auc"]
        self.log("num_valid", round(np.mean(sample_nums)))
        self.log(f"pr_auc", total_pr_auc / len(tagtype_outputs))
        self.val_outputs = []

    def on_validation_epoch_end(self):
        self.get_results(is_validation=True)

    def on_test_epoch_end(self):
        self.get_results(is_validation=False)

    def get_auc_scores(self, targets, logits, tag_names, tag_type, per_tag=False):
        roc_aucs = {}
        pr_aucs = {}
        for i, name in enumerate(tag_names):
            try:
                roc_auc = metrics.roc_auc_score(targets[:, i], logits[:, i])
                pr_auc = metrics.average_precision_score(targets[:, i], logits[:, i])
                roc_aucs[name] = roc_auc
                pr_aucs[name] = pr_auc
                #print('roc_auc: %.4f' % roc_auc)
                #print('pr_auc: %.4f' % pr_auc)
            except ValueError as e:
                pass
                #print('not available')
        mean_roc_auc = np.mean([v for k, v in roc_aucs.items()])
        mean_pr_auc = np.mean([v for k, v in pr_aucs.items()])
        #print(f'{tag_type} roc_auc: %.4f' % mean_roc_auc)
        #print(f'{tag_type} pr_auc: %.4f' % mean_pr_auc)
        results = {'roc_auc': mean_roc_auc,
                   'pr_auc': mean_pr_auc}

        if per_tag:
            to_add = {f"pr_auc_{k}": v for k, v in pr_aucs.items()}
            results.update(to_add)
        return results
        

    def merge_chunk_probs(self, probs, chunk_hop=375):
        # 25hz, 30 seconds, hop 50%
        n_chunks, n_classes, chunk_len = probs.shape # at the resoluation of 25hz of model output
        seq_len = int(n_chunks * chunk_hop + chunk_hop)
        new_probs = np.zeros((n_classes, seq_len))
        count = np.zeros((n_classes, seq_len))
        for i in range(n_chunks):
            new_probs[:, i * chunk_hop : i * chunk_hop + chunk_len] += probs[i, :, :]
            count[:, i * chunk_hop : i * chunk_hop + chunk_len] += 1
        new_probs = new_probs /count
        return new_probs

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
            output = self.model(inputs)[0]
            for tag_type, tag_pred in output.items():
                for pred in tag_pred:
                    pred = torch.sigmoid(pred)
                    chunk_pred = pred.detach().cpu().numpy()
                    res[tag_type].append(np.expand_dims(chunk_pred, 0))

        res_prob = {}
        res_tag = {}
        tag_map = { tag: { idx: category for category, idx in val.items()} for tag, val in self.one_shot_map.items() }
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
                song_pred = self.merge_chunk_probs(song_pred, int(chunk_hop//self._sample_rate * 25))
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