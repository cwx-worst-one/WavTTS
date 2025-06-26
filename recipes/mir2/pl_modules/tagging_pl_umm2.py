import os
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
import json
warnings.filterwarnings("ignore", category=UserWarning)   

def get_auc_scores(targets, logits, per_tag=False):
    roc_aucs = {}
    pr_aucs = {}
    for i in range(targets.shape[-1]):
        try:
            # If the class is not present in the labels, roc_auc_score calculation will raise a ValueError
            roc_auc = metrics.roc_auc_score(targets[:, i], logits[:, i])
            pr_auc = metrics.average_precision_score(targets[:, i], logits[:, i])
            roc_aucs[i] = roc_auc
            pr_aucs[i] = pr_auc
        except ValueError as e:
            # TODO: log the ones without any positive samples
            pass
    mean_roc_auc = np.mean([v for _, v in roc_aucs.items()])
    mean_pr_auc = np.mean([v for _, v in pr_aucs.items()])
    results = {'roc_auc': mean_roc_auc,
                'pr_auc': mean_pr_auc}

    if per_tag:
        to_add = {f"pr_auc_{k}": v for k, v in pr_aucs.items()}
        results.update(to_add)
    return results

def merge_chunk_probs(probs, chunk_hop=375):
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

# TODO: move these into a callback
from recipes.mir2.parquet_dataset.new_label_process import get_id_to_tag_maps

# TODO: move these into a callback
import matplotlib.pyplot as plt

def plot_histogram(logit, target, fn, optimal_threshold=None):
        
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    
    # keep raw logits here because the raw logits are much more informative and easier to interpret
    # logit = sigmoid(logit)
    
    # Combine logits and targets into a single structured array
    data = np.column_stack((logit, target))

    # Group logits into bins
    bins = np.linspace(np.min(logit), np.max(logit), 30)  # bins = np.linspace(-3, 3, 15)
    bin_indices = np.digitize(logit, bins)

    # Count occurrences in each bin for targets 0 and 1
    counts_0 = np.zeros(len(bins))
    counts_1 = np.zeros(len(bins))

    for i, bin_idx in enumerate(bin_indices):
        if target[i] == 0:
            counts_0[bin_idx - 1] += 1
        elif target[i] == 1:
            counts_1[bin_idx - 1] += 1

    # Plot the bar chart
    width = (bins[1] - bins[0]) * 0.8

    
    fig, ax1 = plt.subplots()

    # Plot the bar chart with stacked bars
    # ax1.bar(bins, counts_0, width=width, color='lightgray', label='0 (False)', alpha=0.6, align='center')
    bottom_colors = ['lightgray' if count == 0 else 'gray' for count in counts_1]
    for bin_x, count, color in zip(bins, counts_0, bottom_colors):
        ax1.bar(bin_x, count, width=width, color=color, label=None, alpha=0.6, align='center')
    ax1.bar(bins, counts_1, width=width, bottom=counts_0, color='blue', label='1 (True)', alpha=0.6, align='center')

    # Add labels and legend
    ax1.set_xlabel("Logits")
    ax1.set_ylabel("Count")
    total_true = np.sum(target)
    true_ratio = total_true / len(target)
    # total_false = len(target) - total_true
    ax1.set_title(f"{fn} (T:{int(total_true)}/{len(target)} {true_ratio * 100:.0f}%)")
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.set_xticks(bins)
    ax1.set_xticklabels([f"{b:.3f}" for b in bins], rotation=45)
    
    # Create a second y-axis for probability
    ax2 = ax1.twinx()
    thresholds = np.linspace(np.min(logit), np.max(logit), 50)


    # Initialize lists to store precision, recall, and F1-score
    precision_list = []
    recall_list = []
    f1_list = []
    
    # Compute precision, recall, and F1-score for each threshold
    for threshold in thresholds:
        predictions = (logit >= threshold).astype(int)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(target, predictions, average='binary', zero_division=0)
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    # Plot precision, recall, and F1-score
    ax2.plot(thresholds, precision_list, marker='.', linestyle='-', color='lightcoral', label='Precision', alpha=0.8)
    ax2.plot(thresholds, recall_list, marker='.', linestyle='-', color='lightgrey', label='Recall', alpha=0.8)
    # Add legend for the second axis
    ax2.plot(thresholds, f1_list, marker='.', linestyle='-', color='green', label='F1', alpha=0.8)
    
    if optimal_threshold:
        ax2.axvline(x=optimal_threshold, color='blue', linestyle='--', label=f'Threshold = {threshold:.2f}')

    ax2.set_ylabel("precision, recall, F1-score")
    ax2.set_ylim(0, 1)  # Ensure probability is within [0,1]

    # Add legend for the second axis
    ax2.legend(loc="upper right")

    # Show plot
    plt.tight_layout()
    os.makedirs("tmp/viz", exist_ok=True)
    plt.savefig(f"tmp/viz/{fn}.N={total_true}.png")
    plt.clf()
    plt.close()


def get_optimal_threshold(logit, target):
    # edge case: very little or no support -> assign only the very top threshold
    if np.sum(target) / len(target) < 0.001:
        return np.linspace(np.min(logit), np.max(logit), len(target))[-int(1 + np.sum(target))]

    thresholds = np.linspace(np.min(logit), np.max(logit), 50)
    precision_list = []
    recall_list = []
    f1_list = []
    for threshold in thresholds:
        predictions = (logit >= threshold).astype(int)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(target, predictions, average='binary', zero_division=0)        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    # Find the index of the maximum F1-score
    max_f1_index = np.argmax(f1_list)
    # Get the corresponding threshold
    optimal_threshold = thresholds[max_f1_index]
    return optimal_threshold



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
        tag_map = None,  # not used
        per_tag=False,
        final_pooling = True,
        valid_per_node_max_num=2000,
        test_per_node_max_num=5000,
        optimal_thresholds_json_filepath="recipes/mir2/inference/optimal_thresholds_7905.json",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'required_modules'])
        self.model = model
        self._lr=lr
        self._scheduler_patience=scheduler_patience
        self._scheduler_decay_factor=scheduler_decay_factor
        self._sample_rate = sample_rate
        self._sample_len = sample_len
        self._infer_batch_size = infer_batch_size
        self.tag_types = tag_types
        self.tag_map = tag_map
        #self.tagging_params = { tag_type: tag_map[tag_type].keys() for tag_type in tag_types }
        self.per_tag = per_tag
        self.final_pooling = final_pooling
        self.valid_per_node_max_num = valid_per_node_max_num
        self.test_per_node_max_num = test_per_node_max_num
        self.required_modules = required_modules
        
        self.val_outputs = []

        self.id_to_tag_maps = get_id_to_tag_maps()

        with open(optimal_thresholds_json_filepath, 'r') as f:
            self.optimal_thresholds = json.load(f)

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
        inputs = {}

        inputs["audio"] = batch[0]
        inputs["tags"] = batch[1]
        
        # model prediction
        output = self.model(inputs)
        loss = output['loss']

        # Logging to TensorBoard by default
        self.log("train_loss", output['loss'], prog_bar=True, on_step=True, sync_dist=True)

        return loss


    def test_step(self, batch, batch_idx):
        inputs = {}
        inputs["audio"] = batch[0]
        inputs["tags"] = batch[1]
        uuids_int = batch[2]
        uuids = batch[3]
        losses = []

        # model prediction
        output = self.model(inputs)
        out_dict = {
            "uuid": uuids[0],
        }

        for tag_type in self.tag_types: 
            predictions = torch.mean(output['tag_pred'][tag_type], 0).unsqueeze(0).detach().cpu().numpy()
            target_vectors = inputs["tags"][tag_type][0, :].unsqueeze(0).detach().cpu().numpy()  # indexed 0, because 
            
            # Each tag type for the same audio, is put under different entry in self.val_outputs
            self.val_outputs.append(
                {
                    "tag_type": tag_type,
                    "predictions": predictions,
                    "target_vectors": target_vectors,
                    "uutid": uuids_int[0],
                    "uuid": uuids[0],
                }
            )
            
            # write raw vectors value to 
            out_dict[tag_type] = {
                "predictions": predictions.squeeze(0),
                "target_vectors": target_vectors.squeeze(0),
            }
        
        self.log(f"valid_loss", output["loss"], prog_bar=True, sync_dist=True)
        return out_dict

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
        # if self.global_rank == 0 and self.local_rank == 0: breakpoint()

        outputs_map = {}
        optimal_thresholds = {}
        
        # Aggregate outputs from all samples into a map, keyed by tag_type (e.g. "GENRE", "MOOD", "THEME", ...)
        for outs in self.val_outputs:
            outputs_map.setdefault(outs['tag_type'], [])
            outputs_map[outs['tag_type']].append(outs)
        
        id_to_tag_maps = self.id_to_tag_maps
        
        total_pr_auc = 0
        sample_nums = []
        for tag_type, val_outputs in outputs_map.items():
            # if self.global_rank == 0 and self.local_rank == 0: breakpoint()
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
            
            prd = self.all_gather(prd)  # (N, B, C), where N is the number of GPU processes
            trg = self.all_gather(trg)  # (N, B, C)
            uutid = self.all_gather(uutid)  # (N, B)
            
            prd = rearrange(prd, 'x y z -> (x y) z').cpu().numpy()  # (B', C), where B' is the total samples from all GPUs
            trg = rearrange(trg, 'x y z -> (x y) z').cpu().numpy()
            uutid = rearrange(uutid, 'x y -> (x y)').cpu().numpy()

            _, idx = np.unique(uutid, return_index=True)
            prd = prd[idx, :]
            trg = trg[idx, :]
            
            sample_nums.append(len(idx))
            aucs = get_auc_scores(trg, prd, self.per_tag)
            
            # TODO: move this into a callback at test epoch end
            if self.global_rank == 0 and self.local_rank == 0 and not is_validation:
                per_tag_aucs = get_auc_scores(trg, prd, True)
                optimal_thresholds[tag_type] = []

                # log the per-tag AUCs
                for k, v in per_tag_aucs.items():
                    self.log(f"{tag_type}_{k}", v)
                for i in range(trg.shape[-1]):
                    avg_precision = per_tag_aucs.get(f'pr_auc_{i}')
                    avg_precision_str = "NA" if avg_precision is None else f"{avg_precision:.2f}"

                    

                    # TODO: calculate the optimal thresholds for each tag
                    optimal_threshold = get_optimal_threshold(prd[:, i], trg[:, i])
                    optimal_thresholds[tag_type].append({
                        "tag_id": i,
                        "tag_name": id_to_tag_maps[tag_type][i + 1],
                        "avg_precision": avg_precision,
                        "threshold": optimal_threshold,
                        "threshold_strategy": "peak_f1",
                    })
                    print(f"{tag_type}_{i} mAP: {avg_precision_str}, optimal threshold: {optimal_threshold}")

                    # TODO: add the threshold into the plot
                    tag_name = id_to_tag_maps[tag_type][i + 1].replace(" ", "_").replace("/", "_").replace("\n", "_")
                    plot_histogram(prd[:, i], trg[:, i], f"{tag_type}_{i}.{tag_name}.mAP={avg_precision_str}", optimal_threshold)

                    
                    
            
            if self.per_tag:
                for k, v in aucs.items():
                    self.log(f"{tag_type}_{k}", v)
            else:
                self.log(f"roc_auc_{tag_type}", aucs["roc_auc"])
                self.log(f"pr_auc_{tag_type}", aucs["pr_auc"])

            total_pr_auc += aucs["pr_auc"]
        self.log("num_valid", round(np.mean(sample_nums)))
        self.log(f"pr_auc", total_pr_auc / len(outputs_map))
        self.val_outputs = []

        # TODO: move this into a callback at test epoch end
        if self.global_rank == 0 and self.local_rank == 0 and not is_validation:
            # self.optimal_thresholds = optimal_thresholds
            with open("tmp/optimal_thresholds.json", "w") as f:
                json.dump(optimal_thresholds, f, indent=4)

    def on_validation_epoch_end(self):
        self.get_results(is_validation=True)

    def on_test_epoch_end(self):
        self.get_results(is_validation=False)

    def predict_step(self, batch, batch_idx):
        assert batch[0].shape[0] == 1, "Only support batch size 1 for predict_step"

        sample_len = self._sample_len
        sample_rate = self._sample_rate
        infer_batch_size = self._infer_batch_size
        chunk_len = int(sample_len * sample_rate)
        
        audio = batch[0]
        # filename = batch[1][0]

        # remove the batch dim
        audio = audio.squeeze(0)
        # quantize it to chunk_len for the model
        audio = torch.nn.functional.pad(audio, (0, chunk_len//2))
        # make overlapping clips into a batch of chunk_len
        audio = audio.unfold(0, chunk_len, chunk_len//2)
        if audio.shape[0] > infer_batch_size:
            audio = audio[:infer_batch_size, :]
        output = self.model({"audio": audio})["tag_pred"]
        prediction = {}
        tag_type = list(self.tag_types.keys())[0]
        for tag_type in self.tag_types:
            prediction[tag_type] = torch.mean(output[tag_type], 0).detach().cpu().numpy()
        

        # get tag id to tag name mapping
        id_to_tag_maps = self.id_to_tag_maps
        
        # threshold
        all_tags = {}
        for tag_type in self.optimal_thresholds:
            positive_tags = []
            for tag_threshold in self.optimal_thresholds[tag_type]:
                tag_id = tag_threshold["tag_id"]
                threshold = tag_threshold["threshold"]

                # tag_name = tag_threshold["tag_name"]
                # NOTE: tag_id starts from 0, but index in Vocab starts from 1
                tag_name = id_to_tag_maps[tag_type][tag_id + 1]
                
                # Average precision could be used to filter bad performing tags
                # avg_precision = tag_threshold["avg_precision"]
                
                if prediction[tag_type][tag_id] > threshold:
                    positive_tags.append(tag_name)
            all_tags[tag_type] = positive_tags
        
        out_dict = {
            "finegrained_tags": all_tags,
            "logits": prediction,
            "thresholds": self.optimal_thresholds,
        }


        return out_dict
