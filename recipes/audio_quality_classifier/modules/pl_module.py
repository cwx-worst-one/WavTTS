import torch
from torch import nn 
from collections import OrderedDict
import pytorch_lightning as pl




class AudioQualityClassifierModule(pl.LightningModule):
    def __init__(self, 
        model, 
        pretrained_model_path,
        optimizer_cls, 
    ):
        super().__init__()
        # all parameters in ctor will be saved to self.hparams
        self.save_hyperparameters(ignore=['model'])
    
        if pretrained_model_path is not None:
            model = self.load_pretrained_model(model, pretrained_model_path)
        self.model = model
        # freeze part of the model
        for i, (k, param) in enumerate(self.model.named_parameters()):

            if 'mrds' in k and k.split('.')[3] in ['0', '1', '2', '3']:
                param.requires_grad = False
            if "mpds" in k and k.split('.')[3] in ['0', '1', '2', '3']:
                param.requires_grad = False
    
    def load_pretrained_model(self, model, pretrained_model_path):
        state_dict = torch.load(pretrained_model_path)["state_dict"]
        new_dict = OrderedDict()
        for key in state_dict:
            if 'discriminator' in key:
                new_key = key.replace('discriminator.', '')
                new_dict[new_key] = state_dict[key]

        model.load_state_dict(new_dict, strict=True)

        return model

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        return [optimizer]

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        wavs = batch["audio"].unsqueeze(1)
        # text = batch["text"]
        pairs = batch["pairs"]
        # has_vocal=False
        # with torch.autocast(device_type="cuda", enabled=False):
        #     input_ids, target_ids = self.prepare_feature(wavs.float(), text=text, has_vocal=has_vocal)
        # # exclude_time = time.perf_counter() - t
        # logits = self.model(**input_ids)
        # if isinstance(logits, dict):
        #     logits = logits["logits"]
        # elif isinstance(logits, tuple):
        #     logits = logits[0]


        logits, _ = self.model(wavs)

        output =  []
        for logit in logits:
            output.append(logit.mean(dim=tuple(range(1, logit.ndim))))
        output = torch.stack(output).T.mean(dim=-1)
        losses = []
        for pair in pairs:
            rewards_chosen = output[pair[0]]
            rewards_rejected = output[pair[1]]

            losses.append(-nn.functional.logsigmoid(rewards_chosen - rewards_rejected))
            
        loss = torch.stack(losses).mean()

        self.log_dict(
            {
                'train_loss': loss,
            }, 
            prog_bar=True,
            sync_dist=True
        )

        return loss
 

    def validation_step(self, batch, batch_idx):
        wavs = batch["audio"].unsqueeze(1)
        pairs = batch["pairs"]


        logits, _ = self.model(wavs)

        output =  []
        for logit in logits:
            output.append(logit.mean(dim=tuple(range(1, logit.ndim))))
        output = torch.stack(output).T.mean(dim=-1)
        print(output)

        losses = []
        for pair in pairs:
            rewards_chosen = output[pair[0]]
            rewards_rejected = output[pair[1]]

            losses.append(-nn.functional.logsigmoid(rewards_chosen - rewards_rejected))
            
        loss = torch.stack(losses).mean()

        self.log_dict(
            {
                'val_loss': loss,
            }, 
            prog_bar=True,
            sync_dist=True
        )
