import cruise
import torch
import numpy as np
import functools
import os

from recipes.unit2speech.models.diffusion.dprtnet import DPRTNet
from recipes.unit2speech.models.diffusion.v_diffusion import ARVSampler, Diffusion
from recipes.unit2speech.models.ema import EMAModel
from recipes.unit2speech.modules.utils import get_config_from_file, remove_ddp_module
from recipes.unit2speech.modules.datamodule import U2SDataModule

from cruise.utilities.hdfs_io import hcopy


class U2SModule(cruise.CruiseModule):

    def __init__(
        self,
        # Training params
        lr=3e-4,
        ema_rate=0.999,
        end2end=False,
        weight_decay=0.0,
        lr_anneal_steps=0,
        lr_warmup_steps=0,
        # Model params
        input_dim=256,
        feature_dim=1024,
        num_blocks=8,
        segment_size=64,
        segment_stride=32,
        context_dim=512,
        dropout=0.,
        intra_seq2seq='lstm',
        inter_seq2seq='lstm',
        predict_xstart=False,
        diffusion_steps=0,
        num_chunks=20,
        chunk_length=125, # 20*125=2500 for training
        sampling_length=2500,
        autoencoder="autoencoder",
        autoencoder_path='',
        autoencoder_config='',
        save_interval=2500,
        model_ckpt_dir='./tmp',
        hdfs_path='.'
    ):
        # create model and diffusion
        super().__init__()
        self.save_hparams()
        

    def setup(self, stage) -> None:
        self.model = DPRTNet(
            input_dim=self.hparams.input_dim,
            feature_dim=self.hparams.feature_dim,
            num_blocks=self.hparams.num_blocks,
            num_chunks=self.hparams.num_chunks,
            segment_size=self.hparams.segment_size,
            segment_stride=self.hparams.segment_stride,
            diffusion_steps=self.hparams.diffusion_steps,
            dropout=self.hparams.dropout,
            intra_seq2seq=self.hparams.intra_seq2seq,
            inter_seq2seq=self.hparams.inter_seq2seq,
        )
        if self.hparams.diffusion_steps <= 0:
            self.diffusion = Diffusion(
                steps=self.hparams.diffusion_steps,
                num_chunks=self.hparams.num_chunks,
                chunk_length=self.hparams.chunk_length,
                training=True,
            )
        else:
            self.diffusion = ARVSampler(
                self.model,
                in_channels=self.hparams.input_dim,
                length=self.hparams.num_chunks*self.hparams.chunk_length,
                num_splits=self.hparams.num_chunks,
            )

        self.rank_zero_print("Diffusion Model Params: {:.4f}M".format(
            sum(p.numel() for p in self.model.parameters()) / 1e6))

        # prepare audio encoder
        if self.hparams.autoencoder_config:
            hp = get_config_from_file(f"{self.hparams.autoencoder_config}").hparams
        else:
            hp = get_config_from_file(f"{self.hparams.autoencoder}/config.yaml").hparams
        if self.hparams.autoencoder == 'autoencoder':
            from recipes.unit2speech.models.autoencoder.autoencoder_kl import AutoencoderKL
            autoencoder = AutoencoderKL(hp, stage='enc')
            ckpt = torch.load(self.hparams.autoencoder_path, map_location='cpu')
            state = remove_ddp_module(ckpt['G'])
            autoencoder.load_state_dict(state)
            autoencoder.eval()
            del autoencoder.decoder
        else:
            raise NotImplementedError
        
        self.autoencoder = autoencoder
        
        print("Audio Encoder ({}) Params: {:.4f}M".format(
            self.autoencoder, sum(p.numel() for p in self.autoencoder.parameters()) / 1e6))

    def on_fit_start(self):
        self.rank_zero_print('=========== My custom fit start function is called! ============')
        self.ema_model = EMAModel(self.model, self.hparams.ema_rate)

    def on_before_zero_grad(self, optimizers):
        self.ema_model.update(self.trainer.model.module.module.model)
        # self.rank_zero_info('====Learning rate====')
        # self.rank_zero_info(self.trainer.optimizers[0].param_groups[0]['lr'])

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # save
        if self.trainer.global_rank == 0:
            if self.trainer.global_step % self.hparams.save_interval == 0:
                ckpt_dir = os.path.join(self.hparams.model_ckpt_dir, f"ckpt{(self.trainer.global_step):06d}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(
                    self.trainer.model.module.module.model.state_dict(), 
                    f"{ckpt_dir}/diffusion_model.pt"
                )
                torch.save(
                    self.ema_model.state_dict(),
                    f"{ckpt_dir}/ema_diffusion_model.pt"
                )
                hcopy(ckpt_dir,self.hparams.hdfs_path)

    def training_step(self, batch, batch_idx):
        wavs, tokens = batch
        xs, x_lens = wavs
        tks, tk_lens = tokens
        self.autoencoder.eval() # @FIXME: no_grad doesn't work :(
        zs, z_lens = self.autoencoder(xs, x_lens)
        # with torch.no_grad():
        #     zs, z_lens = self.autoencoder(xs, x_lens)
        if self.diffusion.compute_loss_count % 10 == 0:
           self.rank_zero_info(f'wav lengths = {x_lens}')
           self.rank_zero_info(f'latent lengths = {z_lens}')
           self.rank_zero_info(f'context lengths = {tk_lens}')
           self.rank_zero_info(
               "Latent Rep.: Shape={} Min={} Max={} Mean={} Std={}".format(
                   zs.shape, zs.min(), zs.max(), zs.mean(), zs.std()))
        prompt = []
        for x, x_len in zip(xs, x_lens):
            if x_len > 3 * 24000:
                rand_i = np.random.choice(x_len.cpu()-3*24000)
                prompt.append(x[rand_i:rand_i+3*24000])
            elif x_len < 3 * 24000:
                prompt.append(torch.cat([x[:x_len], torch.zeros(3*24000-x_len, device=x.device)], -1))
            else:
                prompt.append(x[:x_len])
        prompt = torch.stack(prompt)
        kwargs = {"context": (tks, tk_lens),
                  "prompt": prompt,
                  "z_lens": z_lens}
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            zs.detach(),
            kwargs
        )
        losses, t = compute_losses()

        loss = losses["loss"].mean()

        return {'loss': loss}

if __name__ == '__main__':
    cli = cruise.CruiseCLI(U2SModule,
                    trainer_class=cruise.CruiseTrainer,
                    datamodule_class=U2SDataModule)
    cfg, trainer, model, datamodule = cli.parse_args()
    trainer.fit(model, datamodule)
