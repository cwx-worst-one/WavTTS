import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn.utils import clip_grad_value_, clip_grad_norm_

from recipes.umm.models.voc_modules.pitch_predictor.model import PitchPredictor
from recipes.umm.modules.lit_module import Stage3MSS, Stage3
from recipes.umm.modules.lit_module_mkii_dual import DualUMMUtilsMixin, build_disc
from recipes.umm.requires.model_initializer import init_stage3_dual_voc
from recipes.umm.utils.mel_utils import torch_wav2spec


class MelAEKL(Stage3, DualUMMUtilsMixin):
    def __init__(
            self,
            model_cls,
            optimizer_cls,
            scheduler_cls,
            disc_optimizer_cls,
            disc_scheduler_cls,
            criterion_config,
            required_modules=None,
            checkpointing=False,
            extra_params=None,
            load_required_modules_in_init=False,
            branches=None,
            **kwargs,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=lambda: None,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
        self.criterion_config = self.config = config = criterion_config
        self.mel_discs = nn.ModuleDict(
            {
                "full": build_disc(
                    160, disc_hidden_size_max=config.get("disc_hidden_size_max", 256)
                ),
            }
        )
        self.automatic_optimization = False
        self.branches = branches
        if load_required_modules_in_init:
            self.load_required_modules()

    def load_required_modules(self):
        super().load_required_modules()
        if "pitchpdt" in self.hparams.required_modules:
            pitchpdt = self.hparams.required_modules["pitchpdt"]
            if pitchpdt["ckpt_path"].strip() != "":
                self.pitchpdt = PitchPredictor(c_in=160, hidden_size=256, num_layers=3)
                # TODO (Hanoi): add pretrained vocal pitchpdt
                # state_dict = torch.load(pitchpdt["ckpt_path"], map_location='cpu')['state_dict']['model']
                # print(f'Loading pitchpdt model from {pitchpdt["ckpt_path"]}')
                # self.pitchpdt.load_state_dict(state_dict)
        if "vocoder" in self.hparams.required_modules:
            vocoder = self.hparams.required_modules["vocoder"]
            voc_ckpt = vocoder["ckpt_path"].strip()
            if voc_ckpt != "":
                cache_dir = vocoder["cache_dir"].strip()
                self.vocoder = init_stage3_dual_voc(
                    voc_ckpt, self.local_rank, cache_dir
                )["mel_vocoder"]

    def on_train_batch_start(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.model.parameters())
        optimizer_disc = self.hparams.disc_optimizer_cls(self.mel_discs.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        scheduler_disc = self.hparams.disc_scheduler_cls(optimizer_disc)
        return [optimizer, optimizer_disc], [
            {"scheduler": scheduler, "interval": "step"},
            {"scheduler": scheduler_disc, "interval": "step"},
        ]

    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()

        # prepare input/target data
        batch.update(self.prepare_feature(batch))
        batch.update(self.process_tgt_mel(batch, batch["f0"].shape[1] if "f0" in batch else None))

        #########################
        # generator step
        #########################
        losses_gen = {}
        self.toggle_optimizer(optim_g)
        output_dict = self.training_step_gen(batch, losses_gen)

        # generator backward
        optim_g.zero_grad()
        self.manual_backward(
            sum(
                [
                    x
                    for x in losses_gen.values()
                    if isinstance(x, torch.Tensor)
                       and x.requires_grad
                       and x.grad_fn is not None
                ]
            )
        )
        clip_grad_norm_(self.model.parameters(), 1.0)
        optim_g.step()
        sched_g.step(self.global_step // 2)
        self.untoggle_optimizer(optim_g)
        output_dict = {
            k: (v.detach() if isinstance(v, torch.Tensor) else v)
            for k, v in output_dict.items()
        }

        #########################
        # disciminator step
        #########################
        losses_disc = {}
        loss_dict = {f"tr/{k}": v for k, v in losses_gen.items()}
        if self.model.config.w_loss_adv > 0:
            self.toggle_optimizer(optim_d)
            self.training_step_disc(output_dict, batch, losses_disc)
            # discriminator backward
            optim_d.zero_grad()

            self.manual_backward(
                sum(
                    [
                        x
                        for x in losses_disc.values()
                        if isinstance(x, torch.Tensor)
                           and x.requires_grad
                           and x.grad_fn is not None
                    ]
                )
            )
            clip_grad_norm_(self.mel_discs.parameters(), 1.0)
            optim_d.step()
            sched_d.step(self.global_step // 2)
            self.untoggle_optimizer(optim_d)
            loss_dict.update({f"tr/{k}": v for k, v in losses_disc.items()})
        self.log_dict(loss_dict, prog_bar=True, sync_dist=False, rank_zero_only=True)

    def training_step_gen(self, batch, loss_dict):
        output_dict = self.model(batch)
        loss_dict[f"kl"] = output_dict[f"kl_loss_beta"]
        if 'full' in self.branches:
            T = output_dict[f"mel_out_full"].shape[1]
            self.add_mel_loss(output_dict[f"mel_out_full"], batch['full'][:, :T], loss_dict)

            # pitch loss
            w_f0vuv = self.model.config.get("w_loss_pitch", 0.1)
            if self.model.config.add_pitch:
                self.add_pitch_loss(
                    output_dict["f0_out"].squeeze(-1),
                    batch["f0"][:, :T],
                    output_dict["vuv_out"].squeeze(-1),
                    batch["vuv"][:, :T],
                    loss_dict,
                )
                loss_dict["f0"] = loss_dict["f0"] * w_f0vuv
                loss_dict["vuv"] = loss_dict["vuv"] * w_f0vuv

        if self.model.config.w_loss_adv > 0:
            losses_adv = {}
            for t in self.branches:
                o_ = self.mel_discs[t](output_dict[f"mel_out_{t}"])
                p_, h_p_, start_frames = o_["y"], o_.get("h"), o_.get("start_frames")
                self.add_LSGAN_losses(p_, 1, losses_adv, f"A{t}")
            for k in losses_adv:
                losses_adv[k] = losses_adv[k] * self.model.config.w_loss_adv
            loss_dict.update(losses_adv)

        mel = batch['full']
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["bs"] = mel.size(0)
        return output_dict

    def training_step_disc(self, model_out, target_dict, loss_dict):
        for t in self.branches:
            disc_out_real = self.mel_discs[t](target_dict[t])
            p_real, start_frames = disc_out_real["y"], disc_out_real.get("start_frames")
            mel_p = model_out[f"mel_out_{t}"]
            disc_out_fake = self.mel_discs[t](mel_p, None, start_frames)
            p_fake = disc_out_fake["y"]
            self.add_LSGAN_losses(p_real, 1, loss_dict, f"R{t}")
            self.add_LSGAN_losses(p_fake, 0, loss_dict, f"F{t}")

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            # prepare input/target data
            batch.update(self.prepare_feature(batch))
            batch.update(self.process_tgt_mel(batch, batch["f0"].shape[1] if "f0" in batch else None))
            output_dict = self.model(batch)
            for t in ['full']:
                mel_pred = output_dict[f"mel_out_{t}"]
                mel_tgt = batch[t]

                step_cur = self.global_step // 2
                for i, (v_pred, v_gt) in enumerate(zip(mel_pred, mel_tgt)):
                    if i == 5:
                        break
                    self.logger.experiment.add_audio(
                        f"{t}_recording{i:02d}", batch['audio'][i].cpu().numpy(), step_cur, 24000
                    )
                    wav_g = self.mel_torch2wav_np(v_gt)
                    self.logger.experiment.add_audio(
                        f"{t}_g{i:02d}", wav_g, step_cur, 24000
                    )
                    wav_p = self.mel_torch2wav_np(v_pred)
                    self.logger.experiment.add_audio(
                        f"{t}_p{i:02d}", wav_p, step_cur, 24000
                    )

                    fig = plt.figure(figsize=(16, 8))
                    plt.pcolor(v_pred.cpu().T, vmin=-6, vmax=0.5)
                    self.logger.experiment.add_figure(
                        f"val/{t}_p{i:02d}", fig, global_step=step_cur
                    )
                    fig = plt.figure(figsize=(16, 8))
                    plt.pcolor(v_gt.cpu().T, vmin=-6, vmax=0.5)
                    self.logger.experiment.add_figure(
                        f"val/{t}_g{i:02d}", fig, global_step=step_cur
                    )

    def process_tgt_mel(self, batch, max_T=None):
        target_dict = {}
        if self.config.sample_rate == 44100:
            target_dict['full'] = torch_wav2spec(
                batch[f"audio"][:, 0], fmax=self.config.sample_rate // 2,
                sample_rate=self.config.sample_rate, fft_size=self.config.n_fft,
                win_length=self.config.win_length, hop_size=self.config.hop_length
            )
        else:
            target_dict['full'] = torch_wav2spec(batch[f"audio"][:, 0])
        return target_dict
