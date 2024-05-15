import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_

from recipes.umm.models.voc_modules.pitch_predictor.pitch_utils import (
    compute_min_lengths,
)
from recipes.umm.modules import lit_module_logging_utils as logging_utils
from recipes.umm.modules.lit_module import Stage0
from recipes.umm.modules.lit_module_mkii_dual import (
    DualUMMUtilsMixin,
    build_disc,
    get_vocoder,
)
from recipes.umm.utils.mel_utils import torch_wav2spec


def process_tgt_mel(batch, max_T=None):
    """Apply Mel Spec Transform. Refactored from dualumm.DualUMMUtilsMixin()."""
    # @hanoihantrakul 14MAY2024: This should be moved into model.preprocessing in a later MR.
    return torch_wav2spec(F.pad(batch[f"audio"][:, 0], [0, 1200 * 4]))[:, :max_T]


def is_a_tensor_and_should_be_updated(x):
    """Helper function for extracting trainable parameters in generator and discriminator"""
    return isinstance(x, torch.Tensor) and x.requires_grad and x.grad_fn is not None


class ConvUMMGAN(Stage0, DualUMMUtilsMixin):
    """
    ConvUMMGAN for training on vocal music and instrumental-only music.

    Implementation notes @hanoihantrakul 14MAY2024
    I deliberately did not inherit from DualUMM because
    the differences in managing the adversarial loss are large enough
    that this should be its own class.

    DualUMM trains on MSS data. When adapting this to single codebook,
    the preprocessing code is actually closer to ConformerUMM and
    ConvUMM.

    An area which could be refactored is moving the `DualUMMUtilsMixin`
    to a separate file of functions.
    """

    def __init__(
        self,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        disc_optimizer_cls,
        criterion_config,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
        load_required_modules_in_init=False,
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
        self.criterion_config = (
            self.config
        ) = config = criterion_config  # historical naming. This is really just `config`
        self.mel_disc = build_disc(
            160, disc_hidden_size_max=config.get("disc_hidden_size_max", 256)
        )
        self.automatic_optimization = (
            False  # Need to set this for GAN so optimization can be handled manually.
        )
        if load_required_modules_in_init:
            self.load_required_modules()

    def load_required_modules(self):
        # @hanoihantrakul: pitch predictor to be incorporated later
        # if "pitchpdt" in self.hparams.required_modules:
        #     self.pitchpdt = get_pitchpdt(self.hparams, self.local_rank)
        if "vocoder" in self.hparams.required_modules:
            self.vocoder = get_vocoder(self.hparams, self.local_rank)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        """Prepare input features to model. Namely add chroma spec, mel spec and text tokens."""
        input_dict = {}

        # Add chroma features via `self.preprocessing()`
        audio = batch["audio"].squeeze(dim=1).float()
        audio = self.pad_audio(audio)
        feature = self.preprocessing(
            audio
        )  # Under the hood, this is only adding chroma. Ideally, it should also perform the mel transform.
        input_dict.update(feature)

        # Add mel features via `process_tgt_mel` function
        input_dict["mel"] = process_tgt_mel(
            batch
        )  # TODO: this should be moved into model.preprocessing() so that chroma and mel are processed in the same method.

        # Add text tokens if training on vocal music
        if self.config.get("train_on_vocal_music", False):
            input_dict.update(text_ids=batch["token"].long())  # must be long type
        return input_dict

    def configure_optimizers(self):
        """
        DualUMM training needs to setup separate optimizers for:
        1. Model Training
        2. Discriminator Training
        """

        """
        @hanoihantrakul 1 April 2024
        This whole function can be refactorized and reused via composition in 
        both DualUMMv2 and DualUMMv2InstOnly.
        """
        # Append param groups to this list
        params_group = []
        normal_params = []
        special_params = []
        for name, params in self.model.named_parameters():
            if "vq.embedding.weight" in name:
                print("Key {} use zero WD".format(name))
                special_params.append(params)
            else:
                normal_params.append(params)
        params_group.append({"params": special_params, "weight_decay": 0.0})
        params_group.append({"params": normal_params})

        # Configure model optimizer and discriminator optimizer separately
        optimizer = self.hparams.optimizer_cls(params_group)
        optimizer_disc = self.hparams.disc_optimizer_cls(self.mel_disc.parameters())
        scheduler = self.hparams.scheduler_cls(optimizer)
        scheduler_disc = self.hparams.scheduler_cls(optimizer_disc)
        return [optimizer, optimizer_disc], [
            {"scheduler": scheduler, "interval": "step"},
            {"scheduler": scheduler_disc, "interval": "step"},
        ]

    # def on_train_batch_start(self, batch, batch_idx):
    #     """
    #     Implementation Notes:
    #     1 April 2024 @hanoihantrakul
    #     I learned from @Li Tang that VQ is sensitive to normalization. Therefore
    #     TTS team allows BatchNorm parameters to update for the first 20k steps.
    #     Then afer that, the BatchNorm parameters are kept constant. This is an
    #     optimization trick at the time of writing. This logic is invoked at the
    #     start of each batch training step.
    #     """

    #     def _has_a_vocal_vq_layer():
    #         """Check if model has a VQ layer."""
    #         return hasattr(self.model.vq_proj_in, "__len__")

    #     def _model_uses_batchnorm_as_projection_layer():
    #         """Check if model uses BatchNorm as projection layer into VQ."""
    #         # This is hack we saw in Li Tang's code for checking if it is BN
    #         return len(self.model.vq_proj_in) == 3

    #     def _keep_constant_model_batchnorm_params():
    #         """Keep BatchNorm parameters constant. Do not update the params."""
    #         module = self.model.vq_proj_in[2]
    #         if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.SyncBatchNorm)):
    #             self.model.vq_proj_in[2].eval()
    #             self.model.vq_proj_in_inst[2].eval()
    #             if self.trainer.global_step % 1000 == 0:
    #                 print(module.running_mean)

    #     if self.trainer.global_step >= 20_000:
    #         if not _has_a_vocal_vq_layer():
    #             return  # no Vocal VQ so skip
    #         if not _model_uses_batchnorm_as_projection_layer():
    #             return  # not a BatchNorm layer so skip
    #         else:
    #             # make sure BatchNorm params are not updated
    #             _keep_constant_model_batchnorm_params()
    #     return

    def training_step(self, batch, batch_idx):
        """Remember GAN disc and gen steps have to be handled manually."""
        # manually get optimizer and scheduler for GAN step
        optim_g, optim_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()

        # prepare input/target data
        batch.update(self.prepare_feature(batch))

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
                [x for x in losses_gen.values() if is_a_tensor_and_should_be_updated(x)]
            )
        )
        clip_grad_value_(self.model.parameters(), 1.0)
        optim_g.step()
        sched_g.step(self.global_step // 2)
        self.untoggle_optimizer(optim_g)
        output_dict = {
            k: (v.detach() if isinstance(v, torch.Tensor) else v)
            for k, v in output_dict.items()
        }
        loss_dict = {f"tr/{k}": v for k, v in losses_gen.items()}
        #########################
        # disciminator step
        #########################
        losses_disc = {}
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
                        if is_a_tensor_and_should_be_updated(x)
                    ]
                )
            )
            clip_grad_value_(self.mel_disc.parameters(), 1.0)
            optim_d.step()
            sched_d.step(self.global_step // 2)
            self.untoggle_optimizer(optim_d)
            loss_dict.update({f"tr/{k}": v for k, v in losses_disc.items()})

        # logging auxilliary metrics
        if self.trainer.global_step % 100 == 0:
            code_rate = self.get_code_rate(output_dict[f"vq_ids"])
            loss_dict[f"aux/vq_code_rate"] = code_rate
        loss_dict[f"aux/vq_quant_rate"] = self.get_quant_rate(
            output_dict[f"vq_ids"].long(), self.model.config.vq_codebook_size
        )
        if getattr(self.model.vq, "entropy", None) is not None:
            loss_dict[f"aux/vq_entropy"] = self.model.vq.entropy()
        loss_dict["aux/num_mel_frames"] = batch["mel"].size(0) * batch["mel"].size(1)

        self.log_dict(loss_dict, prog_bar=True, sync_dist=False, rank_zero_only=True)

    def training_step_gen(self, batch, loss_dict):
        """Training Step GENERATOR."""
        output_dict = self.model(batch)

        # VQ loss
        if output_dict.get(f"vq_loss") is not None:
            loss_dict[f"vq_loss"] = output_dict[f"vq_loss"]

        # Add VQ Codebook distance logic
        if "vq_mean_distance" in output_dict.keys():
            for k in ["vq_mean_distance", "vq_min_distance", "vq_max_distance"]:
                loss_dict[k] = output_dict[k]

        # Mel Recon loss
        mel_pred, mel_gt = output_dict[f"mel_out"], batch["mel"]
        # Sometimes `mel_pred` is 1 or 2 timesteps longer than `mel_gt` e.g. [7,2315,160] vs [7,2317,160]
        # This is expected and normal behavior
        trim_len = compute_min_lengths(mel_pred, mel_gt, tolerance=5, axis=1)
        mel_pred, mel_gt = mel_pred[:, :trim_len, :], mel_gt[:, :trim_len, :]

        # Apply L1 and/or SSIM loss on mel spec
        if output_dict.get(f"mel_out") is not None:
            mel_loss_weights = {"l1": 1, "ssim": 1}
            self.add_mel_loss(mel_pred, mel_gt, loss_dict, mel_losses=mel_loss_weights)

        # Chroma loss
        if self.model.config.add_chroma:
            chrome_pred, chrome_gt = output_dict[f"chroma_out"], batch["chroma"]
            ############################
            # 3 April 2024 @hanoihantrakul
            # print(f"chrome_pred.shape {chrome_pred.shape} | chrome_gt.shape {chrome_gt.shape}")
            # Sometimes `chrome_pred` is up to 25 timesteps longer than `chrome_gt` e.g. [6, 2660, 12] vs [6, 2644, 12]
            # The delta is also non constant. It it sometimes 16 and sometimes 20 for downsampling=4
            ############################
            trim_len = compute_min_lengths(chrome_pred, chrome_gt, tolerance=25, axis=1)
            chrome_pred, chrome_gt = (
                chrome_pred[:, :trim_len, :],
                chrome_gt[:, :trim_len, :],
            )
            loss_dict[f"chroma_loss"] = F.mse_loss(chrome_pred, chrome_gt)

        # Adversarial Loss
        if self.model.config.w_loss_adv > 0:
            losses_adv = {}
            o_ = self.mel_disc(output_dict[f"mel_out"])
            p_, _, _ = o_["y"], o_.get("h"), o_.get("start_frames")
            self.add_LSGAN_losses(p_, 1, losses_adv, f"Adv_loss")
            for k in losses_adv:
                losses_adv[k] = losses_adv[k] * self.model.config.w_loss_adv
            loss_dict.update(losses_adv)

        # ASR LAS loss
        if self.config.get("train_on_vocal_music", False):
            text_ids = batch["text_ids"]
            w_las = self.model.config.get("w_loss_las", 0.1)
            loss_dict[f"las_loss"] = F.cross_entropy(
                output_dict[f"text_out"].transpose(1, 2), text_ids
            )
            loss_dict[f"las_loss"] = loss_dict[f"las_loss"] * w_las

        return output_dict

    def training_step_disc(self, output_dict, target_dict, loss_dict):
        """Training Step DISCRIMINATOR."""
        disc_out_real = self.mel_disc(target_dict["mel"])
        p_real, start_frames = disc_out_real["y"], disc_out_real.get("start_frames")
        mel_p = output_dict["mel_out"]
        disc_out_fake = self.mel_disc(mel_p, None, start_frames)
        p_fake = disc_out_fake["y"]
        self.add_LSGAN_losses(p_real, 1, loss_dict, f"Real")
        self.add_LSGAN_losses(p_fake, 0, loss_dict, f"Fake")

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            # prepare input/target data
            batch.update(self.prepare_feature(batch))
            output_dict = self.model(batch)

            # Get the instance of the wandb_logger
            wandb_logger = logging_utils.get_wandb_logger(
                self.loggers
            )  # beware not to use self.logger (singular vs. plural)
            num_samples_to_plot = self.model.config.get(
                "num_spectrogram_val_samples_for_plotting", 8
            )

            # Log Mel Spectrograms
            if "mel" in batch.keys():
                gt_mel_wandb_img_list = logging_utils.get_list_of_mel_spec_plots_to_log(
                    batch["mel"], num_samples_to_plot
                )
                recon_mel_wandb_img_list = (
                    logging_utils.get_list_of_mel_spec_plots_to_log(
                        output_dict["mel_out"], num_samples_to_plot
                    )
                )
                wandb_logger.experiment.log({"Mel GT": gt_mel_wandb_img_list})
                wandb_logger.experiment.log({"Mel Recon": recon_mel_wandb_img_list})

            # Log Chroma
            if "chroma" in batch.keys():
                gt_chroma_wandb_img_list = (
                    logging_utils.get_list_of_chroma_spec_plots_to_log(
                        batch["chroma"], num_samples_to_plot
                    )
                )
                recon_chroma_wandb_img_list = (
                    logging_utils.get_list_of_chroma_spec_plots_to_log(
                        output_dict["chroma_out"], num_samples_to_plot
                    )
                )
                wandb_logger.experiment.log({"Chroma GT": gt_chroma_wandb_img_list})
                wandb_logger.experiment.log(
                    {"Chroma Recon": recon_chroma_wandb_img_list}
                )
