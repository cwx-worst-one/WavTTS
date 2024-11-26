import torch
from torch.nn.utils import clip_grad_value_
import pytorch_lightning as pl
from recipes.umm.models.patchgan_disc2D import init_discriminator
from recipes.umm.utils.mel_utils import torch_wav2spec
import torch.nn.functional as F
from recipes.umm.models.voc_modules.pitch_predictor.pitch_utils import (
    compute_min_lengths,
)
from recipes.umm.modules import lit_module_logging_utils as logging_utils
from recipes.umm.models.umm_mel_decoder_helpers import (FrozenUMMTokenizer, compute_l1_loss, compute_ssim_loss, compute_LSGAN_loss)
from torch import nn

class UMMMelDecoderTrainingTask(pl.LightningModule):
    """
    1NOV2024 @hanoihantrakul
    Lightning module for training a a mel decoder based on a pretrained Stage3 UMM model.
    Assumes mono signal at 24khz or 44.1khz.        
    """
    def __init__(
        self,
        config,
        model_cls,
        optimizer_cls,
        optimizer_disc_cls,
        scheduler_cls,
        required_modules=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.automatic_optimization = False
        self.model = model_cls()
        self.requires = {} # a dictionary that Pytorch Lightning will not register as trainable modules
        
        self.n_mel_bins_out = config.n_mel_bins_out 
        self.n_mel_channels_out = config.n_mel_channels_out 
        self.training_sample_rate = config.training_sample_rate 

        self.frozen_tokenizer = None
        self.mel_discriminator = init_discriminator(channels_in=self.n_mel_bins_out) # TODO: add config for mel 160 or mel 128
        self.mel_transform = self._setup_mel_audio_transform(self.n_mel_bins_out, self.training_sample_rate) # TODO: add config for mel 160 or 24khz vs 44khz
        
    def setup(self, stage: str) -> None:
        if self.global_rank == 0:
            print(self.model)
        self.load_required_modules()

    def load_required_modules(self):
        pretrained = self.hparams.required_modules["pretrained"]
        if pretrained["ckpt_path"].strip() != "":
            # Need a wrapper around the Tokenizer, otherwise I run into `ddp_find_unused_params` problems within the `UMMMelDecoderTrainingTask` lit_module
            self.frozen_tokenizer = FrozenUMMTokenizer(pretrained["ckpt_path"], self.local_rank, pretrained["cache_dir"])

    def _setup_mel_audio_transform(self, n_mels, sample_rate):
        return lambda x: torch_wav2spec(x, n_mels, sample_rate)

    def training_step(self, batch, batch_idx):
        #breakpoint()
        #print(batch['audio'].shape)
        #print(torch.max(batch['audio']))

        # Manually get optimizer and scheduler for GAN step
        optim_g, optim_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()

        with torch.no_grad():
            # TODO: FrozenUMMTokenizer should resample 44khz audio to 24khz
            pre_vq_latents = self.frozen_tokenizer.get_wav2pre_vq_latents(batch['audio'], self.training_sample_rate)
            #print("pre_vq_latents.shape", pre_vq_latents.shape)

        # This optimizer needs to be toggled on before the trainable model is called
        self.toggle_optimizer(optim_g)
        mel_pred = self.model.forward(pre_vq_latents)
        
        # Sometimes `mel_pred` is 1 or 2 timesteps longer than `mel_gt` e.g. [7,2315,160] vs [7,2317,160]. This is expected and normal behavior
        mel_gt = self.mel_transform(batch['audio'][:, 0]) # remove channel dimension before spec transform
        trim_len = compute_min_lengths(mel_pred, mel_gt, tolerance=5, axis=1)
        mel_pred, mel_gt = mel_pred[:, :trim_len, :], mel_gt[:, :trim_len, :]
        # print("mel_gt.shape", mel_gt.shape)
        # print("mel_pred.shape", mel_pred.shape)
        
        ###################### 
        ### GENERATOR STEP ###
        ######################
        loss_dict = {}

        ### Traditional Losses are computed in the generator step
        # TODO: abstract these losses to a "traditional mel loss object"
        losses_from_generator = {}
        if self.config.w_loss_l1 > 0:
            losses_from_generator['l1_loss'] = compute_l1_loss(mel_pred, mel_gt) * self.config.w_loss_l1
        if self.config.w_loss_ssim > 0:
            losses_from_generator['ssim_loss'] = compute_ssim_loss(mel_pred, mel_gt) * self.config.w_loss_ssim

        # The Generator wins when the discriminator thinks `mel_pred` is real
        if self.config.w_loss_adv > 0:
            # TODO: abstract these into a "train_gen_step"
            disc_out_dict = self.mel_discriminator(mel_pred) # Ren Yi's PatchGANDisc2D.forward() implementation outputs a dict
            disc_out_levels, _, _  = disc_out_dict["y"], disc_out_dict.get("h"), disc_out_dict.get("start_frames") 
            for i, disc_level_value in enumerate(disc_out_levels):
                # `disc_out_levels` is a list corresponding to the different "levels" of the PatchGan
                losses_from_generator[f"adv_gen_loss_{i}"] = compute_LSGAN_loss(disc_level_value, 1) * self.config.w_loss_adv # generator wins when discriminator thinks these samples are "real"

        # Manual generator step backwards
        optim_g.zero_grad()
        g_loss = sum([x for x in losses_from_generator.values()])
        self.manual_backward(g_loss)
        clip_grad_value_(self.model.parameters(), 1.0)
        optim_g.step()
        sched_g.step(self.global_step // 2)
        self.untoggle_optimizer(optim_g)

        # update loss_dict in generator step
        loss_dict.update({f"train/{k}": v for k, v in losses_from_generator.items()})

        ########################## 
        ### DISCRIMINATOR STEP ###
        ##########################
        if self.config.w_loss_adv > 0:
            # TODO: abstract these into a "discrinator_train_step"
            self.toggle_optimizer(optim_d)
            losses_from_discriminator = {}
            
            # Real Mel Spectrograms
            disc_out_dict_real = self.mel_discriminator(mel_gt)
            disc_out_levels_real, disc_out_start_frames = disc_out_dict_real["y"], disc_out_dict_real.get("start_frames")
            # discriminator wins when discriminator correctly assigns 1 to real mel specrograms
            for i, disc_level_value in enumerate(disc_out_levels_real):
                losses_from_discriminator[f"adv_disc_loss_{i}_real"] = compute_LSGAN_loss(disc_level_value, 1) * self.config.w_loss_adv 
            
            # Fake Mel Spectrograms. Note you must .detach() in order to run second .backwards() on the generated mel spec
            disc_out_dict_fake = self.mel_discriminator(mel_pred.detach(), None, disc_out_start_frames) # disc_out_start_frames=[0] but needs to be passed into forward() function to prevent windowing of mel_spec
            #disc_out_dict_fake = self.mel_discriminator(mel_pred) 
            disc_out_levels_fake = disc_out_dict_fake["y"]
            # discriminator wins when discriminator correctly assigns 0 to fake mel specrograms
            for i, disc_level_value in enumerate(disc_out_levels_fake):
                losses_from_discriminator[f"adv_disc_loss{i}_fake"] = compute_LSGAN_loss(disc_level_value, 0) * self.config.w_loss_adv

            # Manual discriminator step backwards
            optim_d.zero_grad()
            d_loss = sum([x for x in losses_from_discriminator.values()])
            self.manual_backward(d_loss)
            clip_grad_value_(self.mel_discriminator.parameters(), 1.0)
            optim_d.step()
            sched_d.step(self.global_step // 2)
            self.untoggle_optimizer(optim_d)

            # update loss_dict in generator step
            loss_dict.update({f"train/{k}": v for k, v in losses_from_discriminator.items()})

        self.log_dict(loss_dict, prog_bar=True, sync_dist=False, rank_zero_only=True)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            # Get the instance of the wandb_logger
            wandb_logger = logging_utils.get_wandb_logger(self.loggers)  # beware not to use self.logger (singular vs. plural)
            num_samples_to_plot = 8 # TODO: make this configurable

            # Get Recon Mel Spectrograms
            with torch.no_grad():
                pre_vq_latents = self.frozen_tokenizer.get_wav2pre_vq_latents(batch['audio'], self.training_sample_rate)
            mel_pred = self.model.forward(pre_vq_latents)
            recon_mel_wandb_img_list = (logging_utils.get_list_of_mel_spec_plots_to_log(mel_pred, num_samples_to_plot))

            # Get GT Mel Spectrograms
            mel_gt = self.mel_transform(batch['audio'][:, 0]) # remove channel dimension before spec transform
            gt_mel_wandb_img_list = logging_utils.get_list_of_mel_spec_plots_to_log(mel_gt, num_samples_to_plot)
            
            # Log Mel Spectrograms
            wandb_logger.experiment.log({"Mel GT": gt_mel_wandb_img_list})
            wandb_logger.experiment.log({"Mel Recon": recon_mel_wandb_img_list})

    def configure_optimizers(self):
        """
        Need to separately set up optimizers for
        1. Model Training
        2. Discriminator Training
        """
        # the Mel Decoder
        optimizer = self.hparams.optimizer_cls(self.model.parameters()) 
        scheduler = self.hparams.scheduler_cls(optimizer)

        # the discriminator
        optimizer_disc = self.hparams.optimizer_disc_cls(self.mel_discriminator.parameters()) 
        scheduler_disc = self.hparams.scheduler_cls(optimizer_disc)
        return [optimizer, optimizer_disc], [
            {"scheduler": scheduler, "interval": "step"},
            {"scheduler": scheduler_disc, "interval": "step"},
        ]


class UMMMelDecoderTrainingTaskV2(pl.LightningModule):
    """
    Lightning module for training a a mel decoder based on a pretrained Stage3 UMM model.
    Supports Stereo Signals and Complex Mel. 
    Unlike Mono setup, there is extra logic to handle the tokenizer trained in mono as well as how to manage 2 channel downsampling and upsampling
    """
    def __init__(
        self,
        config,
        model_cls,
        optimizer_cls,
        optimizer_disc_cls,
        scheduler_cls,
        required_modules=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.automatic_optimization = False
        self.model = model_cls()
        self.requires = {} # a dictionary that Pytorch Lightning will not register as trainable modules
        
        self.n_mel_bins_out = config.n_mel_bins_out 
        self.n_mel_channels_out = config.n_mel_channels_out 
        self.training_sample_rate = config.training_sample_rate 

        self.frozen_tokenizer = None
        # 11NOV2024: Right now I assume the discriminator takes a concat mel of 160x2 = 320 dimensions in. Later, you should separate this into L and R discriminators
        self.mel_discriminator = init_discriminator(channels_in=self.n_mel_bins_out * self.n_mel_channels_out) # TODO: add config for mel 160 or mel 128
        self.mel_transform = self._setup_mel_audio_transform(self.n_mel_bins_out, self.training_sample_rate) # TODO: add config for mel 160 or 24khz vs 44khz
        
    def setup(self, stage: str) -> None:
        if self.global_rank == 0:
            print(self.model)
        self.load_required_modules()

    def load_required_modules(self):
        pretrained = self.hparams.required_modules["pretrained"]
        if pretrained["ckpt_path"].strip() != "":
            # Need a wrapper around the Tokenizer, otherwise I run into `ddp_find_unused_params` problems within the `UMMMelDecoderTrainingTask` lit_module
            self.frozen_tokenizer = FrozenUMMTokenizer(pretrained["ckpt_path"], self.local_rank, pretrained["cache_dir"])

    def _setup_mel_audio_transform(self, n_mels, sample_rate):
        return lambda x: torch_wav2spec(x, n_mels, sample_rate)

    def training_step(self, batch, batch_idx):
        #print(batch['audio'].shape)
        #print(torch.max(batch['audio']))

        # Manually get optimizer and scheduler for GAN step
        optim_g, optim_d = self.optimizers()
        sched_g, sched_d = self.lr_schedulers()

        with torch.no_grad():
            """
            25NOV2024
            The original tokenizer was trained on mono signals at 24khz. My strategy here was to
            average the left and right channels to get a mono signal. Then tokenize this mono signal.
            During decoding, the mel-deocder will have to construct a Left and Right mel
            spectrogram from this mono token signal. 

            There is another concat strategy. I have left the previous implementation and 
            commented it out in the block below this. 
            I tokenize the left and right audio separately
            and then create a token sequence with both left and right channels. 
            """
            mono_audio = torch.mean(batch['audio'], axis=1, keepdim=True)
            pre_vq_latents = self.frozen_tokenizer.get_wav2pre_vq_latents(mono_audio, self.training_sample_rate)

            # """
            # 1NOV2024 I leave this here as reference. Feel free to delete.
            # This code tokenizes the left and right audio channels respectively. Then, the token sequence
            # has twice the embedding size (32 x 2) which are then decoded into stereo mel spectrograms.
            # """
            # left_audio, right_audio = batch['audio'][:, 0, :], batch['audio'][:, 1, :]
            #left_pre_vq_latents = self.frozen_tokenizer.get_wav2pre_vq_latents(left_audio, self.training_sample_rate)
            #right_pre_vq_latents = self.frozen_tokenizer.get_wav2pre_vq_latents(right_audio, self.training_sample_rate)
            #right_pre_vq_latents = self.frozen_tokenizer.get_wav2pre_vq_latents(right_audio, self.training_sample_rate)
            # print("left_pre_vq_latents.shape", left_pre_vq_latents.shape)
            # print("right_pre_vq_latents.shape", right_pre_vq_latents.shape)
            # TODO: add support for different L and R combination strategies. For now, just concat L=[batch_size, time_steps, 32] and R=[batch_size, time_steps, 32] into  output=[batch_size, time_steps, 64]
        
        # This optimizer needs to be toggled on before the trainable model is called
        self.toggle_optimizer(optim_g)
        mel_pred = self.model.forward(pre_vq_latents)
        #print("mel_pred.shape", mel_pred.shape)
        
        # Sometimes `mel_pred` is 1 or 2 timesteps longer than `mel_gt` e.g. [7,2315,160] vs [7,2317,160]. This is expected and normal behavior
        left_mel_gt = self.mel_transform(batch['audio'][:, 0]) # remove channel dimension before spec transform
        right_mel_gt = self.mel_transform(batch['audio'][:, 1]) # remove channel dimension before spec transform
        trim_len = compute_min_lengths(mel_pred, left_mel_gt, tolerance=5, axis=1)
        mel_pred, left_mel_gt, right_mel_gt = mel_pred[:, :trim_len, :], left_mel_gt[:, :trim_len, :], right_mel_gt[:, :trim_len, :]
        # print("mel_pred.shape", mel_pred.shape)
        # print("left_mel_gt.shape", left_mel_gt.shape)
        # print("right_mel_gt.shape", right_mel_gt.shape)
       
        ###################### 
        ### GENERATOR STEP ###
        ######################
        loss_dict = {}

        ### Traditional Losses are computed in the generator step
        # TODO: abstract these losses to a "traditional mel loss object"
        losses_from_generator = {}
        mel_pred_left, mel_pred_right = torch.tensor_split(mel_pred, 2, dim=2) 
        if self.config.w_loss_l1 > 0:
            losses_from_generator['l1_loss_left'] = compute_l1_loss(mel_pred_left, left_mel_gt) * self.config.w_loss_l1
            losses_from_generator['l1_loss_right'] = compute_l1_loss(mel_pred_right, right_mel_gt) * self.config.w_loss_l1
        if self.config.w_loss_ssim > 0:
            losses_from_generator['ssim_loss_left'] = compute_ssim_loss(mel_pred_left, left_mel_gt) * self.config.w_loss_ssim
            losses_from_generator['ssim_loss_right'] = compute_ssim_loss(mel_pred_right, right_mel_gt) * self.config.w_loss_ssim

        # The Generator wins when the discriminator thinks `mel_pred` is real
        if self.config.w_loss_adv > 0:
            # TODO: abstract these into a "train_gen_step"
            disc_out_dict = self.mel_discriminator(mel_pred) # Ren Yi's PatchGANDisc2D.forward() implementation outputs a dict
            disc_out_levels, _, _  = disc_out_dict["y"], disc_out_dict.get("h"), disc_out_dict.get("start_frames") 
            for i, disc_level_value in enumerate(disc_out_levels):
                # `disc_out_levels` is a list corresponding to the different "levels" of the PatchGan
                losses_from_generator[f"adv_gen_loss_{i}"] = compute_LSGAN_loss(disc_level_value, 1) * self.config.w_loss_adv # generator wins when discriminator thinks these samples are "real"

        # Manual generator step backwards
        optim_g.zero_grad()
        g_loss = sum([x for x in losses_from_generator.values()])
        self.manual_backward(g_loss)
        clip_grad_value_(self.model.parameters(), 1.0)
        optim_g.step()
        sched_g.step(self.global_step // 2)
        self.untoggle_optimizer(optim_g)

        # update loss_dict in generator step
        loss_dict.update({f"train/{k}": v for k, v in losses_from_generator.items()})

        ########################## 
        ### DISCRIMINATOR STEP ###
        ##########################
        if self.config.w_loss_adv > 0:
            # TODO: abstract these into a "discrinator_train_step"
            self.toggle_optimizer(optim_d)
            losses_from_discriminator = {}
            
            # Real Mel Spectrograms
            mel_gt = torch.cat((left_mel_gt, right_mel_gt), dim=2)
            disc_out_dict_real = self.mel_discriminator(mel_gt)
            disc_out_levels_real, disc_out_start_frames = disc_out_dict_real["y"], disc_out_dict_real.get("start_frames")
            # discriminator wins when discriminator correctly assigns 1 to real mel specrograms
            for i, disc_level_value in enumerate(disc_out_levels_real):
                losses_from_discriminator[f"adv_disc_loss_{i}_real"] = compute_LSGAN_loss(disc_level_value, 1) * self.config.w_loss_adv 
            
            # Fake Mel Spectrograms. Note you must .detach() in order to run second .backwards() on the generated mel spec
            disc_out_dict_fake = self.mel_discriminator(mel_pred.detach(), None, disc_out_start_frames) # disc_out_start_frames=[0] but needs to be passed into forward() function to prevent windowing of mel_spec
            #disc_out_dict_fake = self.mel_discriminator(mel_pred) 
            disc_out_levels_fake = disc_out_dict_fake["y"]
            # discriminator wins when discriminator correctly assigns 0 to fake mel specrograms
            for i, disc_level_value in enumerate(disc_out_levels_fake):
                losses_from_discriminator[f"adv_disc_loss{i}_fake"] = compute_LSGAN_loss(disc_level_value, 0) * self.config.w_loss_adv

            # Manual discriminator step backwards
            optim_d.zero_grad()
            d_loss = sum([x for x in losses_from_discriminator.values()])
            self.manual_backward(d_loss)
            clip_grad_value_(self.mel_discriminator.parameters(), 1.0)
            optim_d.step()
            sched_d.step(self.global_step // 2)
            self.untoggle_optimizer(optim_d)

            # update loss_dict in generator step
            loss_dict.update({f"train/{k}": v for k, v in losses_from_discriminator.items()})

        self.log_dict(loss_dict, prog_bar=True, sync_dist=False, rank_zero_only=True)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            # Get the instance of the wandb_logger
            wandb_logger = logging_utils.get_wandb_logger(self.loggers)  # beware not to use self.logger (singular vs. plural)
            num_samples_to_plot = 8 # TODO: make this configurable

            # Get Recon Mel Spectrograms
            with torch.no_grad():
                '''
                25NOV2024 @hanoihantrakul
                By default, I take the average of Left and Right channel and combine into a mono audio.
                I then tokenize this mono audio into tokens. 
                '''
                mono_audio = torch.mean(batch['audio'], axis=1, keepdim=True)
                pre_vq_latents = self.frozen_tokenizer.get_wav2pre_vq_latents(mono_audio, self.training_sample_rate)

                # 25NOV2024 @hanoihantrakul: this code is for tokenizing left and right channels separately. 
                # left_audio, right_audio = batch['audio'][:, 0, :], batch['audio'][:, 1, :]
                # left_pre_vq_latents = self.frozen_tokenizer.get_wav2pre_vq_latents(left_audio, self.training_sample_rate)
                # right_pre_vq_latents = self.frozen_tokenizer.get_wav2pre_vq_latents(right_audio, self.training_sample_rate)
                # pre_vq_latents = torch.cat((left_pre_vq_latents, right_pre_vq_latents), dim=2)

            mel_pred = self.model.forward(pre_vq_latents)
            mel_pred_left, mel_pred_right = torch.tensor_split(mel_pred, 2, dim=2) 

            recon_mel_left_wandb_img_list = logging_utils.get_list_of_spectrogram_plots_to_log(mel_pred_left, num_samples_to_plot, title="Mel Spec Left Recon")
            recon_mel_right_wandb_img_list = logging_utils.get_list_of_spectrogram_plots_to_log(mel_pred_right, num_samples_to_plot, title="Mel Spec Right Recon")

            # Get GT Mel Spectrograms
            left_mel_gt = self.mel_transform(batch['audio'][:, 0]) # remove channel dimension before spec transform
            right_mel_gt = self.mel_transform(batch['audio'][:, 1]) # remove channel dimension before spec transform
            gt_mel_left_wandb_img_list = logging_utils.get_list_of_spectrogram_plots_to_log(left_mel_gt, num_samples_to_plot, title="Mel Spec Left GT")
            gt_mel_right_wandb_img_list = logging_utils.get_list_of_spectrogram_plots_to_log(right_mel_gt, num_samples_to_plot, title="Mel Spec Right GT")
            
            # Log Mel Spectrograms
            wandb_logger.experiment.log({"Mel Recon L": recon_mel_left_wandb_img_list})
            wandb_logger.experiment.log({"Mel Recon R": recon_mel_right_wandb_img_list})
            wandb_logger.experiment.log({"Mel GT L": gt_mel_left_wandb_img_list})
            wandb_logger.experiment.log({"Mel GT R": gt_mel_right_wandb_img_list})
            
    def configure_optimizers(self):
        """
        Need to separately set up optimizers for
        1. Model Training
        2. Discriminator Training
        """
        # the Mel Decoder
        optimizer = self.hparams.optimizer_cls(self.model.parameters()) 
        scheduler = self.hparams.scheduler_cls(optimizer)

        # the discriminator
        optimizer_disc = self.hparams.optimizer_disc_cls(self.mel_discriminator.parameters()) 
        scheduler_disc = self.hparams.scheduler_cls(optimizer_disc)
        return [optimizer, optimizer_disc], [
            {"scheduler": scheduler, "interval": "step"},
            {"scheduler": scheduler_disc, "interval": "step"},
        ]