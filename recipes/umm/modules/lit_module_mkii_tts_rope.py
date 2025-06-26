import torch

from recipes.umm.modules.lit_module import Stage2, Stage3Improved

"""
@hanoihantrakul 5JUL2024 Implementation Notes:
The `lit_module.py` was getting too large and unwieldy.
I separated this out here to make the different tokenizer
modules like the TTS_ROPE variant here easier to follow. 
"""


class Stage2TTSRope(Stage2):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        """
        @hanoihantrakul 5JUL2024
        No additional logic is required compared to the traditional Stage2.
        """
        return super().prepare_feature(batch)

    def load_required_modules(self):
        """
        Load additional perceptual pitch predictor.
        """
        super().load_required_modules()
        if "pitchpdt" in self.hparams.required_modules:
            pitchpdt_config = self.hparams.required_modules["pitchpdt"]
            if pitchpdt_config["ckpt_path"].strip() != "":
                state_dict = pitchpdt_config["init_fn"](
                    pitchpdt_config["ckpt_path"],
                    self.local_rank,
                    pitchpdt_config["cache_dir"],
                )["state_dict"]
                print(f'Loading pitchpdt model from {pitchpdt_config["ckpt_path"]}')
                self.model.pitch_predictor.load_and_eval(state_dict)

    def _shared_step(self, batch):
        """
        @hanoihantrakul 5JUL2024
        I use the same `add_pitch` flag as the original TTS implementation.

        However, TTS Stage2 uses pitch f0_hz loss, ctc loss and mel recon loss.
        In the original Vocal Music training of Stage2, there is a chroma, ctc loss and mel recon loss.
        Here, I need to combine f0_hz loss, ctc loss, mel recon loss and chroma loss. Ideally, the
        `criterion` object should just accept a list of losses to compute, but the codebase from
        SEP2023 did not structure it like this.

        I can't use `criterion.UMMLoss` nor can I use `criterion.CTCPitchMelLoss`. Instead I have
        to use `criterion.UMMLossPitchSupervised` which I originally created for `conf/convumm-pitch`
        model variants.
        """
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        if self.model.config.get("add_ctc", True):
            text_ids = input_dict["text_ids"]

        # To speed up spiking, this class only supports this combination.
        assert self.model.config.add_pitch == True
        assert self.model.config.add_chroma == True
        assert self.model.config.add_ctc == True
        # Only supports criterion.UMMLossPitchSupervised()
        loss_dict = self.criterion(
            ctc_logits=output_dict["ctc_out"],
            text_ids=text_ids,
            recon_mel=output_dict["mel_out"],
            mel=mel,
            recon_chroma=output_dict["chroma_out"],
            chroma=input_dict["chroma"],
            recon_f0=output_dict["f0_out"],
            f0=input_dict["f0"],
            recon_vuv=output_dict["vuv_out"],
            vuv=input_dict["vuv"],
        )

        loss_dict["loss"] = loss_dict["loss_mel"] * self.model.config.w_loss_mel
        if self.model.config.get("add_ctc", True):
            loss_dict["loss"] = (
                loss_dict["loss"] + loss_dict["loss_ctc"] * self.model.config.w_loss_ctc
            )
        if self.model.config.add_chroma:
            loss_dict["loss"] = (
                loss_dict["loss"]
                + loss_dict["loss_chroma"] * self.model.config.w_loss_chroma
            )
        if self.model.config.get("add_pitch", False):
            loss_dict["loss"] = (
                loss_dict["loss"]
                + (loss_dict["f0_loss"] + loss_dict["vuv_loss"])
                * self.model.config.w_loss_pitch
            )
        if self.model.config.get("add_ctc", True):
            loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/w_loss_mel"] = self.model.config.w_loss_mel
        if self.model.config.get("add_ctc", True):
            loss_dict["aux/w_loss_ctc"] = self.model.config.w_loss_ctc
        if self.model.config.add_chroma:
            loss_dict["aux/w_loss_chroma"] = self.model.config.w_loss_chroma
        if self.model.config.get("add_pitch", False):
            loss_dict["aux/w_loss_pitch"] = self.model.config.w_loss_pitch

        loss_dict["flops"] = output_dict["flops"]
        return loss_dict


class Stage3TTSRope(Stage3Improved):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
    ):
        """
        @hanoih 14JUL2024: note that Li Tang uses Stage3Improved for TTS pipeline.
        """
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )
    

    @torch.no_grad()
    def preprocessing(self, x, x_length=None):
        return self.model.preprocessing(x, x_length=x_length)
    
    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def prepare_feature(self, batch):
        if self.model.config.get("cal_attention_mask", False) and "audio_length" in batch:
            input_dict = {}
            audio = batch["audio"].squeeze(dim=1).float()
            audio = self.pad_audio(audio)
            audio_length = batch["audio_length"]
            input_dict.update(audio_length=audio_length)

            if self.model.config.get("add_ctc", True):
                input_dict.update(text_ids=batch["token"])
            feature = self.preprocessing(audio, audio_length)
            input_dict.update(feature)
        else:
            return super().prepare_feature(batch)
        return input_dict

    def load_required_modules(self):
        """
        Load additional perceptual pitch predictor.
        """
        super().load_required_modules()
        if "pitchpdt" in self.hparams.required_modules:
            pitchpdt_config = self.hparams.required_modules["pitchpdt"]
            if pitchpdt_config["ckpt_path"].strip() != "":
                state_dict = pitchpdt_config["init_fn"](
                    pitchpdt_config["ckpt_path"],
                    self.local_rank,
                    pitchpdt_config["cache_dir"],
                )["state_dict"]
                print(f'Loading pitchpdt model from {pitchpdt_config["ckpt_path"]}')
                self.model.pitch_predictor.load_and_eval(state_dict)

    def _shared_step(self, batch):
        """
        @hanoihantrakul 5JUL2024
        I use the same `add_pitch` flag as the original TTS implementation.

        However, TTS Stage2 uses pitch f0_hz loss, ctc loss and mel recon loss.
        In the original Vocal Music training of Stage2, there is a chroma, ctc loss and mel recon loss.
        Here, I need to combine f0_hz loss, ctc loss, mel recon loss and chroma loss. Ideally, the
        `criterion` object should just accept a list of losses to compute, but the codebase from
        SEP2023 did not structure it like this.

        I can't use `criterion.UMMLoss` nor can I use `criterion.CTCPitchMelLoss`. Instead I have
        to use `criterion.UMMLossPitchSupervised` which I originally created for `conf/convumm-pitch`
        model variants.
        """
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        if self.model.config.get("add_ctc", True):
            text_ids = input_dict["text_ids"]

        # To speed up spiking, this class only supports this combination.
        assert self.model.config.add_pitch == True
        assert self.model.config.add_chroma == True
        assert self.model.config.add_ctc == True
        # Only supports criterion.UMMLossPitchSupervised()
        loss_dict = self.criterion(
            ctc_logits=output_dict["ctc_out"],
            text_ids=text_ids,
            recon_mel=output_dict["mel_out"],
            mel=mel,
            recon_chroma=output_dict["chroma_out"],
            chroma=input_dict["chroma"],
            recon_f0=output_dict["f0_out"],
            f0=input_dict["f0"],
            recon_vuv=output_dict["vuv_out"],
            vuv=input_dict["vuv"],
        )

        loss_dict["loss"] = loss_dict["loss_mel"] * self.model.config.w_loss_mel
        if self.model.config.get("add_ctc", True):
            loss_dict["loss"] = (
                loss_dict["loss"] + loss_dict["loss_ctc"] * self.model.config.w_loss_ctc
            )
        if self.model.config.add_chroma:
            loss_dict["loss"] = (
                loss_dict["loss"]
                + loss_dict["loss_chroma"] * self.model.config.w_loss_chroma
            )
        if self.model.config.get("add_pitch", False):
            loss_dict["loss"] = (
                loss_dict["loss"]
                + (loss_dict["f0_loss"] + loss_dict["vuv_loss"])
                * self.model.config.w_loss_pitch
            )
        if self.model.config.get("add_ctc", True):
            loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/w_loss_mel"] = self.model.config.w_loss_mel
        if self.model.config.get("add_ctc", True):
            loss_dict["aux/w_loss_ctc"] = self.model.config.w_loss_ctc
        if self.model.config.add_chroma:
            loss_dict["aux/w_loss_chroma"] = self.model.config.w_loss_chroma
        if self.model.config.get("add_pitch", False):
            loss_dict["aux/w_loss_pitch"] = self.model.config.w_loss_pitch

        # VQ related losses and metrics
        if 'vq_loss' in output_dict.keys() and output_dict["vq_loss"] is not None:
            loss_dict["loss_vq"] = output_dict["vq_loss"]
            loss_dict["loss"] = (
                loss_dict["loss"] + output_dict["vq_loss"] * self.model.config.w_loss_vq
            )
            if self.trainer.global_step % 100 == 0:
                code_rate = self.get_code_rate(output_dict["vq_ids"])
                loss_dict["aux/code_rate"] = code_rate
            quant_rate = self.get_quant_rate(
                output_dict["vq_ids"].long(), self.model.config.vq_codebook_size
            )
            loss_dict["aux/quant_rate"] = quant_rate
            if getattr(self.model.vq, "entropy", None) is not None:
                loss_dict["aux/entropy"] = self.model.vq.entropy()
        # Add VQ Codebook distance logic
        if "vq_mean_distance" in output_dict.keys():
            for k in [
                "vq_mean_distance",
                "vq_min_distance",
                "vq_max_distance",
                "num_zero_mag_codebook_vectors",
            ]:
                loss_dict[k] = output_dict[k]

        loss_dict["flops"] = output_dict["flops"]
        return loss_dict


class Stage3TTSRopeRVQ(Stage3TTSRope):
    def __init__(
        self,
        model_cls,
        criterion_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
    ):
        """
        @hanoih 14JUL2024: note that Li Tang uses Stage3Improved for TTS pipeline.
        """
        super().__init__(
            model_cls=model_cls,
            criterion_cls=criterion_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )


    @torch.no_grad()
    def pad_audio(self, x, x_mask=None):
        return self.model.pad_audio(x, x_mask)

    
    def _shared_step(self, batch):
        """
        @hanoihantrakul 5JUL2024
        I use the same `add_pitch` flag as the original TTS implementation.

        However, TTS Stage2 uses pitch f0_hz loss, ctc loss and mel recon loss.
        In the original Vocal Music training of Stage2, there is a chroma, ctc loss and mel recon loss.
        Here, I need to combine f0_hz loss, ctc loss, mel recon loss and chroma loss. Ideally, the
        `criterion` object should just accept a list of losses to compute, but the codebase from
        SEP2023 did not structure it like this.

        I can't use `criterion.UMMLoss` nor can I use `criterion.CTCPitchMelLoss`. Instead I have
        to use `criterion.UMMLossPitchSupervised` which I originally created for `conf/convumm-pitch`
        model variants.
        """
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        if self.model.config.get("add_ctc", True):
            text_ids = input_dict["text_ids"]
        # To speed up spiking, this class only supports this combination.
        assert self.model.config.add_pitch == True
        assert self.model.config.add_chroma == True
        assert self.model.config.add_ctc == True
        # Only supports criterion.UMMLossPitchSupervised()
        loss_dict = self.criterion(
            ctc_logits=output_dict["ctc_out"],
            text_ids=text_ids,
            recon_mel=output_dict["mel_out"],
            mel=mel,
            recon_chroma=output_dict["chroma_out"],
            chroma=input_dict["chroma"],
            recon_f0=output_dict["f0_out"],
            f0=input_dict["f0"],
            recon_vuv=output_dict["vuv_out"],
            vuv=input_dict["vuv"],
            audio_padding_mask=None, #input_dict["audio_padding_mask"], #(set to None to align with Stage3TTSRope)
        )

        loss_dict["loss"] = loss_dict["loss_mel"] * self.model.config.w_loss_mel
        if self.model.config.get("add_ctc", True):
            loss_dict["loss"] = (
                loss_dict["loss"] + loss_dict["loss_ctc"] * self.model.config.w_loss_ctc
            )
        if self.model.config.add_chroma:
            loss_dict["loss"] = (
                loss_dict["loss"]
                + loss_dict["loss_chroma"] * self.model.config.w_loss_chroma
            )
        if self.model.config.get("add_pitch", False):
            loss_dict["loss"] = (
                loss_dict["loss"]
                + (loss_dict["f0_loss"] + loss_dict["vuv_loss"])
                * self.model.config.w_loss_pitch
            )
        if self.model.config.get("add_ctc", True):
            loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/w_loss_mel"] = self.model.config.w_loss_mel
        if self.model.config.get("add_ctc", True):
            loss_dict["aux/w_loss_ctc"] = self.model.config.w_loss_ctc
        if self.model.config.add_chroma:
            loss_dict["aux/w_loss_chroma"] = self.model.config.w_loss_chroma
        if self.model.config.get("add_pitch", False):
            loss_dict["aux/w_loss_pitch"] = self.model.config.w_loss_pitch
        loss_dict["aux/w_loss_vq"] = self.model.config.w_loss_vq

        # RVQ related losses and metrics
        if "rvq_loss" in output_dict and output_dict["rvq_loss"] is not None:
            loss_dict["loss_rvq"] = output_dict["rvq_loss"].mean()
            loss_dict["loss"] = (
                loss_dict["loss"] + output_dict["rvq_loss"].mean() * self.model.config.w_loss_vq
            )
            if self.trainer.global_step % 100 == 0:
                for r in range(output_dict["rvq_ids"].shape[-1]):
                    code_rate = self.get_code_rate(output_dict["rvq_ids"][..., r])
                    loss_dict[f"aux/code_rate{r}"] = code_rate
        
            for r in range(output_dict["rvq_ids"].shape[-1]):
                quant_rate = self.get_quant_rate(
                    output_dict["rvq_ids"][...,r].long(), self.model.config.vq_codebook_size
                )
                loss_dict[f"aux/quant_rate{r}"] = quant_rate
                if getattr(self.model.rvq.RVQ[r], "entropy", None) is not None:
                    loss_dict[f"aux/entropy{r}"] = self.model.rvq.RVQ[r].entropy()
                if output_dict["ppl"] is not None:
                    loss_dict[f"aux/ppl{r}"] = output_dict["ppl"][..., r]
            # Add VQ Codebook distance logic
            if "vq_mean_distance0" in output_dict.keys():
                for k in [
                    "vq_mean_distance",
                    "vq_min_distance",
                    "vq_max_distance",
                    "num_zero_mag_codebook_vectors",
                ]:
                    for r in range(output_dict["rvq_ids"].shape[-1]):
                        loss_dict[k+str(r)] = output_dict[k+str(r)]

        loss_dict["flops"] = output_dict["flops"]
        return loss_dict
