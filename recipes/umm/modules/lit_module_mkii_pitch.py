import torch

from recipes.umm.modules.lit_module import Stage3

"""
@hanoihantrakul 27MAY2024 Implementation Notes:
The `forward` function of these lit_modules:
-Stage3PitchSupervised
-Stage3PitchPerceptual
-Stage3PitchSupervisedPerceptual 
are very similar. I copy pasted code mainly to speed up spiking whether 
this approach works. I noted which parts are copy pasted.
"""


class Stage3PitchSupervised(Stage3):
    """
    @hanoihantrakul 27MAY2024:
    This model is based on on `lit_module.Stage3MSSPitchSupervised`. However this setup trains
    on the full mix, not the MSS-separated audio.
    """

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
        @hanoihantrakul 1/19/2024
        Notes: Luckily by default, the standard Stage3 pipeline uses the correct "audio" key
        to extract full mix. I keep this as a separate method to make intention clear.
        """
        return super().prepare_feature(batch)

    def _shared_step(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        text_ids = input_dict["text_ids"]

        # To speed up spiking, this class only supports this combination.
        assert self.model.config.add_supervised_pitch == True
        assert self.model.config.add_perceptual_pitch == False
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

        loss_dict["bs"] = mel.shape[0]
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
        if self.model.config.get("add_supervised_pitch", False):
            loss_dict["loss"] = (
                loss_dict["loss"]
                + (loss_dict["f0_loss"] + loss_dict["vuv_loss"])
                * self.model.config.w_loss_supervised_pitch
            )
        if output_dict["vq_loss"] is not None:
            loss_dict["loss_vq"] = output_dict["vq_loss"]
            loss_dict["loss"] = (
                loss_dict["loss"] + output_dict["vq_loss"] * self.model.config.w_loss_vq
            )

        ### 28MAY2024 This part and below is copy pasted
        if self.trainer.global_step % 100 == 0:
            code_rate = self.get_code_rate(output_dict["vq_ids"])
            loss_dict["aux/code_rate"] = code_rate
        quant_rate = self.get_quant_rate(
            output_dict["vq_ids"].long(), self.model.config.vq_codebook_size
        )
        loss_dict["aux/quant_rate"] = quant_rate
        if getattr(self.model.vq, "entropy", None) is not None:
            loss_dict["aux/entropy"] = self.model.vq.entropy()
        if self.model.config.get("add_ctc", True):
            loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/w_loss_mel"] = self.model.config.w_loss_mel
        if self.model.config.get("add_ctc", True):
            loss_dict["aux/w_loss_ctc"] = self.model.config.w_loss_ctc
        loss_dict["aux/noise_scale"] = output_dict.get("noise_scale", 0)
        if self.model.config.add_chroma:
            loss_dict["aux/w_loss_chroma"] = self.model.config.w_loss_chroma
        if output_dict["vq_loss"] is not None:
            loss_dict["aux/w_loss_vq"] = self.model.config.w_loss_vq
        loss_dict["flops"] = output_dict["flops"]
        # Add VQ Codebook distance logic
        if "vq_mean_distance" in output_dict.keys():
            for k in ["vq_mean_distance", "vq_min_distance", "vq_max_distance"]:
                loss_dict[k] = output_dict[k]
        return loss_dict


class Stage3PitchPerceptual(Stage3):
    """
    @hanoihantrakul 27MAY2024:
    This model is based on on `lit_module.Stage3MSSPitchPerceptual`. However this setup trains
    on the full mix, not the MSS-separated audio.

    It is very similar to the class above `lit_module_mkii_pitch.Stage3PitchSupervised`
    """

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

    def _shared_step(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        text_ids = input_dict["text_ids"]

        # To speed up spiking, this class only supports this combination.
        assert self.model.config.add_supervised_pitch == False
        assert self.model.config.add_perceptual_pitch == True
        assert self.model.config.add_chroma == True
        assert self.model.config.add_ctc == True
        # Only supports criterion.UMMLossPitchPerceptual()
        loss_dict = self.criterion(
            ctc_logits=output_dict["ctc_out"],
            text_ids=text_ids,
            recon_mel=output_dict["mel_out"],
            mel=mel,
            recon_chroma=output_dict["chroma_out"],
            chroma=input_dict["chroma"],
            recon_pitch_h=output_dict["h_pred"],
            pitch_h=output_dict[
                "h_gt"
            ],  # it was easier to compute this in the output_dict loop (not input_dict)
        )

        loss_dict["bs"] = mel.shape[0]
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
        if self.model.config.add_perceptual_pitch:
            loss_dict["loss"] = (
                loss_dict["loss"]
                + loss_dict["perceptual_pitch_loss"]
                * self.model.config.w_loss_perceptual_pitch
            )

        ### 28MAY2024 This part and below is copy pasted
        if output_dict["vq_loss"] is not None:
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
        if self.model.config.get("add_ctc", True):
            loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/w_loss_mel"] = self.model.config.w_loss_mel
        if self.model.config.get("add_ctc", True):
            loss_dict["aux/w_loss_ctc"] = self.model.config.w_loss_ctc
        loss_dict["aux/noise_scale"] = output_dict.get("noise_scale", 0)
        if self.model.config.add_chroma:
            loss_dict["aux/w_loss_chroma"] = self.model.config.w_loss_chroma
        if output_dict["vq_loss"] is not None:
            loss_dict["aux/w_loss_vq"] = self.model.config.w_loss_vq
        loss_dict["flops"] = output_dict["flops"]
        # Add VQ Codebook distance logic
        if "vq_mean_distance" in output_dict.keys():
            for k in ["vq_mean_distance", "vq_min_distance", "vq_max_distance"]:
                loss_dict[k] = output_dict[k]
        return loss_dict


class Stage3PitchSupervisedPerceptual(Stage3):
    """
    @hanoihantrakul 27MAY2024:
    This model is based on on `lit_module.Stage3MSSPitchSupervisedPerceptual`. However this setup trains
    on the full mix, not the MSS-separated audio.

    It is very similar to the class above `lit_module_mkii_pitch.Stage3PitchSupervised`
    """

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

    def _shared_step(self, batch):
        input_dict = self.prepare_feature(batch)
        output_dict = self.model(input_dict)

        mel = input_dict["mel"]
        text_ids = input_dict["text_ids"]

        # To speed up spiking, this class only supports this combination.
        assert self.model.config.add_supervised_pitch == True
        assert self.model.config.add_perceptual_pitch == True
        assert self.model.config.add_chroma == True
        assert self.model.config.add_ctc == True
        # Only supports criterion.UMMLossPitchSupervisedPerceptual()
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
            recon_pitch_h=output_dict["h_pred"],
            pitch_h=output_dict[
                "h_gt"
            ],  # it was easier to compute this in the output_dict loop (not input_dict)
        )

        loss_dict["bs"] = mel.shape[0]
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
        if self.model.config.get("add_supervised_pitch", False):
            loss_dict["loss"] = (
                loss_dict["loss"]
                + (loss_dict["f0_loss"] + loss_dict["vuv_loss"])
                * self.model.config.w_loss_supervised_pitch
            )
        if self.model.config.add_perceptual_pitch:
            loss_dict["loss"] = (
                loss_dict["loss"]
                + loss_dict["perceptual_pitch_loss"]
                * self.model.config.w_loss_perceptual_pitch
            )
        if output_dict["vq_loss"] is not None:
            loss_dict["loss_vq"] = output_dict["vq_loss"]
            loss_dict["loss"] = (
                loss_dict["loss"] + output_dict["vq_loss"] * self.model.config.w_loss_vq
            )

        ### 28MAY2024 This part and below is copy pasted
        if self.trainer.global_step % 100 == 0:
            code_rate = self.get_code_rate(output_dict["vq_ids"])
            loss_dict["aux/code_rate"] = code_rate
        quant_rate = self.get_quant_rate(
            output_dict["vq_ids"].long(), self.model.config.vq_codebook_size
        )
        loss_dict["aux/quant_rate"] = quant_rate
        if getattr(self.model.vq, "entropy", None) is not None:
            loss_dict["aux/entropy"] = self.model.vq.entropy()
        if self.model.config.get("add_ctc", True):
            loss_dict["aux/num_text_ids"] = text_ids.size(0) * text_ids.size(1)
        loss_dict["aux/num_mel_frames"] = mel.size(0) * mel.size(1)
        loss_dict["aux/mel_mean"] = mel.mean()
        loss_dict["aux/mel_std"] = mel.std()
        loss_dict["aux/w_loss_mel"] = self.model.config.w_loss_mel
        if self.model.config.get("add_ctc", True):
            loss_dict["aux/w_loss_ctc"] = self.model.config.w_loss_ctc
        loss_dict["aux/noise_scale"] = output_dict.get("noise_scale", 0)
        if self.model.config.add_chroma:
            loss_dict["aux/w_loss_chroma"] = self.model.config.w_loss_chroma
        if output_dict["vq_loss"] is not None:
            loss_dict["aux/w_loss_vq"] = self.model.config.w_loss_vq
        loss_dict["flops"] = output_dict["flops"]
        # Add VQ Codebook distance logic
        if "vq_mean_distance" in output_dict.keys():
            for k in ["vq_mean_distance", "vq_min_distance", "vq_max_distance"]:
                loss_dict[k] = output_dict[k]
        return loss_dict
