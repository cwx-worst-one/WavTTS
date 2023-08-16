from typing import BinaryIO, Iterable, List, Optional, Tuple, Union

import ctranslate2
import faster_whisper
import numpy as np
import torch
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import (
    TranscriptionInfo,
    TranscriptionOptions,
    get_suppressed_tokens,
)
from faster_whisper.vad import VadOptions
from tqdm import tqdm

from recipes.datasets.librilight import LibriLightWebDataModule
from recipes.datasets.libritts import LibriTTSWebDataModule


class WhisperModel(faster_whisper.WhisperModel):
    """
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    """

    def generate_segments_batched(
        self,
        features: np.ndarray,
        tokenizer: faster_whisper.tokenizer.Tokenizer,
        options: faster_whisper.transcribe.TranscriptionOptions,
        encoder_output=None,
    ):
        if features.ndim != 3:
            raise Exception("Make sure to use batched examples")

        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )

        encoder_output = self.encode(features)

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )

        result = self.model.generate(
            encoder_output,
            [prompt] * batch_size,
            length_penalty=options.length_penalty,
            max_length=self.max_length,
            suppress_blank=options.suppress_blank,
            suppress_tokens=options.suppress_tokens,
        )

        tokens_batch = [x.sequences_ids[0] for x in result]

        def decode_batch(tokens: List[List[int]]) -> str:
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            # text_tokens = [token for token in tokens if token < self.eot]
            return tokenizer.tokenizer.decode_batch(res)

        text = decode_batch(tokens_batch)

        return text

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)

        features = faster_whisper.transcribe.get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        language: str = "en",
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        temperature: Union[float, List[float], Tuple[float, ...]] = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[Union[str, Iterable[int]]] = None,
        prefix: Optional[str] = None,
        suppress_blank: bool = True,
        suppress_tokens: Optional[List[int]] = [-1],
        without_timestamps: bool = False,
        max_initial_timestamp: float = 1.0,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        vad_filter: bool = False,
        vad_parameters: Optional[Union[dict, VadOptions]] = None,
    ):

        sampling_rate = self.feature_extractor.sampling_rate
        duration = audio.shape[-1] / sampling_rate
        speech_chunks = None

        features = []
        for a in audio:
            features.append(self.feature_extractor(a.squeeze()))
        features2 = self.feature_extractor(audio[0])
        breakpoint()

        features = np.stack(features)
        language_probability = 1
        encoder_output = None
        all_language_probs = None
        tokenizer = Tokenizer(
            self.hf_tokenizer, self.model.is_multilingual, task=task, language=language
        )

        options = TranscriptionOptions(
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            condition_on_previous_text=condition_on_previous_text,
            temperatures=(
                temperature if isinstance(temperature, (list, tuple)) else [temperature]
            ),
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=get_suppressed_tokens(tokenizer, suppress_tokens),
            without_timestamps=without_timestamps,
            max_initial_timestamp=max_initial_timestamp,
            word_timestamps=word_timestamps,
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
        )
        segments = self.generate_segments(features, tokenizer, options, encoder_output)
        info = TranscriptionInfo(
            language=language,
            language_probability=language_probability,
            duration=duration,
            transcription_options=options,
            vad_options=vad_parameters,
            all_language_probs=all_language_probs,
        )
        return segments, info


def test_whisper():
    batch_size = 8
    # pl_datamodule = LibriLightWebDataModule(
    #     sample_rate=16000,
    #     split="large2",
    #     batch_size=batch_size,
    #     shuffle_buffer_size=100,
    # )
    model_size = "large-v2"
    # model_size = "small"
    # model = WhisperModel(model_size, device="cuda", compute_type="float16", num_workers=8)
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    pl_datamodule = LibriTTSWebDataModule(
        sample_rate=16000, batch_size=8, shuffle_buffer_size=100
    )
    train_loader = pl_datamodule.train_dataloader()

    batch = next(iter(train_loader))

    for _ in tqdm(range(50)):
        segments = model.transcribe(batch["audio"].squeeze().numpy())
        # for audio in batch["audio"]:
        #     segments = model.transcribe(audio.squeeze().numpy())
        #     text = ""
        #     for s in segments[0]:
        #         text += s.text

        #     torchaudio.save(f"{text[:200].strip()}.mp3", audio, pl_datamodule.sample_rate)
