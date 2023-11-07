
from recipes.bigmusic.datasets.lyrics import *

class ConditionalDatasets():
        # T5 Training
        @staticmethod
        def t5_token_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len, music_types):
            datasets = []
            weights = []
            if 'mixture' in music_types:
                mixture_ds = DefaultDatasets.Basic.mcc60m_lossless_dataset(sample_rate, sample_duration)
                mixture_ds_batched = transform_dataset(
                    # only mcc60 has metadata attached for converting to style_text
                    dataset=mixture_ds,
                    segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len), StyleTextT5Transform()],
                    batch_transforms=[AddConditionsTransform("style_tokens,lyrics_tokens"), ],
                    batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(mixture_ds_batched)
                weights.append(0.7)
            if 'instrumental' in music_types:
                instrumental_ds_batched = transform_dataset(
                    dataset=DefaultDatasets.Basic.mcc40m_lossless_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                    segment_transforms=[StyleTextT5Transform()],
                    batch_transforms=[AddConditionsTransform("style_tokens")],
                    batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                    shuffle_buffer_size=shuffle_buffer_size
                )
                datasets.append(instrumental_ds_batched)
                weights.append(0.3)
            ds_valid = transform_dataset(
                dataset=DefaultDatasets.Basic.mcc_validation_dataset(sample_rate, sample_duration),
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len), StyleTextT5Transform()],
                batch_transforms=[AddConditionsTransform("style_tokens,lyrics_tokens")],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=None
            )
            if len(datasets) == 0:
                raise ValueError('Unable to create dataset with music_types', music_types)
            if len(datasets) == 1:
                return datasets[0], ds_valid
            combined_ds =  MultiIterableDataset(
                datasets, 
                num_samples=10_000_000, seed=2023,
                weights=weights
            )
            return combined_ds, ds_valid

        # Vocal conditioning (deprecate)
        @staticmethod
        def vocal_conditional_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len):
            multitask_ds_batched = DefaultDatasets.Batched.style_audio_dataset(sample_rate, sample_duration, batch_size, shuffle_buffer_size, lyrics_max_seq_len)

            vocal_ds_batched = transform_dataset(
                dataset=DefaultDatasets.Basic.resso_mss_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len), VocalChromaTransform()],
                batch_transforms=[AddConditionsTransform("lyrics_tokens,vocal_audio,vocal_chroma")],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=shuffle_buffer_size
            )
            mss_ds_batched = transform_dataset(
                dataset=DefaultDatasets.Basic.resso_mss_dataset(sample_rate=sample_rate, sample_duration=sample_duration),
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len), VocalChromaTransform()],
                batch_transforms=[AddConditionsTransform("style_audio,lyrics_tokens,vocal_audio,vocal_chroma")],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=shuffle_buffer_size
            )
            combined_ds = MultiIterableDataset(
                [multitask_ds_batched, vocal_ds_batched, mss_ds_batched],
                num_samples=10_000_000, seed=2023,
                weights=[0.7, 0.1, 0.2]
            )
            ds_valid = transform_dataset(
                dataset=DefaultDatasets.Basic.karaoke_vocal_validation_dataset(sample_rate, sample_duration),
                segment_transforms=[LyricsTokenTransform.init_espeak_tokenizer(lyrics_max_seq_len), VocalChromaTransform()],
                batch_transforms=[AddConditionsTransform("lyrics_tokens,vocal_audio,vocal_chroma")],
                batch_fn=default_bucket_batcher_fn(sample_rate, sample_duration, batch_size),
                shuffle_buffer_size=None
            )
            return combined_ds, ds_valid
