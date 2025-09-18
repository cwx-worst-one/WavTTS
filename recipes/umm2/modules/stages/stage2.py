import torch
import math
from recipes.umm2.modules.pl_module import Stage0
from recipes.umm2.modules.utils import (
    get_quant_rates_robust,
    get_perplexity,
    get_task_losses
)

from typing import List, Tuple
from recipes.umm2.scripts.test_stage4_wav2tokentag import prob_to_tag, audio_to_torch
from mariana.utils.audio.audio_logger import AudioLogger
logger = AudioLogger()

class Stage2(Stage0):
    def __init__(
        self,
        config,
        model_cls,
        optimizer_cls,
        scheduler_cls,
        required_modules=None,
        checkpointing=False,
        extra_params=None,
    ):
        super().__init__(
            config=config,
            model_cls=model_cls,
            optimizer_cls=optimizer_cls,
            scheduler_cls=scheduler_cls,
            required_modules=required_modules,
            checkpointing=checkpointing,
            extra_params=extra_params,
        )

        self.config = config

    def _shared_step(self, batch):
        output_dict = self.model(batch)

        #  Calculate VQ statistic if vq layer is in model
        if self.trainer.global_step % 100 == 0:
            if "acoustic_vq_ids" in output_dict and "semantic_vq_ids" in output_dict:
                vq_stats = self._calculate_dual_vq_stats(output_dict)
            else:
                vq_stats = self._calculate_vq_stats(output_dict)
            output_dict.update(vq_stats)

        output_dict['aux/bs'] = len(batch['uttid'])
        return output_dict


    def _calculate_vq_stats(self, output_dict: dict) -> dict:
        """
        Calculates quantization statistics based on the model's output.
        Handles both single and multi-codebook cases.
        Optimized for performance with early returns and batch processing.
        """

        # This dictionary will store the new metrics
        stats = {}
        vq_ids_tensor = output_dict.get("vq_ids")
        if vq_ids_tensor is None:
            logger.warning("vq_ids_tensor is None")
            return {}

        vq_ids_long = vq_ids_tensor.long()
        attn_mask = output_dict['attn_mask']
        codebook_size = self.config.vq_codebook_size

        ndim = vq_ids_tensor.ndim
        if ndim == 3:
            stats.update(self._process_multi_codebook(vq_ids_long, attn_mask, codebook_size))
        elif ndim == 2:
            stats.update(self._process_single_codebook(vq_ids_long, attn_mask, codebook_size))
        else:
            logger.warning(f"vq_ids has unsupported dimension {ndim}. Skipping calculation.")
        return stats
        

    def _calculate_dual_vq_stats(self, output_dict: dict) -> dict:
        """
        Calculates acoustic & semantic quantization statistics based on the model's output.
        Handles both single and multi-codebook cases.
        Optimized for performance with early returns and batch processing.
        """

        # This dictionary will store the new metrics
        stats = {}
        semantic_vq_ids_tensor = output_dict.get("semantic_vq_ids")
        acoustic_vq_ids_tensor = output_dict.get("acoustic_vq_ids")
        if semantic_vq_ids_tensor is None:
            logger.warning("semantic_vq_ids_tensor is None")
        if acoustic_vq_ids_tensor is None:
            logger.warning("acoustic_vq_ids_tensor is None")
            # return {}

        semantic_vq_ids_long = semantic_vq_ids_tensor.long()
        acoustic_vq_ids_long = acoustic_vq_ids_tensor.long()
        attn_mask = output_dict['attn_mask']
        acoustic_codebook_size = self.config.vq_codebook_size
        semantic_codebook_size = self.config.get("semantic_vq_codebook_size", acoustic_codebook_size)
        
        ndim = semantic_vq_ids_tensor.ndim
        semantic_stats = {}
        if ndim == 3:
            semantic_stats = self._process_multi_codebook(semantic_vq_ids_long, attn_mask, semantic_codebook_size)
        elif ndim == 2:
            semantic_stats = self._process_single_codebook(semantic_vq_ids_long, attn_mask, semantic_codebook_size)
        else:
            logger.warning(f"semantic_vq_ids has unsupported dimension {ndim}. Skipping calculation.")
        
        for key in semantic_stats:
            stats["semantic_"+key] = semantic_stats[key]

        ndim = acoustic_vq_ids_tensor.ndim
        acoustic_stats = {}
        if ndim == 3:
            acoustic_stats = self._process_multi_codebook(acoustic_vq_ids_long, attn_mask, acoustic_codebook_size)
        elif ndim == 2:
            acoustic_stats = self._process_single_codebook(acoustic_vq_ids_long, attn_mask, acoustic_codebook_size)
        else:
            logger.warning(f"acoustic_vq_ids has unsupported dimension {ndim}. Skipping calculation.")
        for key in acoustic_stats:
            stats["acoustic_"+key] = acoustic_stats[key]

        return stats
    
    def _process_multi_codebook(self, vq_ids_long, attn_mask, codebook_sizes):
        """Process multi-codebook VQ statistics with optimized batch operations."""
        num_codebooks = vq_ids_long.shape[-1]
        stats = {}
        
        # Pre-allocate lists for batch processing
        quant_rates = []
        
        # Process all codebooks with their respective sizes
        for i in range(num_codebooks):
            indices = vq_ids_long[:, :, i]
            
            # Get codebook size for this specific codebook
            if isinstance(codebook_sizes, (list, tuple)):
                codebook_size = codebook_sizes[i] if i < len(codebook_sizes) else codebook_sizes[-1]
            else:
                codebook_size = codebook_sizes  # Single size for all codebooks
            
            quant_stat = get_quant_rates_robust(
                self, indices, attn_mask, codebook_size
            )
            perplexity = get_perplexity(
                self, indices, attn_mask, codebook_size
            )
            
            # Store individual book metrics
            stats[f"aux/perplexity_book_{i}"] = perplexity
            stats[f"aux/quant_rate_book_{i}"] = quant_stat["quant_rate"]
            stats[f"aux/unique_tokens_book_{i}"] = quant_stat["total_unique_tokens"]
            
            quant_rates.append(quant_stat["quant_rate"])
        
        stats["aux/quant_rate_avg"] = sum(quant_rates) / len(quant_rates)
        
        return stats

    def _process_single_codebook(self, vq_ids_long, attn_mask, codebook_sizes):
        """Process single-codebook VQ statistics."""
        # For single codebook, use the first size or the single size
        if isinstance(codebook_sizes, (list, tuple)):
            codebook_size = codebook_sizes[0]
        else:
            codebook_size = codebook_sizes
        
        # Compute both metrics in sequence to potentially reuse computations
        quant_stat = get_quant_rates_robust(
            self, vq_ids_long, attn_mask, codebook_size
        )
        perplexity = get_perplexity(
            self, vq_ids_long, attn_mask, codebook_size
        )
        
        return {
            "aux/quant_rate": quant_stat["quant_rate"],
            "aux/total_unique_tokens": quant_stat["total_unique_tokens"],
            "aux/total_valid_tokens": quant_stat["total_valid_tokens"],
            "aux/perplexity": perplexity
        }


    @staticmethod
    def get_vq_id_from_dict(result_dict):
        if "vq_ids" in result_dict:
            vq_id = result_dict["vq_ids"].squeeze(-1)
        elif "rvq_ids" in result_dict:
            vq_id = result_dict["rvq_ids"].squeeze(-1)
        elif "uq_ids" in result_dict:
            vq_id = result_dict["uq_ids"].squeeze(-1)
        elif "acoustic_vq_ids" in result_dict and "semantic_vq_ids" in result_dict:
            if result_dict["acoustic_vq_ids"].ndim == 3:
                vq_id = torch.cat([result_dict["semantic_vq_ids"],
                                    result_dict["acoustic_vq_ids"]], dim=-1)
            else:
                vq_id = torch.stack([result_dict["semantic_vq_ids"],
                                    result_dict["acoustic_vq_ids"]], dim=-1)
        else:
            raise ValueError("vq_ids or rvq_ids not found in result_dict")
        return vq_id
    
    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2token(self, wav, wav_len=None, **kwargs):
        return self.model.stages[0].wav2token(wav, wav_len, **kwargs)
    
    
    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2latent(self, wav, wav_len, layer_idx=12):
        return self.model.stages[0].wav2latent(wav, wav_len,layer_idx=layer_idx)
    

    def _slice_audio(
        self,
        audio: torch.Tensor,
        audio_length: torch.Tensor, # [B]
        slice_method: str,
        chunk_size: int,
        sample_rate: int
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Slices audio into chunks and returns the absolute sample length for each chunk.

        Args:
            audio (torch.Tensor): Input audio tensor of shape [B, 1, T].
            slice_method (str): Slicing method: {'full', 'even', 'max'}.
            chunk_size (int): Chunk size in seconds for slicing.
            sample_rate (int): Audio sample rate.

        Returns:
            Tuple[List[torch.Tensor], List[int]]: A tuple containing:
                - A list of audio chunks.
                - A list of absolute sample lengths for each corresponding chunk.
                (e.g., if a chunk's shape is [1, 1, 2400], its length is 2400).
        """

        if slice_method == 'full':
            chunks = [audio]
            chunk_lengths_tensor = [audio_length]
            return chunks, chunk_lengths_tensor
        else:
            n_samples = audio.shape[-1]
            if n_samples <= 0:
                chunks = []
            else:
                chunk_samples = int(chunk_size * sample_rate) 

                if chunk_samples <= 0:
                    chunks = [audio]
                else:
                    if slice_method == 'even':
                        num_chunks = math.ceil(n_samples / chunk_samples) if chunk_samples > 0 else 1
                        chunk_samples = math.ceil(n_samples / num_chunks)
                    elif slice_method == 'max': # redundant code to keep readbility
                        chunk_samples = chunk_samples

                    chunks = list(torch.split(audio, chunk_samples, dim=-1))

                    # Merge the tail if it's too short
                    if len(chunks) > 1 and chunks[-1].shape[-1] < sample_rate * 5:
                        last_chunk = chunks.pop()
                        chunks[-1] = torch.cat([chunks[-1], last_chunk], dim=-1)

        chunk_lengths_tensor = []
        cum_length = torch.zeros(audio.shape[0], dtype=torch.long, device=self.device)
        assert cum_length.shape == audio_length.shape, f"audio length tensor shape is incorrect"
        for chunk in chunks:
            cum_length += chunk.shape[-1]
            remain_length = audio_length - cum_length
            if torch.all(remain_length >= 0):
                chunk_lengths_tensor.append(torch.tensor(chunk.shape[-1], device=self.device).repeat(chunk.shape[0]))
            else:
                last_chunk_length = []
                for i in range(chunk.shape[0]):
                    if remain_length[i] < 0:
                        valid_len = remain_length[i] + chunk.shape[-1]
                        last_chunk_length.append(valid_len if valid_len > 0 else 0)
                    else:
                        last_chunk_length.append(chunk.shape[-1])
                chunk_lengths_tensor.append(torch.tensor(last_chunk_length, device=self.device))

        return chunks, chunk_lengths_tensor


    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2requires(self, audio, audio_length=None, requires=None, slice_method='full', chunk_size=45):
        """
        Args:
            requires: list of requirements, support {token, latent, tag, loss}
            audio: [B, 1, T] (please make sure the audio is mono)
            slice_method: {full, even, max}
            chunk_size: in seconds, only used when slice_method is not full, recommended value: 45 (between 30-60s)

        Return:
            output_dict: dict that contains keys and values of requires
        """

        if requires is None:
            requires = []
        if audio_length is None:
            audio_length = torch.tensor(audio.shape[-1], device=self.device).repeat(audio.shape[0])
            
        assert audio.ndim == 3 and audio.shape[1] == 1, "Input audio must be mono with shape [B, 1, T]."
        audio = audio.to(self.device)


        sample_rate = self.model.stages[0].config.sample_rate
        frame_rate = self.model.stages[0].config.frame_rate

        audio_chunks, audio_chunks_len = self._slice_audio(audio, audio_length, slice_method, chunk_size, sample_rate) 

        collected_outputs = {
            "token": [], "latent": [], "tag_pred": [], "losses": {}
        }

        for chunk_index, chunk in enumerate(audio_chunks):
           
            pseudo_label = {
                "input_ids": torch.zeros(chunk.shape[0], 1, dtype=torch.long, device=self.device),
                "attention_mask": torch.ones(chunk.shape[0], 1, dtype=torch.long, device=self.device)
            }

            if 'loss' in requires:
                result_dict = self.model(
                    {
                        "audio":chunk.squeeze(1),
                        "audio_length":audio_chunks_len[chunk_index],
                        "token": pseudo_label
                    }
                )
            else:
                result_dict = self.wav2token(
                    wav=chunk.squeeze(1),
                    wav_len=audio_chunks_len[chunk_index]
                )


            if "token" in requires:
                collected_outputs["token"].append(self.get_vq_id_from_dict(result_dict))
            if "latent" in requires:
                collected_outputs["latent"].append(result_dict["latent"])
            if "tag" in requires:
                collected_outputs["tag_pred"].append(result_dict["tag_pred"])
            if "loss" in requires:
                chunk_losses = get_task_losses(self, result_dict)
                for task, loss in chunk_losses.items():
                    if task not in collected_outputs["losses"]:
                        collected_outputs["losses"][task] = []
                    collected_outputs["losses"][task].append(loss)
        
        
        final_output = {}
        if collected_outputs["token"]:
            final_output["token"] = torch.cat(collected_outputs["token"], dim=1)
        if collected_outputs["latent"]:
            final_output["latent"] = torch.cat(collected_outputs["latent"], dim=1)

        if collected_outputs["tag_pred"]:
            avg_tag_pred = torch.stack(collected_outputs["tag_pred"], dim=0).mean(dim=0)
            tag_types = self._get_tag_types()
            final_output["tag"] = prob_to_tag(avg_tag_pred, tag_types)

        if collected_outputs["losses"]:
            for task, loss_list in collected_outputs["losses"].items():
                final_output[f"loss_{task}"] = torch.stack(loss_list).mean()

        
        return final_output


    def _slice_one_audio(
        self,
        audio: torch.Tensor, # Shape [1, 1, T]
        slice_method: str,
        chunk_size: int,
        sample_rate: int
    ) -> List[torch.Tensor]:
        """
        Slices a SINGLE audio tensor into chunks.

        Args:
            audio (torch.Tensor): A single input audio tensor of shape [1, 1, T].
            slice_method (str): Slicing method: {'full', 'even', 'max'}.
            chunk_size (int): Chunk size in seconds for slicing.
            sample_rate (int): Audio sample rate.

        Returns:
            List[torch.Tensor]: A list of audio chunks for the single audio input.
        """
        if slice_method == 'full':
            return [audio]

        n_samples = audio.shape[-1]
        if n_samples <= 0:
            return []

        chunk_samples = int(chunk_size * sample_rate)
        if chunk_samples <= 0:
            return [audio]

        if slice_method == 'even':
            # Calculate chunk size to make them as even as possible
            num_chunks = math.ceil(n_samples / chunk_samples)
            chunk_samples = math.ceil(n_samples / num_chunks)
        # For 'max', chunk_samples is already set to the maximum allowed size

        chunks = list(torch.split(audio, chunk_samples, dim=-1))

        # If the last chunk is too short, merge it with the previous one.
        # This avoids having a tiny chunk that might not be useful.
        if len(chunks) > 1 and chunks[-1].shape[-1] < sample_rate * 5: # 5-second threshold
            last_chunk = chunks.pop()
            chunks[-1] = torch.cat([chunks[-1], last_chunk], dim=-1)

        return chunks


    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def wav2requires_samplewise(self, audio, audio_length, requires=None, slice_method='full', chunk_size=45):
    

        if requires is None:
            requires = []

            
        assert audio.ndim == 3 and audio.shape[1] == 1, "Input audio must be mono with shape [B, 1, T]."
        audio = audio.to(self.device)

        final_outputs = {key: [] for key in ["token"]}


        batch_size = audio.shape[0]
        sample_rate = self.model.stages[0].config.sample_rate
        frame_rate = self.model.stages[0].config.frame_rate


        for i in range(batch_size):
            if audio_length[i] <= 0:
                final_outputs["token"].append(torch.tensor([], dtype=torch.long, device=self.device))
                continue
            # Get a single audio sample and its actual length
            one_audio_sample = audio[i:i+1, :, :audio_length[i]]

            # Slice this single audio into chunks
            audio_chunks = self._slice_one_audio(
                one_audio_sample, slice_method, chunk_size, self.model.stages[0].config.sample_rate
            )


            # We can batch the chunks of a single audio for efficiency
            chunk_batch = torch.nn.utils.rnn.pad_sequence(
                [c.squeeze(0).squeeze(0) for c in audio_chunks],
                batch_first=True
            ).unsqueeze(1) # [Num_Chunks, 1, Chunk_T]

            chunk_lengths = torch.tensor([c.shape[-1] for c in audio_chunks], device=self.device)
            pseudo_label = {
                "input_ids": torch.zeros(chunk_batch.shape[0], 1, dtype=torch.long, device=self.device),
                "attention_mask": torch.ones(chunk_batch.shape[0], 1, dtype=torch.long, device=self.device)
            }
            if 'loss' in requires:
                result_dict = self.model({
                    "audio": chunk_batch.squeeze(1),
                    "audio_length": chunk_lengths,
                    "token": pseudo_label
                })
            else:
                # Assuming wav2token can handle a batch of chunks
                result_dict = self.wav2token(
                    wav=chunk_batch.squeeze(1),
                    wav_len=chunk_lengths
                )

        
            if "token" in requires:
                concatenated_tokens = self.get_vq_id_from_dict(result_dict).flatten()
                final_outputs["token"].append(concatenated_tokens)

            if "latent" in requires:
                concatenated_latents = result_dict['hidden_states'].flatten(start_dim=0, end_dim=1)
                final_outputs["latent"].append(concatenated_latents)

            if "tag" in requires:
                raise NotImplementedError("tag_pred is not implemented in UMM2.")
                # concatenated_tags = torch.cat(result_dict["tag_pred"], dim=1) # Adjust dim as needed
                # final_outputs["tag_pred"].append(concatenated_tags)

            if "loss" in result_dict:
                sample_losses = get_task_losses(self, result_dict)
                for key, loss in sample_losses.items():
                    if key not in final_outputs:
                        final_outputs[key] = []
                    final_outputs[key].append(loss)


        # --- Collate final outputs across all samples ---
        final_collated_outputs = {}

        # 1. Pad the tokens to create a single [B, T] tensor
        if "token" in requires and final_outputs.get("token"):
            final_collated_outputs['token'] = torch.nn.utils.rnn.pad_sequence(
                final_outputs['token'], batch_first=True, padding_value=0
            )
            assert final_collated_outputs['token'].shape[0] == batch_size, f"{final_collated_outputs['token'].shape[0]} != {batch_size}"

        # Handle latent if present (returns a list of tensors)
        if "latent" in requires and final_outputs.get("latent"):
            final_collated_outputs['latent'] = final_outputs['latent']

        # 2. Average each loss type across the batch
        if "loss" in requires:
            loss_keys = [k for k in final_outputs if k not in ['token', 'latent']]
            for key in loss_keys:
                if final_outputs[key]: # Check if list is not empty
                    # Stack list of scalar tensors into one tensor and calculate mean
                    final_collated_outputs[key] = torch.stack(final_outputs[key]).mean()

    
        return final_collated_outputs        


@torch.no_grad()
@torch.cuda.amp.autocast(enabled=False)
def wav2requires_samplewise_dev(self, audio, audio_length, requires=None, slice_method='full', chunk_size=45, forward_chunk_size=32):
    """
    Processes audio in chunks for better GPU utilization.

    Args:
        audio (torch.Tensor): Input audio tensor of shape [B, 1, T].
        audio_length (torch.Tensor): Tensor containing the length of each audio sample.
        requires (list, optional): A list of required outputs. Defaults to ['token'].
        slice_method (str, optional): The method for slicing audio. Defaults to 'full'.
        chunk_size (int, optional): The size of each audio chunk in seconds. Defaults to 45.
        forward_chunk_size (int, optional): The number of chunks to process in a single forward pass. Defaults to 32.
    """
    if requires is None:
        requires = ['token']

    assert audio.ndim == 3 and audio.shape[1] == 1, "Input audio must be mono with shape [B, 1, T]."
    audio = audio.to(self.device)
    batch_size = audio.shape[0]

    # --- Step 1: Collect all chunks from all audio samples ---
    all_chunks = []
    # Keep track of which sample each chunk belongs to
    chunk_to_sample_idx = []
    # Store the number of chunks for each original audio sample
    num_chunks_per_sample = [0] * batch_size

    for i in range(batch_size):
        one_audio_sample = audio[i:i+1, :, :audio_length[i]]
        audio_chunks = self._slice_one_audio(
            one_audio_sample, slice_method, chunk_size, self.model.stages[0].config.sample_rate
        )
        
        if audio_chunks:
            all_chunks.extend(audio_chunks)
            num_chunks_per_sample[i] = len(audio_chunks)
            chunk_to_sample_idx.extend([i] * len(audio_chunks))

    if not all_chunks:
        return {} # Return early if there's no data to process

    # --- Step 2: Process all chunks in mini-batches of size `forward_chunk_size` ---
    num_total_chunks = len(all_chunks)
    # Dictionary to store the output for each chunk
    chunk_outputs = {key: [None] * num_total_chunks for key in requires}
    if 'loss' in requires:
        # Assuming 'loss' implies a set of specific loss keys
        chunk_outputs['loss_g'] = [None] * num_total_chunks 

    for i in range(0, num_total_chunks, forward_chunk_size):
        # Create a mini-batch of chunks
        chunk_batch_list = all_chunks[i:i + forward_chunk_size]
        
        chunk_batch = torch.nn.utils.rnn.pad_sequence(
            [c.squeeze(0).squeeze(0) for c in chunk_batch_list],
            batch_first=True,
            padding_value=0.0
        ).unsqueeze(1)

        chunk_lengths = torch.tensor([c.shape[-1] for c in chunk_batch_list], device=self.device)

        # --- Model Forward Pass ---
        if 'loss' in requires:
            pseudo_label = {
                "input_ids": torch.zeros(chunk_batch.shape[0], 1, dtype=torch.long, device=self.device),
                "attention_mask": torch.ones(chunk_batch.shape[0], 1, dtype=torch.long, device=self.device)
            }
            result_dict = self.model({
                "audio": chunk_batch.squeeze(1),
                "audio_length": chunk_lengths,
                "token": pseudo_label
            })
        else:
            result_dict = self.wav2token(wav=chunk_batch.squeeze(1), wav_len=chunk_lengths)

        # --- Store results for each chunk in the mini-batch ---
        for j in range(len(chunk_batch_list)):
            global_chunk_idx = i + j
            if "token" in requires:
                chunk_outputs["token"][global_chunk_idx] = self.get_vq_id_from_dict(result_dict)[j]
            if "latent" in requires:
                chunk_outputs["latent"][global_chunk_idx] = result_dict['hidden_states'][j]
            if 'loss' in requires and 'loss_g' in result_dict:
                 # Note: This assumes loss is computed per chunk or can be mapped back.
                 # If loss is per-batch, you might need to assign the same loss to all chunks in this mini-batch.
                chunk_outputs["loss_g"][global_chunk_idx] = result_dict['loss_g'].detach()

    # --- Step 3: Re-assemble chunk outputs back to their original audio samples ---
    final_outputs = {key: [] for key in requires}
    if 'loss' in requires:
        final_outputs['loss_g'] = []

    current_chunk_offset = 0
    for i in range(batch_size):
        num_chunks = num_chunks_per_sample[i]
        if num_chunks == 0:
            if "token" in requires: final_outputs["token"].append(torch.tensor([], dtype=torch.long, device=self.device))
            if "latent" in requires: final_outputs["latent"].append(torch.tensor([], device=self.device))
            # Handle losses if necessary, maybe append a zero or skip
            continue

        chunk_slice = slice(current_chunk_offset, current_chunk_offset + num_chunks)
        
        if "token" in requires:
            sample_tokens = torch.cat(chunk_outputs["token"][chunk_slice], dim=0)
            final_outputs["token"].append(sample_tokens)
        
        if "latent" in requires:
            sample_latents = torch.cat(chunk_outputs["latent"][chunk_slice], dim=0)
            final_outputs["latent"].append(sample_latents)

        if "loss" in requires and "loss_g" in chunk_outputs:
            sample_losses = [l for l in chunk_outputs["loss_g"][chunk_slice] if l is not None]
            if sample_losses:
                final_outputs["loss_g"].append(torch.stack(sample_losses).mean())

        current_chunk_offset += num_chunks

    # --- Step 4: Collate final outputs for the entire batch ---
    final_collated_outputs = {}
    if "token" in requires and final_outputs.get("token"):
        final_collated_outputs['token'] = torch.nn.utils.rnn.pad_sequence(
            final_outputs['token'], batch_first=True, padding_value=-1
        )

    if "latent" in requires and final_outputs.get("latent"):
        final_collated_outputs['latent'] = final_outputs['latent']

    if "loss" in requires:
        loss_keys = [k for k in final_outputs if 'loss' in k]
        for key in loss_keys:
            if final_outputs[key]:
                final_collated_outputs[key] = torch.stack(final_outputs[key]).mean()

    return final_collated_outputs