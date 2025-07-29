import torch
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from recipes.umm2.modules.utils import get_task_losses
import inspect

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, blank=266, padding=0):
        super().__init__()
        self.blank = blank
        self.padding = padding

    def forward(self, emission_batch):
        res = []
        for emission in emission_batch:
            indices = torch.argmax(emission, dim=-1)  # [num_seq,]
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [
                i.item() for i in indices if (i != self.blank and i != self.padding)
            ]
            res.append(indices)
        return res

def song_slice_to_audio_slice(song_slice, chunk_dur=30):
    # chunk song slice into audio slices
    audio_slices = []
    text_slices = []
    st, et = 0, 0
    text = ""
    for phrase in song_slice.phrases:
        _st, _et = phrase.time_span
        if phrase.text == "" or phrase.lyrics_confidence is None or \
            (phrase.lyrics_confidence is not None and phrase.lyrics_confidence < 0.6):
            this_text = ""
        else:
            this_text = phrase.text
        text += this_text

        if _et - st > chunk_dur:
            audio_slices.append([st, _et])
            text_slices.append(text)
            st = _et
            text = ""
        else:
            et = _et

    if _et - st < 10:
        audio_slices[-1][1] = _et
        text_slices[-1] += text
    else:
        audio_slices.append([st, _et])
        text_slices.append(text)
    return audio_slices, text_slices

def detect_consecutive_repetitions(numbers):
    if not numbers:
        return {}
    
    consecutive_counts = {}
    current_num = numbers[0]
    current_count = 1
    for num in numbers[1:]:
        if num == current_num:
            current_count += 1
        else:
            if current_count not in consecutive_counts:
                consecutive_counts[current_count] = 1
            else:
                consecutive_counts[current_count] += 1
            current_num = num
            current_count = 1
    if current_count not in consecutive_counts:
        consecutive_counts[current_count] = 1
    else:
        consecutive_counts[current_count] += 1
    return consecutive_counts

def levenshtein_distance(hypothesis: list, reference: list):
    """levenshtein distance between two sequences
    C: correct
    W: wrong
    I: insert
    D: delete
    S: substitution

    :param hypothesis 
    :param reference
    """
    len_hyp = len(hypothesis)
    len_ref = len(reference)
    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)
    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j

    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hypothesis[i-1] == reference[j-1]:
                cost_matrix[i][j] = cost_matrix[i-1][j-1]
            else:
                substitution = cost_matrix[i-1][j-1] + 1
                insertion = cost_matrix[i-1][j] + 1
                deletion = cost_matrix[i][j-1] + 1
                compare_val = [substitution, insertion, deletion]   # 优先级

                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = [] 
    i = len_hyp
    j = len_ref
    nb_map = {"N": len_ref, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:     # correct
            if i-1 >= 0 and j-1 >= 0:
                match_idx.append((j-1, i-1))
                nb_map['C'] += 1
            i -= 1
            j -= 1
        elif ops_matrix[i_idx][j_idx] == 2:   # insert
            i -= 1
            nb_map['I'] += 1
        elif ops_matrix[i_idx][j_idx] == 3:   # delete
            j -= 1
            nb_map['D'] += 1
        elif ops_matrix[i_idx][j_idx] == 1:   # substitute
            i -= 1
            j -= 1
            nb_map['S'] += 1

        if i < 0 and j >= 0:
            nb_map['D'] += 1
        elif j < 0 and i >= 0:
            nb_map['I'] += 1

    match_idx.reverse()
    wrong_cnt = cost_matrix[len_hyp][len_ref]
    nb_map["W"] = wrong_cnt

    return wrong_cnt, match_idx, nb_map


class TokenEvaluator(torch.nn.Module):
    def __init__(self, pl_module, config, model_type="VQ", 
                 segment_size=30, slice_length=[5, 15, 30], inference_R=None):
        super().__init__()
        self.pl_module = pl_module
        self.config = config
        self.model = pl_module.model
        self.segment_size = segment_size
        self.slice_length = slice_length
        self.sample_rate = config.sample_rate
        self.frame_rate = config.frame_rate
        self.ctc_decoder = GreedyCTCDecoder(blank=self.config.ctc_blank_id, padding=0)
        if model_type == "UQ":
            self.codebook_size = 2 ** (int(math.log2(self.config.vq_codebook_size)) * self.config.vq_codebook_dim)
            self.h = 1
        elif model_type == "HUQ":
            self.codebook_size = 2 ** (int(math.log2(self.config.vq_codebook_size)) * int(self.config.vq_codebook_dim // self.model.uq.h))
            self.h = self.model.uq.h
        else:
            self.codebook_size = self.config.vq_codebook_size
            self.h = self.config.get("rvq", 1)
        print("codebook_size: ", self.codebook_size)
        self.load_required_modules()

        # support inference with given R (RVQ)
        self.inference_R = inference_R


    def load_required_modules(self, local_rank=0):
        self.pl_module.requires = {}
        for module_name, loader_config in self.pl_module.required_modules.items():
            if module_name == "pretrained":
                continue
            print(f"loading module {module_name}...")
            if "loader" in loader_config:
                _args = {k: v for k, v in loader_config.items() if k != "loader"}
                loader = loader_config["loader"](**_args)
                self.pl_module = loader.load_model(pl_module=self.pl_module)
            else:
                hpath = loader_config['hpath']
                initializer = loader_config['initializer']
                self.pl_module.requires.update(initializer(hpath, local_rank=local_rank, cache_dir="./"))


    @torch.no_grad()
    def get_tokens(self, audio, audio_len=None, slice_mode="full", chunk_dur=60, slice_info=None):
        """
        audio shape: [B, n_signal_sample]
        """
        sliced_audios = []
        slice_texts = None
        n_samples = audio.shape[-1]
        n_secs = float(audio.shape[-1]) / self.sample_rate

        if slice_mode == 'full':
            vq_id = self.get_vq_id(audio, audio_len)
            sliced_audios.append(audio)
        
        elif slice_mode in ['max', 'even']:
            vq_id = []
            # evenly slice the audio in a way that the `chunk_size` is as close as possible to `chunk_dur`
            if slice_mode == 'even':  
                chunk_num = math.ceil(n_secs / chunk_dur)
                chunk_size = math.ceil(n_secs / chunk_num)
            # always slice the audio with the maximum `chunk_dur`, combine the tail audio < 1s
            elif slice_mode == 'max':
                chunk_size = chunk_dur
            
            st = 0
            while st < n_samples:
                _st, _et = int(st*self.sample_rate), int((st+chunk_size)*self.sample_rate)
                # merge the tail if the remaining chunk is too short (< 1s) 
                if n_samples - _et < self.sample_rate * 1 or audio[...,_et:].shape[-1] < self.sample_rate * 1:
                    _et = n_samples
                _vq_id = self.get_vq_id(audio[..., _st:_et], None)
                vq_id.append(_vq_id)
                sliced_audios.append(audio[..., _st:_et])
                if _et >= n_samples:
                    break
                st += chunk_size
            vq_id = torch.cat(vq_id, dim=1)
        elif slice_mode == 'section':
            assert slice_info is not None
            audio_slice, slice_texts = song_slice_to_audio_slice(slice_info, chunk_dur)
            vq_id = []
            for st, et in audio_slice:
                _st, _et = int(st*self.sample_rate), int(et*self.sample_rate)
                if n_samples - _et < self.sample_rate * 1 or audio[...,_et:].shape[-1] < self.sample_rate * 1:
                    _et = n_samples
                _vq_id = self.get_vq_id(audio[..., _st:_et], None)
                vq_id.append(_vq_id)
                sliced_audios.append(audio[..., _st:_et])
            vq_id = torch.cat(vq_id, dim=1)
        else:
            raise NotImplementedError(f"slice_mode {slice_mode} not implemented")
        
        return vq_id, sliced_audios, slice_texts
    

    def get_vq_id(self, audio, audio_len):
        result_dict = self.pl_module.wav2token(audio, audio_len, inference_R=self.inference_R)

        if "vq_ids" in result_dict:
            vq_id = result_dict["vq_ids"]
        elif "rvq_ids" in result_dict:
            vq_id = result_dict["rvq_ids"]
        elif "uq_ids" in result_dict:
            vq_id = result_dict["uq_ids"]
        else:
            raise ValueError("vq_ids or rvq_ids not found in result_dict")
        return vq_id

    def get_loss_dict(self, output_dict):
        loss_dict = get_task_losses(self.pl_module, output_dict)
        return loss_dict
    
    def get_outputs(self, output_dict):
        outputs = {}
        for key in output_dict:
            if key.endswith("out"):
                outputs[key] = output_dict[key]
        return outputs
        
    def compute_locality(self, ref_token, token):
        assert ref_token.shape == token.shape
        return torch.sum(ref_token == token) / torch.prod(torch.tensor(ref_token.shape))

    def update_code_usage(self, tokens):
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(-1)
        for i in range(tokens.shape[-1]):
            self.code_count[i] += torch.bincount(tokens[...,i].flatten().cpu(),
                                                 minlength=self.codebook_size)

    def reset_code_count(self):
        self.h = self.inference_R if self.inference_R is not None else self.h
        self.code_count = torch.zeros(self.h, self.codebook_size)

    def compute_code_rate(self):
        code_rate = []
        for i in range(self.code_count.shape[0]):
            code_rate.append(torch.sum(self.code_count[i] > 0) / len(self.code_count[i]))
        return code_rate
    
    def plot_token_distribution(self, path="./test.png"):
        fig = plt.figure()
        colors = ["r", "blue", "g", "black"]
        legend_list = []
        for r in range(self.code_count.shape[0]):
            if r > len(colors):
                plt.plot(self.code_count[r])
            else:
                plt.plot(self.code_count[r], color=colors[r])
            legend_list.append(f"{r+1}")
        plt.legend(legend_list)
        plt.savefig(path)
        plt.close(fig)

        fig = plt.figure()
        for r in range(self.code_count.shape[0]):
            sorted_code_count = self.code_count[r]
            sorted_code_count = sorted(sorted_code_count)[::-1]
            if r > len(colors):
                plt.plot(sorted_code_count)
            else:
                plt.plot(sorted_code_count, color=colors[r])
        plt.legend(legend_list)
        path_prefix, path_suffix = ".".join(path.rsplit(".")[:-1]), path.rsplit(".")[-1]
        plt.savefig(path_prefix + "_sorted." + path_suffix)
        plt.close(fig)

    @torch.no_grad()
    def locality(self, audio):
        locality = {}
        for _slice_len in self.slice_length:
            if _slice_len > audio.shape[-1] / self.sample_rate:
                continue
            # fix as the center
            start_sec = int((audio.shape[-1] / self.sample_rate - _slice_len) / 2)
            target_audio_slice = audio[..., start_sec*self.sample_rate: (start_sec + _slice_len) * self.sample_rate]


            target_tokens, _, _ = self.get_tokens(target_audio_slice, None)
            # 1) padding zeros
            left_padding_sec = int((self.segment_size - _slice_len) / 2) # make sure it can be divided by frame_rate
            right_padding_sec = self.segment_size - _slice_len - left_padding_sec
            zero_padding_audio_segment = F.pad(target_audio_slice, 
                                               (left_padding_sec*self.sample_rate, int(right_padding_sec*self.sample_rate)), 
                                                mode='constant', value=0)

            zero_padding_tokens, _, _ = self.get_tokens(zero_padding_audio_segment)
            st_token = left_padding_sec * self.frame_rate
            zero_padding_target_tokens = zero_padding_tokens[:, st_token: st_token+target_tokens.shape[1]]
            # print("zero padding", zero_padding_target_tokens.shape, target_tokens.shape)
            zero_padding_locality = self.compute_locality(zero_padding_target_tokens, target_tokens)

            # 2) with contexts
            tokens, _, _ = self.get_tokens(audio)
            st_token = start_sec * self.frame_rate
            all_tokens_target_tokens = tokens[:, st_token:st_token+target_tokens.shape[1]]
            # print("all token ", all_tokens_target_tokens.shape, target_tokens.shape)
            context_locality = self.compute_locality(all_tokens_target_tokens, target_tokens)
            
            # 3) move audio to left/right border
            left_border_tokens, _, _ = self.get_tokens(torch.cat((target_audio_slice, audio), dim=-1))
            # print("left_border_tokens", left_border_tokens[:,:target_tokens.shape[1]].shape, target_tokens.shape)
            left_border_locality = self.compute_locality(left_border_tokens[:,:target_tokens.shape[1]], target_tokens)

            right_border_tokens, _, _ = self.get_tokens(torch.cat((audio, target_audio_slice), dim=-1))
            # print("right_border_tokens", right_border_tokens[:,-target_tokens.shape[1]:].shape, target_tokens.shape)
            right_border_locality = self.compute_locality(right_border_tokens[:,-target_tokens.shape[1]:], target_tokens)
            
            locality[f"{_slice_len}s"] = [zero_padding_locality, context_locality, left_border_locality, right_border_locality]

        return locality


    def locality_chunk(self, audio, chunk_size):
        """
        Computes the token locality for a given audio by comparing the tokens
        generated from the full audio with the tokens generated by processing
        the audio in smaller, independent chunks.

        This method assesses how much the tokenization of a specific audio segment
        is influenced by its surrounding context. A high locality score suggests
        that the token output for a segment is consistent, regardless of whether
        it's processed in isolation or as part of a larger audio file.

        Args:
            audio (torch.Tensor): The input audio waveform, expected to be in a
                                tensor format, e.g., [batch, channels, samples].
            chunk_size (int or float): The duration of each chunk in seconds. The
                                    audio will be split into segments of this
                                    length for batch processing.

        Returns:
            dict: A dictionary containing the locality score. The key is formatted
                as f"chunk_locality_{chunk_size}", and the value is the computed
                locality result from self.compute_locality.
        """

        audio_legnth = audio.shape[-1]

        # Convert audio_legnth to a tensor on the same device as audio
        audio_length_tensor = torch.tensor(audio_legnth, device=audio.device).unsqueeze(0)

        # Get tokens for full audio
        full_tokens, *_ = self.get_tokens(audio, audio_len=audio_length_tensor)
                
        # Calculate samples per chunk (assuming sample rate in self.sample_rate)
        samples_per_chunk = int(chunk_size * self.sample_rate)
        
        # Create batch of chunks
        chunks = []
        chunk_lengths = []
        for i in range(0, audio.shape[-1], samples_per_chunk):
            chunk = audio[..., i:i + samples_per_chunk]
            chunks.append(chunk)
            chunk_lengths.append(chunk.shape[-1])
        
        # Pad chunks to same length for batching
        max_chunk_len = max(chunk_lengths)
        padded_chunks = []
        for chunk in chunks:
            if chunk.shape[-1] < max_chunk_len:
                padding = torch.zeros(*chunk.shape[:-1], max_chunk_len - chunk.shape[-1], device=chunk.device)
                chunk = torch.cat([chunk, padding], dim=-1)
            padded_chunks.append(chunk)
        
        # Stack into batch and get tokens
        batch_chunks = torch.stack(padded_chunks, dim=0)
        batch_lengths = torch.tensor(chunk_lengths, device=audio.device)
        batch_tokens, *_ = self.get_tokens(batch_chunks, batch_lengths)
        
        # Extract tokens up to actual lengths and concatenate
        chunk_tokens = []
        for i, length in enumerate(chunk_lengths):
            tokens_len = int(length * batch_tokens.shape[-1] / max_chunk_len)
            chunk_tokens.append(batch_tokens[i, :tokens_len])
        chunk_tokens = [t for t in chunk_tokens if t.numel() > 0]
        
        # Concatenate chunk tokens
        concatenated_tokens = torch.cat(chunk_tokens, dim=0) if chunk_tokens else torch.empty(0, device=audio.device)
        full_tokens = full_tokens.squeeze(0)

        min_len = min(concatenated_tokens.shape[0], full_tokens.shape[0])

        # Compare with full tokens

        result =  self.compute_locality(concatenated_tokens[0:min_len], full_tokens[0:min_len])
        
        return {
            f"chunk_locality_{chunk_size}": result
        }

    def token_repetition(self, tokens):
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(-1)
        return detect_consecutive_repetitions(tokens[...,0].squeeze().tolist())
    
    def ctc_wer(self, output_text_logits, ref_text_id, tokenizer=None):
        """
        output_text_logits: [B, n_output, n_logits]
        ref_text_id: [B, n_ref]
        """
        if ref_text_id.ndim == 1:
            ref_text_id = ref_text_id.unsqueeze(0)

        gt_text_ids = []
        for text_id in ref_text_id:
            gt_text_ids.append(
                [
                    i.item()
                    for i in text_id
                    if (i != self.ctc_decoder.blank and i != self.ctc_decoder.padding)
                ]
            )
        est_text_ids = self.ctc_decoder(output_text_logits)

        transcript = None
        if tokenizer is not None:
            transcript = [tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(text_id))
                          for text_id in est_text_ids]

        wers = []
        for b in range(len(gt_text_ids)):
            wrong_cnt, _, nb_map = levenshtein_distance(hypothesis=est_text_ids[b], reference=gt_text_ids[b])
            if nb_map["N"] == 0:
                continue
            wer, ins, _del, sub = nb_map["W"] / nb_map["N"], nb_map["I"] / nb_map["N"], nb_map["D"] / nb_map["N"], nb_map["S"] / nb_map["N"]
            wers.append([wer, ins, _del, sub])

        if transcript is None:
            return wers
        else:
            return wers, transcript
