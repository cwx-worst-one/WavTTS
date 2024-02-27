from .base_speech_loader import _BaseSpeechLoader
import data.pipe_dataloaders.data as pd
import numpy as np
import librosa
import io
import os
import soundfile as sf
import re


SPACE_NORMALIZER = re.compile(r"\s+")

class VoiceboxDataLoader(_BaseSpeechLoader):
    """
    Simple data loader that produces super-frame features and output tokenized as mixed-units. 
    Does buffer based bucket batching and shuffling. Batching criterion is frames.
    """

    def set_loader_params(self, 
                          ori_sr = 16000,
                          target_sr = 24000,
                          audio_chunk_length = 1600,
                          phone_chunk_length = 500,
                          random_chunk_prob = 1.0,
                          batch_size = 32,
                          batch_buffer_size = 1000,
                          p_drop_x=0.3, 
                          p_drop_l=0.2, 
                          p_conditional_drop=0.0,
                          p_drop_audio_frames=(0.7, 1.0), 
                          p_drop_from_0_audio_frame = None,
                          st_drop_audio_frame = 0,
                          p_drop_duration_frames=(0.1, 1.0),
                          p_drop_from_0_duration_frame = None,
                          st_drop_duration_frame = 0,
                          training_mode='audio',
                          chunk_path_gen=None,
                          chunk_paths=None,
                          filter_criteria=None,):
        """
        Kinds of data loader parameters:

        Args:
            ori_sr: original sampling rate of the chunk datas
            target_sr: target sampling rate of the chunk datas
            audio_chunk_length: the length of one audio chunk in audio mel spectrum frames
            phone_chunk_length: the length of one phone chunk in phone duration frames
            random_chunk_prob: the probability to randomly chunk the audio or duration sequence
            batch_size: batch size
            batch_buffer_size: buffer size for bucket based batching
            p_drop_x: probability to drop x (the whole audio mel spectrum frame sequence)
            p_drop_l: probability to drop l (the whole duration frame sequence)
            p_conditional_drop: probability to drop both zy and ctx, this is used for classifier-free guidance inference (see Section 3.5 in Voicebox paper)
            p_drop_audio_frames: probability to drop one audio frame in the audio mel spectrum frame sequence
            p_drop_from_0_audio_frame: probability to drop from the 0th audio frame
            st_drop_audio_frame: the start frame of the dropped segment other than the 0th frame
            p_drop_duration_frames: probability to drop one duration frame in the duration frame sequence
            p_drop_from_0_duration_frame: probability to drop from the 0th duration frame
            st_drop_duration_frame: the start frame of the dropped segment other than the 0th frame
            training_mode: audio, duration, codec, audio+phone, codec+phone
            chunk_path_gen: a string indicating how to generate the chunk path. If None, chunk path and chunk name will not be modified
            chunk_paths: a dict of chunk paths. If None, chunk path and chunk name will not be modified
            filter_criteria: a dict of filter criteria. If None, no filter will be applied
        """
        self._ori_sr = ori_sr
        self._target_sr = target_sr
        self._audio_chunk_length = audio_chunk_length
        self._phone_chunk_length = phone_chunk_length
        self._random_chunk_prob = random_chunk_prob
        self._batch_size = batch_size
        self._batch_buffer_size = batch_buffer_size
        self._p_drop_x = p_drop_x
        self._p_drop_l = p_drop_l
        self._p_conditional_drop = p_conditional_drop
        self._p_drop_audio_frames = p_drop_audio_frames
        self._p_drop_from_0_audio_frame = p_drop_from_0_audio_frame
        self._st_drop_audio_frame = st_drop_audio_frame
        self._p_drop_duration_frames = p_drop_duration_frames
        self._p_drop_from_0_duration_frame = p_drop_from_0_duration_frame
        self._st_drop_duration_frame = st_drop_duration_frame
        self._chunk_path_gen = chunk_path_gen
        self._chunk_paths = chunk_paths
        self._filter_criteria = filter_criteria
        self._training_mode = training_mode
        if self._training_mode == 'audio':
            self._batch_conf = (['audio', 'audio_ctx', 'audio_ctx_mask', 'alignment'], audio_chunk_length) # a tuple of (keys list, length), which means all the keys in the keys list will be padded into chunks with the specified length. 
        elif self._training_mode == 'duration':
            self._batch_conf = (['duration', 'duration_ctx', 'duration_ctx_mask', 'phone_sequence'], phone_chunk_length)
        elif self._training_mode == 'codec':
            self._batch_conf = (['audio', 'audio_ctx', 'audio_ctx_mask', 'alignment', 'codec'], audio_chunk_length)
        elif self._training_mode == 'audio+phone':
            self._batch_conf = (['audio', 'audio_ctx', 'audio_ctx_mask', 'alignment'], audio_chunk_length)
        elif self._training_mode == 'codec+phone':
            self._batch_conf = (['audio', 'audio_ctx', 'audio_ctx_mask', 'alignment', 'codec'], audio_chunk_length)
        elif self._training_mode == 'audio+tts_phone+codec':
            self._batch_conf = None # in this mode, we are using dynamic batching for all data
            self._phone2id = self._load_tts_phone_dict()
        elif self._training_mode == 'audio+tts_phone':
            self._batch_conf = None # in this mode, we are using dynamic batching for all data
            self._phone2id = self._load_tts_phone_dict()
        else:
            raise ValueError('training_mode must be audio, duration, codec, audio+phone, codec+phone, audio+tts_phone+codec, or audio+tts_phone')
    
    def _load_tts_phone_dict(self):
        """
        Load the tts phone dict
        """
        phone2id = {}
        tts_phone_dict = os.path.dirname(os.path.abspath(__file__)) + '/tts_phone.dictionary'
        with open(tts_phone_dict, 'r') as f:
            for line in f:
                line = line.strip()
                phone, phone_id = line.split()
                phone2id[phone] = int(phone_id)
        # add <unk> to phone2id
        phone2id['<unk>'] = len(phone2id)
        return phone2id

    def _chunk_audio(self, example):
        """
        Chunk the audio sequence at the beginning of the dataloader pipeline.
        NOTE: chunking can be either at the beginning or at the end of the dataloader pipeline.
        chunking at the beginning:
              Good thing: we can keep the meaning of various parameters, e.g., p_drop_x, p_drop_l.
              Bad thing: the implementation is complicated
        chunking at the end:
              Good thing: the implementation is easy
              Bad thing: the meaning of various parameters is changed, e.g., p_drop_x, p_drop_l, because when we do chunking we are actually dropping some data which makes p_drop_x, p_drop_l not acurate.
        """
        if example['alignment'].shape[0] > self._audio_chunk_length:
            if np.random.rand() < self._random_chunk_prob:
                start = np.random.randint(0, example['alignment'].shape[0] - self._audio_chunk_length + 1)
            else:
                start = 0
            example['alignment'] = example['alignment'][start:start+self._audio_chunk_length, ...]
            if 'audio' in example:
                example['audio'] = example['audio'][start:start+self._audio_chunk_length, ...]
            if 'codec' in example:
                example['codec'] = example['codec'][start:start+self._audio_chunk_length, ...]

        return example
    
    def _chunk_duration(self, example):
        """
        Chunk the duration sequence at the beginning of the dataloader pipeline.
        """
        # drop unncessary keys
        del example['alignment']
        del example['duration_ori']
        # do the chunking
        if example['phone_sequence'].shape[0] > self._phone_chunk_length:
            if np.random.rand() < self._random_chunk_prob:
                start = np.random.randint(0, example['phone_sequence'].shape[0] - self._phone_chunk_length + 1)
            else:
                start = 0
            example['phone_sequence'] = example['phone_sequence'][start:start+self._phone_chunk_length, ...]
            example['duration'] = example['duration'][start:start+self._phone_chunk_length, ...]
        return example

    def _transform(self, example):
        """
        Resample the audio to 24k
        args:
            example: a dict
        return:
            example: a dict
        """
        byte_stream = io.BytesIO(example['audio'])
        with sf.SoundFile(byte_stream, 'r') as f:
            samples = f.read()
        if self._ori_sr != self._target_sr:
            samples = librosa.resample(samples, orig_sr=self._ori_sr, target_sr=self._target_sr, res_type='kaiser_best')
        example['audio'] = samples
        return example
    
    def _tokenize_line(self, line):
        line = SPACE_NORMALIZER.sub(" ", line)
        line = line.strip()
        return line.split()
    
    def _convert_phone_seq_to_phone_id(self, example):
        """
        Convert phone sequence to phone id
        args:
            example: a dict
        return:
            example: a dict
        """
        phone_id = []
        words = self._tokenize_line(example['tts_phone_seq'])
        for phone in words:
            if phone not in self._phone2id:
                print(f'unknown phone {phone}, use <unk> instead')
                phone = '<unk>'
            phone_id.append(self._phone2id[phone])
        example['tts_phone_seq'] = np.array(phone_id, dtype=np.int32)
        return example

    def _get_chunk_path_name(self, chunk_name, file_types):
        if self._chunk_path_gen == "youtube900K":
            chunk_path = {}
            for file_type in file_types:
                if file_type == "audio":
                    chunk_path[file_type] = os.path.join(self._chunk_paths[file_type], os.path.dirname(chunk_name))
                else:
                    chunk_path[file_type] = self._chunk_paths[file_type]
            chunk_name = os.path.basename(chunk_name)
        else:
            raise ValueError('Unknown chunk_path_gen: {}'.format(self._chunk_path_gen))
        return chunk_path, chunk_name

    def _filter_example(self, example):
        """
        Filter the examples according to the filtering criteria
        """
        if self._filter_criteria is None:
            return True
        for key in self._filter_criteria:
            if key == "max_audio_len_seconds":
                if example['audio'].shape[0] / self._target_sr > self._filter_criteria["max_audio_len_seconds"]:
                    return False
            elif key == "min_audio_len_seconds":
                if example['audio'].shape[0] / self._target_sr < self._filter_criteria["min_audio_len_seconds"]:
                    return False
            elif key == "min_dnsmos":
                if float(example['dnsmos']) < self._filter_criteria["min_dnsmos"]:
                    return False
            elif key == "filter_multi_spk" and self._filter_criteria["filter_multi_spk"]:
                if '<SPK' in example['stosoutput']:
                    return False
            else:
                raise ValueError('Unknown filter criteria: {}'.format(key))
        return True
    
    def _drop_data(self, example):
        """
        drop the data which will not be used for training
        """
        if "dnsmos" in example: # dnsmos is only used for filtering examples, so we can drop it
            del example["dnsmos"]
        if "stosoutput" in example: # stosoutput is only used for filtering examples, so we can drop it
            del example["stosoutput"]
        return example

    def create_datablocks(self, chunk_list):
        """
        Create a data block list which will be the input of a data pipe. A data pipe is a chain of data blocks. 
        In this basic class, we just created an empty data pipe. The pipe should generate a dict of tensors, and this dict should contain the following keys:
            'ref': audio mel spectrum frames (audio mode) / per phone duration frames (duration mode)
            'ref_len': valid length of audio mel spectrum frames (audio mode) / per phone duration frames (duration mode)
            'ctx': audio mel spectrum frames with masks (audio mode) / per phone duration frames with masks (duration mode)
            'zy': frame-level phone transcript (audio mode) / phone sequence (duration mode)
        Args:
            chunk_list: A chunk list generated by BinaryChunkLister. This is usually the input of the first block

        Return:
            A list of data blocks
        """
        # regenerate the chunk path and chunk name if it is necessary
        if self._chunk_path_gen is not None:
            for idx, chunk in enumerate(chunk_list):
                chunk_path, chunk_name = self._get_chunk_path_name(chunk['chunkname'], chunk['chunktype'])
                chunk['chunkpath'] = chunk_path
                chunk['chunkname'] = chunk_name
                chunk_list[idx] = chunk

        list_sampler = pd.ListSampler(chunk_list,
                                    sharding=True,
                                    shuffle=False, 
                                    gpu_rank=self._rank, 
                                    gpu_size=self._num_replicas)
        
        chunk_deserializer = pd.BinaryChunkDeserializer()
        
        if self._training_mode in ['audio', 'audio+phone', 'codec', 'codec+phone']:
            # transform the audio bytes to audio samples, and resample to 24k
            transform = pd.Transform(self._transform)      
            # mel spectrum generator
            mel_spectrum_generator = pd.MelSpectrumGen()
            # chunking
            chunking = pd.Transform(self._chunk_audio)
        elif self._training_mode == 'duration':
            # filter the examples if the duration sequence is too short
            phone_seq_length_filter = pd.Filter(lambda example: example['duration'].shape[0] > 10)
            chunking = pd.Transform(self._chunk_duration)
        elif self._training_mode == 'audio+tts_phone+codec':
            # transform the audio bytes to audio samples, and resample to 24k
            transform = pd.Transform(self._transform)
            # filter the examples
            filters = pd.Filter(self._filter_example)
            # drop dnsmos and stosoutput
            drop_data = pd.Transform(self._drop_data)
            # mel spectrum generator
            mel_spectrum_generator = pd.MelSpectrumGen()
            # phone sequence to phone id
            phone2id = pd.Transform(self._convert_phone_seq_to_phone_id)
        
        # audio phone frame sync, interpolation
        audio_phone_frame_sync = pd.AudioPhoneSync()

        # per phone duration geneartion
        per_phone_duration_generator = pd.PerPhoneDurationGen()

        # ghost silence insertion
        ghost_silence_inserter = pd.GhostSilenceInsert()

        # masking according to voicebox
        voicebox_masking = pd.Masking(p_drop_x=self._p_drop_x, 
                                      p_drop_l=self._p_drop_l, 
                                      p_drop_audio_frames=self._p_drop_audio_frames, 
                                      p_drop_from_0_audio_frame = self._p_drop_from_0_audio_frame,
                                      st_drop_audio_frame = self._st_drop_audio_frame,
                                      p_drop_duration_frames=self._p_drop_duration_frames,
                                      p_drop_from_0_duration_frame = self._p_drop_from_0_duration_frame,
                                      st_drop_duration_frame = self._st_drop_duration_frame,
                                      p_conditional_drop=self._p_conditional_drop,)

        # batcher
        batcher = pd.BasicBatcher(self._batch_size, self._batch_buffer_size)

        # batch to tensors
        batch_to_tensors = pd.VoiceboxBatchToTensor(self._batch_conf)

        # renaming the batches
        batch_wrapper = pd.VoiceboxBatchWrapper(self._training_mode)
        
        if self._training_mode in ['audio', 'audio+phone', 'codec', 'codec+phone']:
            datablock_list = [
                list_sampler,
                chunk_deserializer,
                transform,
                mel_spectrum_generator,
                audio_phone_frame_sync, # keys: audio, alignment, (codec)
                chunking, # keys: audio, alignment, (codec)
                per_phone_duration_generator, # keys: audio, alignment, (codec), duration, duration_ori, phone_sequence
                ghost_silence_inserter, # keys: audio, alignment, (codec), duration, duration_ori, phone_sequence
                voicebox_masking, # keys: audio, audio_ctx, audio_ctx_mask, alignment, (codec), duration, duration_ctx, duration_ctx_mask, duration_ori, phone_sequence
                batcher, # keys: audio, audio_ctx, audio_ctx_mask, alignment, (codec), duration, duration_ctx, duration_ctx_mask, duration_ori, phone_sequence
                batch_to_tensors, # keys: (audio, audio_ctx, audio_ctx_mask,, alignment) for audio mode, (audio, audio_ctx, audio_ctx_mask, alignment, codec) for codec mode
                batch_wrapper, # keys: (ref, ref_len, ctx, ctx_mask, zy) for audio mode or (ref, ref_len, ctx, ctx_mask, zy, codec) for codec mode
            ]
        elif self._training_mode == 'duration':
            datablock_list = [
                list_sampler,
                chunk_deserializer,
                audio_phone_frame_sync, # keys: alignment
                per_phone_duration_generator, # keys: alignment, duration, duration_ori, phone_sequence
                phone_seq_length_filter, # keys: alignment, duration, duration_ori, phone_sequence
                ghost_silence_inserter, # keys: alignment, duration, duration_ori, phone_sequence
                chunking, # keys: phone_sequence, duration
                voicebox_masking, # keys: duration, duration_ctx, duration_ctx_mask, phone_sequence
                batcher, # keys: duration, duration_ctx, duration_ctx_mask, phone_sequence
                batch_to_tensors, # keys: duration, duration_ctx, duration_ctx_mask, phone_sequence
                batch_wrapper, # keys: ref, ref_len, ctx, ctx_mask, zy
            ]
        elif self._training_mode == 'audio+tts_phone+codec':
            datablock_list = [
                list_sampler,
                chunk_deserializer, # keys: audio, codec, tts_phone_seq, dnsmos, stosoutput
                transform, # keys: audio, codec, tts_phone_seq, dnsmos, stosoutput
                filters, # keys: audio, codec, tts_phone_seq, dnsmos, stosoutput
                drop_data, # keys: audio, codec, tts_phone_seq
                phone2id, # keys: audio, codec, tts_phone_seq
                mel_spectrum_generator, # keys: audio, codec, tts_phone_seq
                audio_phone_frame_sync, # keys: audio, codec, tts_phone_seq
                voicebox_masking, # keys: audio, audio_ctx, audio_ctx_mask, codec, tts_phone_seq
                batcher, # keys: audio, audio_ctx, audio_ctx_mask, codec, tts_phone_seq
                batch_to_tensors, # keys: audio, audio_len, audio_ctx, audio_ctx_mask, codec, tts_phone_seq
                batch_wrapper, # keys: ref, ref_len, ctx, ctx_mask, codec, zy
            ]
        elif self._training_mode == 'audio+tts_phone':
            datablock_list = [
                list_sampler,
                chunk_deserializer, # keys: audio, tts_phone_seq, dnsmos, stosoutput
                transform, # keys: audio, tts_phone_seq, dnsmos, stosoutput
                filters, # keys: audio, tts_phone_seq, dnsmos, stosoutput
                drop_data, # keys: audio, tts_phone_seq
                phone2id, # keys: audio, tts_phone_seq
                mel_spectrum_generator, # keys: audio, tts_phone_seq
                voicebox_masking, # keys: audio, audio_ctx, audio_ctx_mask, tts_phone_seq
                batcher, # keys: audio, audio_ctx, audio_ctx_mask, tts_phone_seq
                batch_to_tensors, # keys: audio, audio_len, audio_ctx, audio_ctx_mask, tts_phone_seq
                batch_wrapper, # keys: ref, ref_len, ctx, ctx_mask, zy
            ]
        else:
            raise ValueError('Unknown training mode: {}'.format(self._training_mode))

        return datablock_list