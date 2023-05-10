''' BaseCeModel and BaseCeSolution '''

import random
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from dataloader import FalconReader
from core.models.vad.ce_encoder import *
from core.models.vad.acoustic_head import *

# pylint: disable=unused-import
from core.criterions import *
from core.utils import logging, FalconDict
from core.solutions.inference import BaseInfer, INFERS
from core.solutions.base_solution import BaseSolution, register_solution


class VadCeModelExporter(BaseInfer):
    '''export vad ce model to onnx'''

    NAME = 'vad_ce'
    INPUTS = [FalconDict(name='fbank', type=torch.float32, shape=['B', 'T', -1])]
    OUTPUTS = [FalconDict(name='encoder_out', type=torch.float32, shape=['B', 'T', -1])]

    def __init__(self, model, **kwargs):
        '''
        init function
        '''
        super().__init__(kwargs)
        self.encoder = model.encoder
        self.args = model.args
        self.eval()
        fbank_dim = kwargs.get('fbank_dim')
        tgt_vocab_size = kwargs.get('tgt_vocab_size')
        self._inputs[0].shape[2] = fbank_dim
        self._outputs[0].shape[2] = tgt_vocab_size
        self._convert_stream_flag = kwargs.get('encoder_convert_stream', True)

    def forward(self, fbank):
        '''
        forward for vad ce model
        '''
        encoder_out, _, _, _ = self.encoder(fbank)
        output_layer_type = self.args.get('output_layer_type', 'softmax')
        if output_layer_type == 'softmax':
            encoder_out = F.softmax(encoder_out.float(), dim=-1)
        elif output_layer_type == 'log_softmax':
            encoder_out = F.log_softmax(encoder_out.float(), dim=-1)
        else:
            raise RuntimeError('not supported output layer type: %s' % (output_layer_type))
        return encoder_out

    def sample_inputs(self):
        fbank = self._generate_input_data(0, method='ones', dynamic_axis=[1, 512, -1])
        datas = [fbank]
        if self._convert_stream_flag:
            global_state_in = self._generate_input_data(0, dynamic_axis=[1, -1])
            datas.append(global_state_in)
        return tuple(datas)

    def load_cmvn(self, mean, inv_std):
        '''
        Load cmvn for model export
        '''
        self.eval()
        mvn_concat_size = self.args.get('mvn_concat_size', 1)
        with torch.no_grad():
            self.encoder.mean.copy_(torch.tensor(np.tile(mean, mvn_concat_size)))
            self.encoder.inv_std.copy_(torch.tensor(np.tile(inv_std, mvn_concat_size)))


@register_solution("VadCeModel")
class VadCeModel(BaseSolution):
    '''
    VAD CE model.
    '''

    def __init__(self, args):
        '''
        init function for CE skeleton model.
        '''
        super().__init__()
        self.args = args
        self.encoder = eval(args.encoder_type)(args)
        self.criterion = eval(args.criterion_type)(args)

    def forward(self, batch_data):
        '''
        Forward for CE base module
        '''
        fbank = batch_data['src']
        fbank_mask = batch_data['src_mask']
        ce_label = batch_data['ce_label']
        encoder_out, _, encoder_mask, target = self.encoder(fbank, fbank_mask, ce_label)

        # criterion for loss computation
        forward_out = self.criterion(encoder_out, fbank_mask, target, encoder_mask)
        return forward_out

    def load_cmvn(self, mean, inv_std):
        '''
        Load cmvn for model export
        '''
        self.eval()
        mvn_concat_size = self.args.get('mvn_concat_size', 1)
        with torch.no_grad():
            self.encoder.mean.copy_(torch.tensor(np.tile(mean, mvn_concat_size)))
            self.encoder.inv_std.copy_(torch.tensor(np.tile(inv_std, mvn_concat_size)))

    def register_infers(self):
        infers = [VadCeModelExporter(self, **self.args)]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)

    def export(self, *_args, **_kwargs):
        '''export onnx'''
        for infer_name in self._infer_names:
            INFERS[infer_name].export()


class VadTsCeModelExporter(BaseInfer):
    '''export ts-vad model to onnx'''

    NAME = 'ts-vad'
    INPUTS = [
        FalconDict(name='fbank', type=torch.float32, shape=[1, -1, 'T']),
        FalconDict(name='embedding', type=torch.float32, shape=[1, -1, -1]),
    ]
    OUTPUTS = [FalconDict(name='activition', type=torch.float32, shape=[1, -1, 'T'])]

    def __init__(self, model, **kwargs):
        '''
        init function
        '''
        super().__init__(kwargs)
        self.args = model.args
        self.encoder = model.encoder
        fbank_dim = kwargs.get('fbank_dim')
        self.num_speakers = model.args.num_classes
        self.embedding_dim = model.encoder.front_end_linear.output_dim

        # fbank: [B, F, T], embedding: [B, C, E], activation: [B, C, T']
        self._inputs[0].shape[1] = fbank_dim
        self._inputs[1].shape[1] = self.num_speakers
        self._inputs[1].shape[2] = self.embedding_dim
        self._outputs[0].shape[1] = self.num_speakers
        self.eval()

    def forward(self, fbank, embedding):
        '''
        forward for vad ce model
        '''
        encoder_out = torch.sigmoid(self.encoder(fbank, embedding)).transpose(1, 2)
        return encoder_out

    def sample_inputs(self):
        fbank = self._generate_input_data(0, dynamic_axis=[1, -1, 512])
        embedding = self._generate_input_data(1, dynamic_axis=[1, -1, -1])
        return fbank, embedding


class MultiVadTsCeModelExporter(BaseInfer):
    '''export multi ts-vad model to onnx'''

    NAME = 'multi-ts-vad'
    INPUTS = [
        FalconDict(name='fbank', type=torch.float32, shape=[1, -1, -1, 'T']),
        FalconDict(name='embedding', type=torch.float32, shape=[1, -1, -1]),
    ]
    OUTPUTS = [FalconDict(name='activition', type=torch.float32, shape=[1, -1, 'T'])]

    def __init__(self, model, **kwargs):
        '''
        init function
        '''
        super().__init__(kwargs)
        self.args = model.args
        self.encoder = model.encoder
        fbank_dim = kwargs.get('fbank_dim')
        channel_dim = kwargs.get('channel_dim')
        self.num_speakers = model.args.num_classes
        self.embedding_dim = model.encoder.front_end_linear.output_dim

        # fbank: [B, C, F, T], embedding: [B, C, E], activation: [B, C, T']
        self._inputs[0].shape[1] = channel_dim
        self._inputs[0].shape[2] = fbank_dim
        self._inputs[1].shape[1] = self.num_speakers
        self._inputs[1].shape[2] = self.embedding_dim
        self._outputs[0].shape[1] = self.num_speakers
        self.eval()

    def forward(self, fbank, embedding):
        '''
        forward for vad ce model
        '''
        encoder_out = torch.sigmoid(self.encoder(fbank, embedding)).transpose(1, 2)
        return encoder_out

    def sample_inputs(self):
        fbank = self._generate_input_data(0, dynamic_axis=[1, -1, -1, 512])
        embedding = self._generate_input_data(1, dynamic_axis=[1, -1, -1])
        return fbank, embedding


@register_solution("VadTsCeModel")
class VadTsCeModel(BaseSolution):
    '''
    VAD Ts CE model.
    '''

    def __init__(self, args):
        '''
        init function for Ts-CE skeleton model.
        '''
        super().__init__()
        self.args = args
        args.setdefault("num_classes", args.target_speaker_num)
        self.encoder = eval(args.encoder_type)(args)
        self.criterion = eval(args.criterion_type)(args)
        self.embeddings = None
        self.use_same_utt_embedding = args.get("use_same_utt_embedding", True)
        self._register_load_state_dict_pre_hook(self._model_load_hook)

    def init_embeddings(self, filelist):
        '''
        load speaker embedding for training
        '''
        chunk_size = 1000
        parrallel_chunk_num = 30
        reader = FalconReader(filelist, chunk_size)
        keys = []
        embeddings = []
        reader_keys = reader.list_keys()
        entry_num = len(reader_keys)
        chunk_idxs = [i * chunk_size for i in range(entry_num // chunk_size)]
        for st in range(0, len(chunk_idxs), parrallel_chunk_num):
            chunks = chunk_idxs[st : st + parrallel_chunk_num]
            tmp_embeddings = sum(reader.read_many(chunks, True), [])
            key_start = chunk_idxs[st]
            for off in range(len(chunks) * chunk_size):
                key_idx = key_start + off
                if len(reader_keys[key_idx]) <= 0:
                    continue
                keys.append(reader_keys[key_idx])
            for tmp_emb in tmp_embeddings:
                embeddings.append(pickle.loads(tmp_emb))
        del reader
        self.spk2idx = {}
        self.embeddings = [0] * len(keys)
        for key_id, (key, emb) in enumerate(zip(keys, embeddings)):
            if key in self.spk2idx:
                logging.warning(
                    "WARNING: multiple entries are found for spk %s in the embeddings. "
                    "Keep the first one.",
                    key,
                )
                continue

            # The key of the embedding is now "utt-spk-index"
            if self.use_same_utt_embedding:
                # If using the embeddings from the same utterance,
                # the speaker name is "utt-spk", else the name is simply "spk".
                spk_name = key.rsplit("-", 1)[0]
            else:
                spk_name = key.rsplit("-", 2)[1]

            self.spk2idx.setdefault(spk_name, [])
            self.spk2idx[spk_name].append(key_id)
            self.embeddings[key_id] = emb
        self.embeddings = torch.tensor(np.stack(self.embeddings), dtype=torch.float32).to(
            self.encoder.ts_detections.weight.device
        )
        self.speakers = list(set(self.spk2idx.keys()))
        self.embed_dim = self.embeddings.shape[-1]

    def get_embeddings(self, batch_data):
        '''
        get speaker embedding
        '''
        if self.embeddings is None:
            self.init_embeddings(self.args.embedding_list)

        batch_emb = torch.zeros(
            (len(batch_data['speakers']), len(batch_data['speakers'][0]), self.embed_dim),
            device=batch_data['feature'].device,
        )
        for batch_idx, speaker_list in enumerate(batch_data['speakers']):
            real_speaker_list = ['' if spk == '' else spk.rsplit("-", 1)[1] for spk in speaker_list]
            if '' in speaker_list:
                random.shuffle(self.speakers)
                spk_idx = 0
                for idx, spk in enumerate(speaker_list):
                    if spk == '':
                        while (
                            self.use_same_utt_embedding
                            and self.speakers[spk_idx].rsplit("-", 1)[1] in real_speaker_list
                        ) or (
                            not self.use_same_utt_embedding
                            and self.speakers[spk_idx] in real_speaker_list
                        ):
                            spk_idx += 1
                        if self.use_same_utt_embedding:
                            speaker_list[idx] = self.speakers[spk_idx]
                            real_speaker_list[idx] = self.speakers[spk_idx].rsplit("-", 1)[1]
                        else:
                            real_speaker_list[idx] = self.speakers[spk_idx]

            if self.use_same_utt_embedding:
                # randomly select an embedding from the same speaker in the same utterance
                batch_emb[batch_idx] = self.embeddings[
                    [random.choice(self.spk2idx[spk]) for spk in speaker_list]
                ]
            else:
                # randomly select an embedding from the same speaker in any utterances
                batch_emb[batch_idx] = self.embeddings[
                    [random.choice(self.spk2idx[spk]) for spk in real_speaker_list]
                ]
        return batch_emb

    def forward(self, batch_data):
        '''
        Forward for Ts-CE base module
        '''
        # The resnet front-end needs the input shape to be [B, F, T]
        fbank = batch_data['feature'].transpose(-1, -2)
        ts_ce_label = batch_data['label'].transpose(1, 2)  # [B, T, N]
        # TODO (ts-vad): the mask may be used in the future
        fbank_mask = torch.ones_like(fbank[:, 0, :])
        if self.args.label_pooling != 0:
            ts_ce_label = (
                F.avg_pool2d(ts_ce_label, (self.args.label_pooling, 1), ceil_mode=True) >= 0.5
            ).float()
            target_mask = (
                F.avg_pool1d(
                    fbank_mask.unsqueeze(1), (self.args.label_pooling), ceil_mode=True
                ).squeeze(1)
                >= 0.5
            )
        ts_embeddings = self.get_embeddings(batch_data)

        # This is necessary to avoid the model overfitting
        speaker_num = ts_embeddings.shape[1]
        speakers_list = list(range(speaker_num))
        shuffle_list = random.sample(speakers_list, speaker_num)
        shuffle_ts_embeddings = ts_embeddings[:, shuffle_list, :]
        shuffle_ts_ce_label = ts_ce_label[:, :, shuffle_list]

        # criterion for loss computation
        logits = self.encoder(fbank, shuffle_ts_embeddings, fbank_mask, shuffle_ts_ce_label)
        forward_out = self.criterion(logits, fbank_mask, shuffle_ts_ce_label, target_mask)
        return forward_out

    def _model_load_hook(
        self,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        '''Load model from pretrained ResNet model'''
        model_keys = [f"{prefix}{key}" for key in self.state_dict().keys()]
        old_state_dict = state_dict.copy()
        state_dict.clear()

        def get_name(old_name):
            old_name = (
                old_name.replace("acoustic_backbone_module", "front_end_resnet")
                .replace("pooling_module", "front_end_gsp")
                .replace("segment_network_module.mlp", "front_end_linear")
            )
            for key in model_keys:
                if old_name in key:
                    return key
            return old_name

        for name, param in old_state_dict.items():
            new_name = get_name(name)
            state_dict[new_name] = param

    def inference(self, fbank, ts_embeddings):
        '''
        Args:
            fbank: [B, F, T] or [F, T]
            ts_embeddings: [B, C, E] or [C, E]
        Return:
            target_act: [B, C, T']
            The activations for C speakers at T' time (down-sampled by factor 8)
        '''
        if fbank.ndim == 2 and ts_embeddings.ndim == 2:
            fbank = fbank.unsqueeze(dim=0)
            ts_embeddings = ts_embeddings.unsqueeze(dim=0)
        assert (
            fbank.ndim == 3 and ts_embeddings.ndim == 3 and fbank.shape[0] == ts_embeddings.shape[0]
        )
        if isinstance(fbank, np.ndarray):
            fbank = torch.from_numpy(fbank)
        if isinstance(ts_embeddings, np.ndarray):
            ts_embeddings = torch.from_numpy(ts_embeddings)
        assert isinstance(fbank, torch.Tensor) and isinstance(ts_embeddings, torch.Tensor)
        # Copy the data into the right device
        fbank = fbank.to(self.encoder.ts_detections.weight.device)
        ts_embeddings = ts_embeddings.to(self.encoder.ts_detections.weight.device)
        with torch.no_grad():
            target_logits = self.encoder(fbank, ts_embeddings)
            # From logits to probabilities
            target_act = torch.sigmoid(target_logits)
        return target_act.transpose(1, 2).squeeze()

    def register_infers(self):
        infers = [VadTsCeModelExporter(self, **self.args)]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)


class VadCeAdaptModelEncoderExporter(BaseInfer):
    '''export vad ce adapt model's encoder to onnx'''

    NAME = 'vad_ce_adapt_encoder'
    INPUTS = [FalconDict(name='fbank', type=torch.float32, shape=['B', 'T', -1])]
    OUTPUTS = [FalconDict(name='encoder_out', type=torch.float32, shape=['B', 'T', -1])]

    def __init__(self, model, **kwargs):
        '''
        init function
        '''
        super().__init__(kwargs)
        self.encoder = model.encoder
        self.args = model.args
        self.eval()
        fbank_dim = kwargs.get('fbank_dim')
        self._inputs[0].shape[2] = fbank_dim
        if kwargs.get('selected_dfsmn_layer', -1) != -1:
            self._outputs.append(
                FalconDict(name='selected_encoder_out', type=torch.float, shape=['B', 'T', -1])
            )

    def forward(self, fbank):
        '''
        Forward for vad ce adapt model
        '''
        encoder_out, selected_encoder_out, _, _ = self.encoder(fbank)

        if selected_encoder_out is not None:
            return encoder_out, selected_encoder_out

        return encoder_out

    def sample_inputs(self):
        fbank = self._generate_input_data(0, method='ones', dynamic_axis=[1, 512, -1])
        return fbank


class VadCeAdaptModelHeadExporter(BaseInfer):
    '''export vad ce adapt model's head to onnx'''

    NAME = 'vad_ce_adapt_head'
    INPUTS = [FalconDict(name='encoder_out', type=torch.float32, shape=['B', 'T', -1])]
    OUTPUTS = [FalconDict(name='head_out', type=torch.float32, shape=['B', 'T', -1])]

    def __init__(self, model, **kwargs):
        super().__init__(kwargs)
        self.ce_head = model.ce_head
        self.args = model.args
        self.eval()
        head_input_size = kwargs.get('head_input_size')
        self._inputs[0].shape[2] = head_input_size

    def forward(self, encoder_out):
        '''
        Forward for vad ce adapt model
        '''
        head_out = self.ce_head(encoder_out)
        output_layer_type = self.args.get('output_layer_type', 'softmax')
        if output_layer_type == 'softmax':
            head_out = F.softmax(head_out.float(), dim=-1)
        elif output_layer_type == 'log_softmax':
            head_out = F.log_softmax(head_out.float(), dim=-1)
        else:
            raise RuntimeError('not supported output layer type: %s' % (output_layer_type))
        return head_out

    def sample_inputs(self):
        encoder_out = self._generate_input_data(0, method='ones', dynamic_axis=[1, 512, -1])
        return encoder_out


@register_solution("MultiVadTsCeModel")
class MultiVadTsCeModel(VadTsCeModel):
    '''
    Muti channel VAD Ts CE model.
    '''

    def get_embeddings(self, batch_data):
        '''
        get speaker embedding
        '''
        if self.embeddings is None:
            self.init_embeddings(self.args.embedding_list)

        batch_emb = torch.zeros(
            (
                len(batch_data['speakers']),
                len(batch_data['speakers'][0]),
                self.args.num_channels,
                self.embed_dim,
            ),
            device=batch_data['feature'].device,
        )
        for batch_idx, speaker_list in enumerate(batch_data['speakers']):
            real_speaker_list = ['' if spk == '' else spk.rsplit("-", 1)[1] for spk in speaker_list]
            if '' in speaker_list:
                random.shuffle(self.speakers)
                spk_idx = 0
                for idx, spk in enumerate(speaker_list):
                    if spk == '':
                        while (
                            self.use_same_utt_embedding
                            and self.speakers[spk_idx].rsplit("-", 1)[1] in real_speaker_list
                        ) or (
                            not self.use_same_utt_embedding
                            and self.speakers[spk_idx] in real_speaker_list
                        ):
                            spk_idx += 1
                        if self.use_same_utt_embedding:
                            speaker_list[idx] = self.speakers[spk_idx]
                            real_speaker_list[idx] = self.speakers[spk_idx].rsplit("-", 1)[1]
                        else:
                            real_speaker_list[idx] = self.speakers[spk_idx]

            if self.use_same_utt_embedding:
                # randomly select an embedding from the same speaker in the same utterance
                batch_emb[batch_idx] = self.embeddings[
                    [random.choice(self.spk2idx[spk]) for spk in speaker_list]
                ]
            else:
                # randomly select an embedding from the same speaker in any utterances
                batch_emb[batch_idx] = self.embeddings[
                    [random.choice(self.spk2idx[spk]) for spk in real_speaker_list]
                ]
        return batch_emb

    def forward(self, batch_data):
        '''
        Forward for Ts-CE base module
        '''
        # The resnet front-end needs the input shape to be [B, C, F, T]
        fbank = batch_data['feature'].transpose(-1, -2)
        ts_ce_label = batch_data['label'].transpose(1, 2)  # [B, T, N]
        fbank_mask = torch.ones_like(fbank[:, 0, 0, :])
        if self.args.label_pooling != 0:
            ts_ce_label = (
                F.avg_pool2d(ts_ce_label, (self.args.label_pooling, 1), ceil_mode=True) >= 0.5
            ).float()
            target_mask = (
                F.avg_pool1d(
                    fbank_mask.unsqueeze(1), (self.args.label_pooling), ceil_mode=True
                ).squeeze(1)
                >= 0.5
            )
        ts_embeddings = self.get_embeddings(batch_data)

        # This is necessary to avoid the model overfitting
        speaker_num = ts_embeddings.shape[1]
        speakers_list = list(range(speaker_num))
        shuffle_list = random.sample(speakers_list, speaker_num)
        shuffle_ts_embeddings = ts_embeddings[:, shuffle_list, :]
        shuffle_ts_ce_label = ts_ce_label[:, :, shuffle_list]

        # criterion for loss computation
        logits = self.encoder(fbank, shuffle_ts_embeddings, fbank_mask, shuffle_ts_ce_label)
        forward_out = self.criterion(logits, fbank_mask, shuffle_ts_ce_label, target_mask)
        return forward_out

    def inference(self, fbank, ts_embeddings):
        '''
        Args:
            fbank: [B, C, F, T] or [C, F, T]
            ts_embeddings: [B, N, C, E] or [N, C, E]
        Return:
            target_act: [B, N, T']
            The activations for N speakers at T' time (down-sampled by factor 8)
        '''
        if fbank.ndim == 3 and ts_embeddings.ndim == 3:
            fbank = fbank.unsqueeze(dim=0)
            ts_embeddings = ts_embeddings.unsqueeze(dim=0)
        assert (
            fbank.ndim == 4 and ts_embeddings.ndim == 4 and fbank.shape[0] == ts_embeddings.shape[0]
        )
        if isinstance(fbank, np.ndarray):
            fbank = torch.from_numpy(fbank)
        if isinstance(ts_embeddings, np.ndarray):
            ts_embeddings = torch.from_numpy(ts_embeddings)
        assert isinstance(fbank, torch.Tensor) and isinstance(ts_embeddings, torch.Tensor)
        # Copy the data into the right device
        fbank = fbank.to(self.encoder.ts_detections.weight.device)
        ts_embeddings = ts_embeddings.to(self.encoder.ts_detections.weight.device)
        with torch.no_grad():
            target_logits = self.encoder(fbank, ts_embeddings)
            # From logits to probabilities
            target_act = torch.sigmoid(target_logits)
        return target_act.transpose(1, 2).squeeze()


@register_solution("VadCeAdaptModel")
class VadCeAdaptModel(BaseSolution):
    '''
    VAD CE adaptation model.
    '''

    def __init__(self, args):
        '''
        init function for CE skeleton model.
        '''
        super().__init__()
        self.args = args
        self.encoder = eval(args.encoder_type)(args)
        self.ce_head = eval(args.head_type)(args)
        self.criterion = eval(args.criterion_type)(args)
        if self.args.freeze_encoder:
            print('freeze encoder')
            for _, param in self.encoder.named_parameters():
                param.requires_grad = False

    def forward(self, batch_data):
        '''
        Forward for CE base module
        '''
        if self.args.freeze_encoder:
            self.encoder.eval()

        fbank = batch_data['src']
        fbank_mask = batch_data['src_mask']
        ce_label = batch_data['ce_label']
        encoder_out, selected_encoder_out, encoder_mask, target = self.encoder(
            fbank, fbank_mask, ce_label
        )

        if self.args.head_use_selected_encoder_out:
            head_input = selected_encoder_out
        else:
            head_input = encoder_out
        head_out = self.ce_head(head_input)

        # criterion for loss computation
        forward_out = self.criterion(head_out, fbank_mask, target, encoder_mask)

        return forward_out

    def load_cmvn(self, mean, inv_std):
        '''
        Load cmvn for model export
        '''
        self.eval()
        mvn_concat_size = self.args.get('mvn_concat_size', 1)
        with torch.no_grad():
            self.encoder.mean.copy_(torch.tensor(np.tile(mean, mvn_concat_size)))
            self.encoder.inv_std.copy_(torch.tensor(np.tile(inv_std, mvn_concat_size)))

    def register_infers(self):
        infers = [
            VadCeAdaptModelEncoderExporter(self, **self.args),
            VadCeAdaptModelHeadExporter(self, **self.args),
        ]
        self._infer_names = []
        for infer in infers:
            assert infer.NAME not in INFERS
            INFERS[infer.NAME] = infer
            self._infer_names.append(infer.NAME)

    def export(self, *_args, **_kwargs):
        '''export onnx'''
        for infer_name in self._infer_names:
            INFERS[infer_name].export()
