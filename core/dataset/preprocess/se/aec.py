import random
import numpy as np
import torch
import librosa
import scipy
from .augmentation import (
    iir_aug,
    nlp_simulation_custom,
    tde,
    fix_length_zero,
    fix_length_repeat,
    fix_lengths_zeros,
    mix,
    parallel_aec_rlsyz_cpu_dsp_no_sub_16k,
)


def aec_load_simulate_data(self, item):
    """Simulator entry for aec training.
    Args:
        item: dict contains outside data

    Return:
        item: include data simulated.
    """
    gen_cfg = self.gen_cfg
    max_delay = int(gen_cfg.max_delay / 1000 * gen_cfg.sampling_rate)

    cfg_key = self.parse_block('clean')
    speech_data_out_dict = self._data_manager.get_audio_and_rir(
        waveform=item[self.wav_key],
        cfg_key=cfg_key,
        use_outside_data='true',
        length=None,
        # for rir data
        ref_direction=None,
        src_idx=0,
    )
    speech = speech_data_out_dict['audio_data']

    if random.random() < gen_cfg.speech_aug_ratio:
        if random.random() < gen_cfg.eq_ratio:
            speech = iir_aug(speech)
        else:
            speech = nlp_simulation_custom(speech)

    speech, mask = fix_length_zero(speech, self._max_speech_length)
    speech_data_out_dict['audio_data'] = speech

    snr_ser = [0, 0]
    # noise
    noise_cfg_key = self.parse_block('noise')
    noise_data_dict = self._data_manager.get_audio_and_rir(cfg_key=noise_cfg_key)  # norir

    noise = noise_data_dict['audio_data']
    noise = fix_length_repeat(noise, self._max_speech_length)
    noise_data_dict['audio_data'] = noise
    snr_ser[0] = (
        self._cfg[noise_cfg_key]['snr_min']
        + (self._cfg[noise_cfg_key]['snr_max'] - self._cfg[noise_cfg_key]['snr_min'])
        * random.random()
    )

    # reference & echo
    if random.random() < gen_cfg.real_echo_ratio:
        tmp = self._data_manager._audio_data.read_from_reader(
            data_type='ref_echo', unserialize_type='pickle'
        )
        echo, ref = tmp['echo'], tmp['ref']
        agc_ratio = gen_cfg.agc_min + (gen_cfg.agc_max - gen_cfg.agc_min) * random.random()
        agc_ratio = agc_ratio / (np.max(np.abs(ref)) + 1e-3)
        ref = ref * agc_ratio

        # est_time_delay
        if gen_cfg.return_tde > 0:
            min_length = min(echo.shape[-1], ref.shape[-1])
            time_delay = tde(
                echo[:min_length], ref[:min_length], whole_utterance_tde=False, return_tde=True
            )
        else:
            time_delay = np.zeros(echo.shape)
    else:
        # fake_echo : speech : fake_echo_opt == 0.7 : 0.3 * 0.6 : 0.3 * 0.4
        # ref
        if random.random() < 0.7:
            ref = self._data_manager._audio_data.read_from_reader(data_type='fake_echo').astype(
                np.float32
            )
        elif random.random() < 0.6:
            ref = self._data_manager._audio_data.read_from_reader(data_type='speech').astype(
                np.float32
            )
        else:
            ref = self._data_manager._audio_data.read_from_reader(data_type='fake_echo_opt').astype(
                np.float32
            )
        # fake echo
        # pitch perturb
        if random.random() < gen_cfg.pitch_ratio:
            step = random.randint(-3, 3)
            echo = librosa.effects.pitch_shift(ref, sr=16000, n_steps=step)
        else:
            echo = ref.copy()

        # eq perturb
        if random.random() < gen_cfg.eq_ratio:
            echo = iir_aug(echo)[0, :]

        # nlp simulation
        echo = nlp_simulation_custom(echo)
        # rir
        rir, _ = self._data_manager._rir_data.read_from_reader(data_type='direct')

        rir_delay = np.argmax(np.abs(rir))
        # fake delay
        fake_delay = random.randint(1, max_delay)
        ori_echo_len = echo.shape[-1]
        if random.random() < gen_cfg.delay_perturb_ratio:
            echo_out1 = np.concatenate([np.zeros(fake_delay), echo], axis=0)[:ori_echo_len]
            perturb_delay = fake_delay + int(
                random.randint(1, max_delay) * np.sign(random.random() - 0.5)
            )
            perturb_delay = max(perturb_delay, 0)
            echo_out2 = np.concatenate([np.zeros(perturb_delay), echo], axis=0)[:ori_echo_len]
            perturb_start = random.randint(ori_echo_len // 4, ori_echo_len // 4 * 3)
            echo[:perturb_start] = echo_out1[:perturb_start]
            echo[perturb_start:] = echo_out2[perturb_start:]
            time_delay = np.ones(echo.shape) * (fake_delay + rir_delay)
            time_delay[perturb_start:] = perturb_delay + rir_delay
        else:
            echo = np.concatenate([np.zeros(fake_delay), echo], axis=0)[:ori_echo_len]
            time_delay = np.ones(echo.shape) * (fake_delay + rir_delay)
        # convolution
        echo = scipy.signal.fftconvolve(echo, rir, 'full')[: echo.shape[-1]]

    # length_check
    if echo.shape[-1] != ref.shape[-1]:
        l_min = min(echo.shape[-1], ref.shape[-1])
        echo = echo[:l_min]
        ref = ref[:l_min]
    # clipping
    if random.random() < gen_cfg.clip_ratio:
        echo = echo / (np.max(np.abs(echo)) + 1e-3) * gen_cfg.agc_max * (1 + 4 * random.random())
        echo = np.clip(echo, a_min=-gen_cfg.agc_max, a_max=gen_cfg.agc_max)
    if random.random() < gen_cfg.ref_clip_ratio:
        ref = ref / (np.max(np.abs(ref)) + 1e-3) * gen_cfg.agc_max * (1 + 4 * random.random())
        ref = np.clip(ref, a_min=-gen_cfg.agc_max, a_max=gen_cfg.agc_max)
    ref = ref.reshape(1, -1)
    echo = echo.reshape(1, -1)
    time_delay = time_delay.reshape(1, -1)
    outs = fix_lengths_zeros([ref, echo, time_delay], self._max_speech_length)
    ref, echo, time_delay = outs[0], outs[1], outs[2]

    snr_ser[1] = gen_cfg['ser_min'] + (gen_cfg['ser_max'] - gen_cfg['ser_min']) * random.random()

    speech_flag = random.random() < gen_cfg.speech_ratio
    noise_flag = random.random() < gen_cfg.noise_ratio
    echo_flag = random.random() < gen_cfg.echo_ratio
    item['speech_flag'] = speech_flag
    item['noise_flag'] = noise_flag
    item['echo_flag'] = echo_flag
    item['echo'] = echo
    item['ref'] = ref
    item['time_delay'] = time_delay
    item['clean'] = speech_data_out_dict
    item['noise'] = noise_data_dict
    item['length'] = self._max_speech_length
    item['snr'] = snr_ser
    item['speech_mask'] = mask
    return item


def aec_simulate_collate_fn(self, batch_data, batch_out):
    '''aec simulator collate function'''
    # collate clean
    batch_out['clean'] = self.collate_audio_and_rir('clean', batch_data)
    # collate disturb
    batch_out['noise'] = self.collate_audio_and_rir('noise', batch_data)


def aec_simulator_fn(self, item):
    '''aec simulator function'''
    gen_cfg = self._cfg.general
    snr = item.pop('snr')
    shape = item.pop('batch_shape')
    mask = item.pop('speech_mask')
    bsz, mic_num, length = shape
    clean_data = item.pop('clean')
    clean_data = self.get_audio_data(clean_data, bsz, mic_num, length, snr.device)
    speech = clean_data.clone()
    speech *= mask.reshape(bsz, 1, -1)

    noise_data = item.pop('noise')
    noise = self.get_audio_data(noise_data, bsz, mic_num, length, snr.device)
    echo = item.pop('echo')
    ref = item.pop('ref')
    echo = mix(speech, echo, snr[..., 1], speech_flag=True, frame_max=False, gen_cfg=gen_cfg)
    noise = mix(speech, noise, snr[..., 0], speech_flag=True, frame_max=True, gen_cfg=gen_cfg)

    speech = speech * item['speech_flag'].reshape(-1, 1, 1)
    echo = echo * item['echo_flag'].reshape(-1, 1, 1)
    ref = ref * item['echo_flag'].reshape(-1, 1, 1)
    noise = noise * item['noise_flag'].reshape(-1, 1, 1)

    noisy_speech = speech + noise
    mic = noisy_speech + echo

    # agc
    agc_ratio = self.rand_agc_ratio(mic)
    mic *= agc_ratio
    echo *= agc_ratio
    noise *= agc_ratio
    speech *= agc_ratio
    noisy_speech *= agc_ratio
    ratio_ref = self.rand_agc_ratio(ref)
    ref *= ratio_ref

    mic = mic[:, 0, :]
    ref = ref[:, 0, :]
    echo = echo[:, 0, :]
    noise = noise[:, 0, :]
    noisy_speech = noisy_speech[:, 0, :]
    speech = speech[:, 0, :]

    aec, ref_tde = parallel_aec_rlsyz_cpu_dsp_no_sub_16k(mic, ref)
    # 40ms delay from linear aec
    aec = torch.nn.functional.pad(aec[:, 640:], [0, 640], 'constant', 0.0)

    item['ref'] = ref
    item['echo'] = echo
    item['mic'] = mic
    item['noise'] = noise
    item['noisy_speech'] = noisy_speech
    item['speech'] = speech
    item['aec'] = aec
    item['ref_tde'] = ref_tde

    return item


def aec_infer_simulator_fn(self, item):
    '''aec inference simulator function'''
    mic = item['mic']
    if self._cfg.general.get('inference_linear_aec_flag', False):
        ref = item['ref']
        aec, ref_tde = parallel_aec_rlsyz_cpu_dsp_no_sub_16k(mic, ref)
        # 40ms delay from linear aec
        aec = torch.nn.functional.pad(aec[:, 640:], [0, 640], 'constant', 0.0)
    else:
        aec = item['aec']
        ref_tde = item['ref_tde']
        aec = aec[..., 800:]  # remove 50ms aec delay

    # length align
    min_length = min(mic.shape[-1], ref_tde.shape[-1])
    min_length = min(min_length, aec.shape[-1])
    item['mic'] = mic[..., :min_length]
    item['aec'] = aec[..., :min_length]
    item['ref_tde'] = ref_tde[..., :min_length]
    return item
