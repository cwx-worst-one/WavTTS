import torch 
import torchaudio

def trim_audio_list_to_same_length_and_mono(audio_list, sample_rate, duration_sec):
    """
    Take a list of loaded audio files, make them mono and trim to the same length.
    Then concat into a torch tensor.
    """
    start_idx = 0
    end_idx = start_idx + (duration_sec * sample_rate)
    mono_audio_list = []

    for x in audio_list:
        x = x[None, :, :] # add batch_size=1
        mono_audio = torch.mean(x[:, :, start_idx:end_idx], axis=1, keepdim=True)
        mono_audio_list.append(mono_audio)
    
    return torch.cat(mono_audio_list, dim=0)

def prepare_mel_for_inversion(mel_left, mel_right):
    """
    Concat the left and right mel spectrograms in a format consistent
    with the inversion algorithm
    """
    mel_for_inversion = torch.cat([mel_left[:, None, :, :], mel_right[:, None, :, :]], dim=1) 
    # mel_for_inversion has shape (batch_size, n_channels, n_timesteps, n_mels)
    mel_for_inversion = mel_for_inversion.transpose(2,3) 
    # mel_for_inversion has shape (batch_size, n_channels, n_mels, n_timesteps)
    return mel_for_inversion

def save_audio_file(audio, sample_rate, filename):
    audio = audio.cpu().detach()
    torchaudio.save(filename, audio, sample_rate)