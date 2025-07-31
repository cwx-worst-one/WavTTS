from pathlib import Path
import librosa
import torch
from recipes.sacodec.modules.sacodec_module_umm import SACodecModule
from tqdm import tqdm
# from IPython.display import Audio
import torchaudio

def load_audio(audio_fp, to_24k=True):
    # librosa way
    audio, sr = librosa.load(audio_fp, sr=44100, mono=False)
    input_audio = torch.from_numpy(audio).unsqueeze(0).float().to('cuda:0')
    if len(input_audio.shape) == 2:
        input_audio = input_audio.unsqueeze(0)

    assert sr == 44100, f"Audio not in 44100k {sr}"
    if input_audio.shape[1] == 1:
        print("Converting mono to stereo", input_audio.shape)
        input_audio = input_audio.repeat(1, 2, 1)
    
    if input_audio.shape[-1] > 44100 * 500:
        print('Long Input audio shape', input_audio.shape)
        input_audio = input_audio[:, :, :44100 * 500]
        
    
    if to_24k:
        input_audio = torchaudio.functional.resample(
            input_audio, orig_freq=44100, new_freq=24000
        )
        input_audio = input_audio.mean(dim=-2, keepdim=True)

    return input_audio, sr

def main(ckpt_path, reference_dir, output_dir):
    model = SACodecModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        strict=False
    ).cuda().eval()
    model.setup('predict')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_audio_files = [f for f in Path(reference_dir).glob('*.wav') if f.is_file()]

    # iterate through the audio files
    for audio_fp in tqdm(input_audio_files):
        recon_audio_fp = output_dir / Path(audio_fp).relative_to(reference_dir)
        recon_audio_fp.parent.mkdir(parents=True, exist_ok=True)
        if recon_audio_fp.exists(): continue
            
        input_audio, sr = load_audio(audio_fp, to_24k=False)
        with torch.no_grad():
            y_g = model.reconstruct_audio(input_audio.cuda())
            # logamp, pha, rea, imag = model.encoder.audio_to_spec(input_audio.cuda())
            # encoder_results = model.encoder(logamp, pha, return_loss=False)
            # decoder_latent = encoder_results["latent"]
            # logamp_g, pha_g, rea_g, imag_g, y_g = model.decoder(decoder_latent)

        # Save the reconstructed audio
        torchaudio.save(
            uri=str(recon_audio_fp),
            src=y_g.squeeze(0).cpu(),
            sample_rate=sr
        )
        
        del y_g
        torch.cuda.empty_cache()


## TO download validation dataset:
"""
sudo apt install zip unzip -y
valset_local_dir=/tmp
valset_hdfs_path=hdfs://haruna/home/byte_data_seed/lf_lq/speech/user/andrew.shaw/datasets/valset_56wavs_120s.zip
valset_name="$(basename -- $valset_hdfs_path .zip)"
hdfs dfs -get $valset_hdfs_path $valset_local_dir
unzip $valset_local_dir/$valset_name.zip -d $valset_local_dir
meta_lst=$valset_local_dir/$valset_name/test_full.lst

reference_dir = /tmp/valset_56wavs_120s/reference

python3 recipes/sacidec/scripts/reconstruct.py \
    --ckpt_path /mnt/bn/ashaw-lq/checkpoints/0317_mp3_compress/sacodec_checkpoint_epoch=0_step=180000_val_loss=1.0580.ckpt \
    --reference_dir /tmp/valset_56wavs_120s/reference \
    --output_dir /mnt/bn/ashaw-lq/eval/results_debug/reconstruct_test/
"""

"""
To Run metrics:

OUTPUT_PATH=/mnt/bn/ashaw-lq/eval/results_latent2wav/0317_mp3_compress/baseline
python3 recipes/codec_benchmark/metrics.py     --ref_audio_dir /mnt/bn/ashaw-lq/eval/valset_56wavs_120s/reference     --pred_audio_dir $OUTPUT_PATH
fadtk clap-laion-audio /mnt/bn/ashaw-lq/eval/valset_56wavs_120s/reference $OUTPUT_PATH
"""


if '__main__' == __name__:
    import argparse
    parser = argparse.ArgumentParser(
    )
    parser.add_argument("--ckpt_path", type=str, help="Path to the model")
    parser.add_argument("--reference_dir", type=str, help="Path to the wav directory")
    parser.add_argument("--output_dir", type=str, help="Path to save audio")
    args = parser.parse_args()

    main(args.ckpt_path, args.reference_dir, args.output_dir)

# ## test
# if '__main__' == __name__:
#     reference_dir = "/mnt/bn/ashaw-lq/eval/valset_56wavs_120s/reference"
#     # ckpt_path = '/mnt/bn/ashaw-lq/checkpoints/0317_mp3_compress/sacodec_checkpoint_epoch=0_step=180000_val_loss=1.0580.ckpt'
#     ckpt_path = ".module_cache/vocoder/sacodec/sacodec_checkpoint_epoch=0_step=180000_val_loss=1.1525.ckpt"
#     output_dir = "/mnt/bn/ashaw-lq/eval/results_debug/reconstruct_test/"

#     main(ckpt_path, reference_dir, output_dir)
