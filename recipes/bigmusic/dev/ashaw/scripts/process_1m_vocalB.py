import samantha.utils.hdfs_helper as hh
from tqdm import tqdm
import json
import pandas as pd
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, Any
import argparse
import numpy as np

def is_audio_metrics_good(audio_metrics: Dict[str, Any]) -> Tuple[bool, str]:
    # Clipping
    clip = audio_metrics.get("clipping", {})
    if clip.get("rate", 0) >= 5e-5:
        return False
    for ch in ["left", "right"]:
        if clip.get(f"peak_rate_{ch}", 0) >= 0.05:
            return False
    # Loudness
    loudness = audio_metrics.get("loudness", {})
    if (
        loudness.get("integrated_loudness", -7) > -5
        or loudness.get("max_mom_loud", -7) >= 0
        or loudness.get("max_short_term_loud", -7) >= 0
    ):
        return False
    # RMS stats
    rms_stats = audio_metrics.get("rms_stats", {})
    if rms_stats.get("peak", 0) > 3:
        return False
    for ch in ["left", "right"]:
        if (
            rms_stats.get(f"{ch}_total", -10) > -5
            or rms_stats.get(f"{ch}_total", -10) < -40
            or rms_stats.get(f"normed_std_{ch}", -10) < -19.5
        ):
            return False
    # Cutoff frequency
    cutoff_freq = audio_metrics.get("cutoff_frequency", {})
    for ch in ["left", "right"]:
        if (
            cutoff_freq.get(f"rel_{ch}", 48000) < 15000
            and cutoff_freq.get(f"rel_{ch}_conf", 0) > 0.6
            and cutoff_freq.get(f"band_std_{ch}", 10) < 5
        ):
            return False
    # Phase
    phase = audio_metrics.get("phase_check", {})
    if (
        phase.get("has_phase_issue", False)
        or abs(phase.get("rms_downmix_diff", 0.1)) > 3
    ):
        return False
    return True

def is_valid_metadata(metadata):
#     if metadata['meta_song_language'] != 'en' or metadata['final_language'] != 'English': 
    if metadata['final_language'] != 'English': 
#         print('Invalid', metadata['meta_song_language'], metadata['final_language'])
        return False
    valid_audio_metrics = is_audio_metrics_good(metadata.get('audio_metrics', {}))
    if not valid_audio_metrics: 
#         print('Invalid', metadata['audio_metrics'].keys())
        return False
    return True

def is_valid_lyrics(lyrics, confidence_threshold=0.8):
    if lyrics is None: 
        return False
    confidences = []
    for utterance in lyrics:
        if 'confidence' in utterance:
            confidence = float(utterance["confidence"])
        elif 'additions' in utterance:
            confidence = float(utterance["additions"]["confidence"])
        else:
            # some lyrics may not have confidence (force alignment). return True
            return True
        if confidence == 0:
            continue
        confidences.append(confidence)
    if len(confidences) == 0:
        return False
    return np.array(confidences).mean() > confidence_threshold

def extract_metadata_and_utterances(index_data):
    if 'metadata' in index_data:
        metadata = index_data['metadata']
    else:
        metadata = index_data
    # Hiphop has format metadata: {..., lyrics: []}, THe rest has format { metadata: {}, lyrics: []}
    if 'lyrics' in metadata:
        lyrics = metadata['lyrics']
    elif 'lyrics' in index_data:
        lyrics = index_data['lyrics']
    else:
        lyrics = None

    utterances = None
    if lyrics and 'utterances' in lyrics:
        # v1 (asr, no punctuation)
        utterances = lyrics['utterances']
    elif lyrics and 'result' in lyrics:
        # v2 (asr + punctuation)
        utterances = lyrics['result'][0]['utterances']

    return metadata, lyrics, utterances

def run_genres(genre_splits, label="B", target_size=1_000_000):
    index_dir = Path('/mnt/bn/lyrics-to-song/ashaw/data/mcc/mcc60_lossless_asr/url2idx')
    index_downloaded_dir = Path('/mnt/bn/lyrics-to-song/ashaw/data/mcc/mcc60_lossless_asr/indexes_vocal_merge_downloaded')
    output_dir = Path(f"/mnt/bn/lyrics-to-song/ashaw/data/mcc/mcc60_lossless_asr/vocal_{label}/{target_size}/indexes")
    output_dir.mkdir(exist_ok=True, parents=True)
    index_list_fps = list(index_dir.glob(f'vocal-{label}-*.txt'))

    dir2final_genre = {
    #     'alternative-rock': 'Alternative Rock',
        'blues': 'Blues',
        'childhood': 'Childhood',
        'classical': 'Classical',
        'country': 'Country',
        'devotional': 'Devotional',
        'easy-listening': 'Easy Listening',
        'electronic': 'Electronic',
    #     'experimental': 'Experimental',
        'folk': 'Folk',
        'hip-hop-rap': 'Hip Hop',
    #     'indie-folk': 'Indie Folk',
    #     'indie-pop': 'Indie Pop',
        'jazz': 'Jazz',
        'metal': 'Metal',
        'new-age': 'New Age',
        'pop': 'Pop',
        'r-b-soul': 'R&B',
        'reggae': 'Reggae',
        'rock': 'Rock',
        'soundtrack': 'SoundTrack',
    #     'techno': 'Techno',
    #     'trance': 'Trance',
        'trap-rap': 'Trap Rap',
    }

    valid_final_genres = set(dir2final_genre.values())
    num_songs_per_genre = round(target_size // len(valid_final_genres), -2)
    # num_songs_per_genre = 62000

    if genre_splits is not None:
        genre_splits = genre_splits.split(',')
        dir2final_genre = { k:v for k,v in dir2final_genre.items() if k in genre_splits }

    final_genre_count = Counter()
    for index_list_fp  in index_list_fps:
        df_url2idx = pd.read_csv(index_list_fp, sep='\t', header=None)
        output_url2idx = output_dir.parent/index_list_fp.name

        index_list_genre = index_list_fp.stem.replace(f'vocal-{label}-', '')
        if index_list_genre not in dir2final_genre: 
            print('index_list_genre not in supported', index_list_genre)
            continue
        if output_url2idx.exists():
            output_url2idx.unlink()
        pbar = tqdm(df_url2idx.iterrows())
        for idx, (tar_fp, meta_index_fp_str) in pbar:
            meta_index_fp = Path(meta_index_fp_str)

            dir_genre = str(meta_index_fp.stem).replace(f'vocal-{label}-', '').split('_shard')[0]
            if dir_genre not in dir2final_genre: 
                print('dir_genre not in supported', dir_genre)
                continue

            target_final_genre = dir2final_genre[dir_genre]
            current_count = final_genre_count[target_final_genre]
            # Already have enough songs. Move on
            if current_count > num_songs_per_genre: continue
            
            relative_path_fp = '/'.join(meta_index_fp.parts[-4:])
            output_fp = output_dir/relative_path_fp
            output_fp.parent.mkdir(parents=True, exist_ok=True)
            local_meta_index_fp = index_downloaded_dir/relative_path_fp
            if not local_meta_index_fp.exists():
                local_meta_index_fp.parent.mkdir(parents=True, exist_ok=True)
    #             print('Downloading to local meta:', local_meta_index_fp)
                hh.get(meta_index_fp_str, local_meta_index_fp)
            df_index = pd.read_csv(local_meta_index_fp, sep='\t', header=None)
            final_genres = []
            lines = []
            for idx, (song_id, meta_str) in df_index.iterrows():
                meta_json = json.loads(meta_str)
                
                # Hiphop has format metadata: {..., lyrics: []}, THe rest has format { metadata: {}, lyrics: []}
                metadata, lyrics, utterances = extract_metadata_and_utterances(meta_json)
                song_final_genre = metadata['final_genre']
                song_final_genre = song_final_genre.split(',')[0].split('/')[0]
                if song_final_genre != target_final_genre: 
                    print('Final genre does not equal', song_final_genre, target_final_genre)
                    continue
                if not is_valid_metadata(metadata): 
                    # print('Invalid metadata')
                    continue
                if not is_valid_lyrics(utterances):
                    # print('Invalid lyrics')
                    continue
                final_genres.append(target_final_genre)
                meta_str = json.dumps({ 'metadata': metadata, 'lyrics': lyrics })
                out_str = f'{song_id}\t{meta_str}\n'
                lines.append(out_str)
            if len(lines) / df_index.shape[0] < 0.1:
                print('Not enough songs:', song_final_genre, target_final_genre, len(lines), df_index.shape, len(lines) / df_index.shape[0])
                continue
            else:
                final_genre_count += Counter(final_genres)
                pbar.set_description(f"[{song_final_genre}] {final_genre_count[target_final_genre]}")
                if final_genre_count[target_final_genre] > num_songs_per_genre:
                    print(final_genre_count)
            with open(output_fp, 'w') as f:
                f.writelines(lines)
            with open(output_url2idx, 'a') as f:
                f.write(f'{tar_fp}\t{str(output_fp)}\n')
    print('Final genre count:', final_genre_count)
    print('Total songs', sum(final_genre_count.values()))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--genres", type=str, default=None, help="input meta path"
    )
    parser.add_argument(
        "--label", type=str, default="B", help="input meta path"
    )
    parser.add_argument(
        "--target_size", type=int, default=1_000_000, help="input meta path"
    )
    args = parser.parse_args()
    run_genres(args.genres, args.label, args.target_size)

# mlx worker launch --gpu 0 -- python3 recipes/bigmusic/dev/ashaw/scripts/process_1m_vocalB.py --target_size 2400000 # balances out to 2 million
# mlx worker launch --gpu 0 -- python3 recipes/bigmusic/dev/ashaw/scripts/process_1m_vocalB.py --target_size 1000000
# mlx worker launch --gpu 0 -- python3 recipes/bigmusic/dev/ashaw/scripts/process_1m_vocalB.py --target_size 500000
# mlx worker launch --gpu 0 -- python3 recipes/bigmusic/dev/ashaw/scripts/process_1m_vocalB.py --target_size 300000
# mlx worker launch --gpu 0 -- python3 recipes/bigmusic/dev/ashaw/scripts/process_1m_vocalB.py --target_size 100000

    # mlx worker launch --gpu 0 -- python3 recipes/bigmusic/dev/ashaw/scripts/process_1m_vocalB.py --target_size 50000
    # mlx worker launch --gpu 0 -- python3 recipes/bigmusic/dev/ashaw/scripts/process_1m_vocalB.py --target_size 25000

# mlx worker launch --gpu 0 -- python3 recipes/bigmusic/dev/ashaw/scripts/process_1m_vocalB.py --target_size 2000000 --label A