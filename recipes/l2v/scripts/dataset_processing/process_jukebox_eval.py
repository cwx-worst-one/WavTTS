# import sys
# sys.path.append("/mnt/bn/ashaw-us/repos/samantha")  # Replace with your local samantha path

import json
import requests
from pathlib import Path
import webdataset
from tqdm import tqdm
import os
from IPython.display import Audio


base_output_dir = Path(f'/mnt/bn/ashaw-us/processed_data/jukebox_soundcloud/')
base_output_dir.mkdir(parents=True, exist_ok=True)

# ==== Download Song list ====
song_json = requests.get('https://jukebox.openai.com/table.json').json()
with open(base_output_dir/'song_table.json', 'w') as f:
    json.dump(song_json, f)

# ==== Download Song Audio and Metadata ====
songs_unseen = [song for song in song_json['table'] if song['collection'] == 'Unseen lyrics']
for song in tqdm(songs_unseen):
    song_id = song['id']
    output_dir = base_output_dir/str(song_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    if (output_dir/'metadata.json').exists():
#         print('Already downloaded', song_id)
        continue
    request_result = requests.get(f'https://jukebox.openai.com/songs/{song_id}/metadata.json')
    song_json = request_result.json()
    if not song_json.get('lyrics'):
        print("could not find lyrics:", song_id, song_json)
        continue

    print('Soundcloud:', song_json['soundcloud_permalink'])
    with open(output_dir/'metadata.json', 'w') as f:
        json.dump(song_json, f)
    
    with open(output_dir/'lyrics.txt', 'w') as f:
        f.writelines(song_json['lyrics'])
        
    soundcloud_url = song_json['soundcloud_permalink']
    
    cmd = f'cd {output_dir} && scdl -l "{soundcloud_url}" --onlymp3 --path "{output_dir}"'
    os.system(cmd)

# ==== Run Force alignment ====

# SA API
def run_vc_query_request(audio_data, request_params):
    base_url = 'https://openspeech.bytedance.com/api/v1/vc'
    appid = "lv"
    token = "lv_token"
    auth_header = { 'Authorization': f'Bearer; {token}' }

        
    request_params = { 'appid': appid, **request_params }
    response = requests.post(
                 f'{base_url}/submit',
                 params=request_params,
                 headers={
                    'content-type': 'audio/m4a',
                     **auth_header
                 },
                 data=audio_data
             )
    if response.status_code != 200:
        print('Error submit response = {}'.format(response.text))
    assert(response.status_code == 200)
    assert(response.json()['message'] == 'Success')

    job_id = response.json()['id']
    response = requests.get(
            f'{base_url}/query',
            params=dict(
                appid=appid,
                id=job_id,
            ),
            headers=auth_header
    )
    assert(response.status_code == 200)
    return response.json()

def run_video_caption(audio_data, endpoint_type='maliva'):
    language = 'en-US'

    request_params = {
        'caption_type': 'speech',
        'language': language,
    }
    return run_vc_query_request(audio_data, request_params, endpoint_type)  

def run_video_caption_alignment(audio_data, audio_text):
    language = 'en-US'
    request_params = {
        'caption_type': 'speech',
        'caption_category': 2,
        'language': language,
        'audio_text': audio_text,
    }
    return run_vc_query_request(audio_data, request_params)

# Transciption
def transcribe_song(song_path, overwrite=True):
    transcription_fp = song_path/'transcription.json'
    try:
        audio_fp = next(song_path.glob('*.mp3'))
    except:
        print('Audio does not exist', song_path)
        return
    if overwrite is False and transcription_fp.exists():
        print('Already transcribed', song_path)
        return
    with open(song_path/'metadata.json', 'r') as f:
        song_json = json.load(f)
        audio_text = song_json['lyrics']
        
    if '4min' in song_json['condition']:
        print('Skipping song. Too long', song_json)
        return

    with open(audio_fp, 'rb') as f:
        audio_data = f.read()

    transcription_result = run_video_caption_alignment(audio_data, audio_text)
    lyrics_txt = ' '.join([x['text'] for x in transcription_result['utterances']])
    
    if 'utterances' not in transcription_result:
        print('Error with transcription:', transcription_result)

    with open(transcription_fp, 'w') as f:
        json.dump(transcription_result, f)
        
    return lyrics_txt, audio_data


song_paths = [d for d in Path(f'/mnt/bn/ashaw-us/processed_data/jukebox_soundcloud/').iterdir() if d.is_dir()]
for song_path in song_paths:
    transcribe_song(song_path)


# # ==== Create Tar Files ====

tar_name = 'jukebox_eval.tar'
output_tar_path = base_output_dir/tar_name
output_tar_path.parent.mkdir(exist_ok=True, parents=True)
sink = webdataset.TarWriter(str(output_tar_path))

lyrics_index_output_fp = base_output_dir/f'{tar_name}.index'
lyrics_writer = open(lyrics_index_output_fp, 'w')


for index, song_path in tqdm(enumerate(song_paths)):
    song_id = song_path.name
    try:
        audio_fp = next(song_path.glob('*.mp3'))
    except:
        print('Audio does not exist', song_path)
        continue
    metadata_fp = song_path/'metadata.json'
    transcription_fp = song_path/'transcription.json'
    
    if not transcription_fp.exists():
        print('Transcription does not exist:', song_path)
        continue
    
    with open(metadata_fp, 'r') as f:
        song_json = json.load(f)
        
    with open(transcription_fp, 'r') as f:
        transcription_json = json.load(f)
        
    with open(audio_fp, 'rb') as f:
        audio_data = f.read()

    sink.write({
        "__key__": song_id,
        'mp3': audio_data,
    })

    lyrics_str = json.dumps({ 'lyrics': transcription_json, 'metadata': song_json })
    lyrics_writer.write(f'{song_id}\t{lyrics_str}\n')

print('Tarred files to output url:', index, output_tar_path, lyrics_index_output_fp)
sink.close()
lyrics_writer.close()

# ==== EXAMPLE: Create Dataloader ====
from recipes.l2v.datasets.lyrics import LyricsDataset
from recipes.l2v.datasets.tokenizers.phoneme_tokenizer import PhonemeTokenizer
from recipes.l2v.datasets.transforms.lyrics import LyricsTokenTransform
import IPython.display as ipd


return_phoneme_tokens = True
def lyrics_check(item):
    if item is None: return None
    lyrics = item['lyrics']
    words = lyrics.split(' ')
    if len(words) < 5: 
        print(words)
        return None # filter out low word count
    return item

if return_phoneme_tokens:# To return phoneme tokens:
    lyrics_tokenizer = PhonemeTokenizer(allow_unknown=True)
    segment_transforms = [lyrics_check, LyricsTokenTransform(lyrics_tokenizer, 150)]
else:
    segment_transforms = []
    
audio_keys = { 'target_audio': 'mp3', 'mulan_audio': 'mp3' }
url2index = { '/mnt/bn/ashaw-us/processed_data/jukebox_soundcloud/jukebox_eval.tar': '/mnt/bn/ashaw-us/processed_data/jukebox_soundcloud/jukebox_eval.tar.index'}
        
dataset = LyricsDataset(
    url2index=url2index,
    sample_rate=24000,
    sample_duration=10.0,
    segment_transforms=segment_transforms,
    audio_keys=audio_keys,
    max_num_segments=1,
    use_pipe=True
)
it = iter(dataset)


item = next(it)
print(item)
ipd.Audio(item["target_audio"], rate=24000)


# from samantha.dataio.webdataset.extension import IndexedWebDataset
# dataset = IndexedWebDataset(
#     url2index=url2index,
# )
# it = iter(dataset)
# item = next(it)
# item['__index_data__']
# Audio(item['mp3'])
