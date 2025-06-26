import soundfile as sf
import webdataset
import subprocess, os
from tqdm import tqdm

from midi_utils import process_transcription_result, note2midi, chord_to_midi
from concat_vocal2midi_data import pop909

dump_dir = 'pop909_dump'
os.makedirs(dump_dir, exist_ok=True)

urls = pop909('train')
dataset = webdataset.WebDataset(urls).decode()

for item in tqdm(dataset):
    fn = item['__key__'].replace('pop909_', '')
    audio = item['audio.npy']
    midi_notes = item['note_seq.pickle']
    onset_seq = item['onset_seq.pickle']

    sf.write(os.path.join(dump_dir, fn + '.wav'), audio, 24000)
    subprocess.call(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', os.path.join(dump_dir, fn + '.wav'), '-b:a', '128k', os.path.join(dump_dir, fn + '.mp3')])
    os.remove(os.path.join(dump_dir, fn + '.wav'))
    note2midi(midi_notes, midi_notes[-1]['end'], os.path.join(dump_dir, fn + '.mid'))
