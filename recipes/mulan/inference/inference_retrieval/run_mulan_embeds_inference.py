import io
import os
import glob
import numpy as np
import pickle
import torch
import torch
import subprocess
import soundfile as sf
from typing import Tuple
from pathlib import Path
from tqdm import tqdm
from typing import List
import subprocess
from scipy.io.wavfile import write
import json
import os
import random

from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import time
import webdataset as wds
from recipes.mulan.dataset.utils import *
from samantha.dataio.webdataset.extension import IndexedWebDataset
import json
import librosa
import io
from samantha.dataio.parquet.parquet_dataset import ParquetDataset
# Load tokenizer
from transformers import AutoTokenizer

device = "cuda:0"

# Load mulan model
from mulan_modules import create_mulan_model


## TODO: replace with your own mulan model checkpoint
# ckpt_path = "/mnt/bn/audio-diffusion/mulan/ongoing/mulan-step=014000-median_rank_1=160-kaggle.ckpt"
# ckpt_path = "/mnt/bn/mm-data/user/xuchen.song/mulan_tag/mulan-step=024600-kaggle.ckpt"
# prepare the following ckpt_path using these two commands:
# cd /opt/tiger/arnold_starter/
# hdfs dfs get hdfs://haruna/home/byte_speech_sv/mulan/experiments/MuLan_large/qqmusic_1115_callbacks/checkpoints/mulan-step=012800-kaggle.ckpt
ckpt_path = "/mnt/bn/audio-diffusion/xuchen/mulan/models/mulan-step=014800-kaggle.ckpt"
mulan_module = create_mulan_model(ckpt_path, device=device).eval()


## TODO: replace with your own audio folder (input) and audio embed folder (output)
## Assume one audio file is either one npy file or one wav file
# audio_folder = "/mnt/bn/mm-data/user/xuchen.song/mulan_tag/audio_sample"
audio_folder = "/opt/tiger/samantha/audio_samples"
os.makedirs(audio_folder, exist_ok=True)

audio_embed_folder = audio_folder.replace("/audio_sample", "/audio_embeds_mulan_30ks")
text_embed_folder = audio_folder.replace("/audio_sample", "/text_embeds_mulan")
os.makedirs(audio_embed_folder, exist_ok=True)
os.makedirs(text_embed_folder, exist_ok=True)

def pattern_apply(text_content):
    try:
        pattern = r'"text":\s*"([^"]+)"'
        match_res = re.search(pattern, text_content)
        if match_res:
            extracted_text = match_res.group(1)
        else:
            return ""
    except:
        print("wrong text!")
        return ""
    return extracted_text
def extract_segments(wav, sr=24000, duration: int = 10, stride: int = 5):
    """Extract audio segments of `duration` seconds every `stride` seconds."""
    wavs = torch.tensor(wav).unfold(1, sr * duration, sr * stride)[
        0, :, : sr * duration
    ]
    return wavs
def process_audio_pq(data, segment = None):
    audio, _ = librosa.load(io.BytesIO(data["wav"]), sr = None)
    data["audio.npy"] = audio
    del data["wav"]
    data = process_audio(data, segment)
    return data
class SSTKDataset(IterableDataset):
    def __init__(self, name="sstk", mode="train",**kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.name = name
        dataset_id = 105
        self.dataset = (
            ParquetDataset(dataset_id, handler=wds.warn_and_continue, **kwargs)
            .map(self._process_audio)
            .map(self._process_text)
            .map(tokenize_text(self.tokenizer, mode))
        )
    def _process_audio(self, data):
        metadata = json.loads(data["meta"])

        #"vad": {"segment": [{"start": 1840, "end": 4210}, {"start": 48720, "end": 49700}, {"start": 53050, "end": 54850}, {"start": 86020, "end": 88240}, {"start": 88530, "end": 89540}], 
        # "extra": {"voice_duration_in_seconds": 8.38, "audio_duration_in_seconds": 154.10526, "voice_proportion": 0.054378}} 
        segment = metadata.get("vad", {}).get("segment", None)
        #print("segment in _process_audio", segment)
        return process_audio_pq(data, segment)


    def _process_text(self, data):
        metadata = json.loads(data["meta"])
        if self.name == "sstk":
            title = metadata.get('title', "")
            description = metadata.get('description', "")
            keywords = metadata.get('keywords', "")
            genres = metadata.get('genres', "")
            instruments = metadata.get('instruments', "")
            text_fields = [title, description, keywords, genres, instruments]
            data["text"] = ". ".join([t for t in text_fields if t])
            data["music_id"] = fix_hash(data["__key__"])
            return data  
        else:
            return data

    def __iter__(self):
        return iter(self.dataset)

# cnt = 0
# dataset = CMDatasetPar(name="qq", mode="test")
# for item in dataset:
#     # print(item.keys())
#     # print(item["audio"])
#     # print(item["text"])
#     audio_id = item["uttid"]
#     audio_wav = item["audio"]
#     #print("audio_wav", audio_wav.shape)
#     #audio_load torch.Size([1, 240000])
#     audio_segments = extract_segments(
#             audio_wav.reshape([1, -1]), sr=24000, duration=10, stride=10
#         ).to(
#             device
#         )  # [M, sr*duration]

#         # 4. Call mulan model to calculate audio embeds
#     with torch.no_grad():
#         music_emb = (
#             mulan_module.music_encoder(audio_segments.unsqueeze(1))
#             .cpu()
#             .data.numpy()
#         )
#     print("music_emb", music_emb.shape)
#     np.save(f"{audio_embed_folder}/{audio_id}.mulan_emb.npy", music_emb)

    #break
    #breakpoint()

def infer_audio_embeds():
    """Assume the audios are stored in npy or wav format."""
    cnt = 0
    dataset = SSTKDataset(name="sstk", mode="test")
    for item in dataset:
        # print(item.keys())
        # print(item["audio"])
        # print(item["text"])
        audio_wav = item["audio"]
        audio_id = item["uttid"]
        audio_npy =  audio_wav.numpy()[0, :]
        write(audio_folder +"/"+ audio_id+".wav", 24000, audio_npy)
        # assert (
        #     audio_wav.ndim == 1
        # ), """The audios are expected to be mono wave of shape [T]"""

        # 3. split the complete audio into a minibatch of segments
        audio_segments = extract_segments(
            audio_wav.reshape([1, -1]), sr=24000, duration=10, stride=10
        ).to(
            device
        )  # [M, sr*duration]

        # 4. Call mulan model to calculate audio embeds
        with torch.no_grad():
            music_emb = (
                mulan_module.music_encoder(audio_segments.unsqueeze(1))
                .cpu()
                .data.numpy()
            )
        # print("music_emb", music_emb.shape)
        np.save(f"{audio_embed_folder}/{audio_id}.mulan_emb.npy", music_emb)
        cnt += 1
        if cnt == 10000:
            break


print("\nInferring the audio embeds...\n")
infer_audio_embeds()


""""Hard coded method, the list of text pool"""


def prepare_texts():
    text_pool_1 = [
        "EDM Techno",
        "EDM House",
        "EDM Disco",
        "Dubstep",
        "Electronic Ambient",
        "8 Bit / Chiptune",
        "Country Bluegrass",
        "Hiphop Trap",
        "jazz hip hop",
        "K-Pop",
        "Dance Pop",
        "Easy listening",
        "Chinese pop",
        "Metal",
        "Jazz Swing",
        "Big Band Jazz",
        "Bossa Nova",
        "RnB Funk",
        "Classical Music",
        "Latin Pop",
        "Chinese Opera",
        "EDM House: Groovy rhythms",
        "Let's create some EDM House vibes",
        "Disco style",
        "heavy bass drops and electronic twists as Dubstep",
        "Seeking soothing electronic soundscapes and ambient vibes",
        "Chiptune sound",
        "Country Bluegrass",
        "Rhythmic Trap beats with an urban edge.",
        "Blend smooth grooves and rap in Jazz Hip Hop fashion.",
        "catchy melodies pop with a dynamic K-Pop touch",
        "Feel the groove of Dance Pop - energetic rhythms and infectious hooks",
        "Chill with some Easy Listening",
        "Modern beats meet cultural resonance in Chinese pop tracks",
        "Metal style music",
        "Jazz Swing invites you to groove with upbeat rhythms",
        "Lively tunes from big band jazz",
        "Bossa Nova vibes with Brazilian charm and rhythmic tunes",
        "Groovy beats and soulful melodies, like RnB or Funk style.",
        "Classical Music with orchestration",
        "Spice it up with Latin Pop",
        "traditional Chinese Opera",
        "Happy",
        "Chill/Calm/Relaxing",
        "Excited",
        "Funny",
        "Dreamy",
        "Sorrow/Sad",
        "Thrilling/Suspenseful/Tense",
        "Shocking/magnificent",
        "Angry/Aggressive",
        "Dynamic/Energetic",
        "Dramatic",
        "Romantic",
        "Feel the happy vibes",
        "Unwind with chill, calm, relaxing tracks that is perfect for mellow moments",
        "energetic and exciting beats",
        "Have a laugh with funny tunes - playful melodies that bring a smile.",
        "Get lost in dreamy sounds",
        "Sorrow and sad tunes",
        "Thrilling, suspenseful, tense music",
        "Shocking/magnificent",
        "music for a movie scene where the anger is boiling",
        "Give me some dynamic and energetic rhythms",
        "Feel the emotions with Dramatic music",
        "Feelin' the love? Give me some Romantic melodies.",
        "flute",
        "clarinet",
        "saxophone",
        "trumpet",
        "drum",
        "piano",
        "violin",
        "cello",
        "bass",
        "acoustic guitar",
        "electric guitar",
        "synthesizer",
        "marimba",
        "accordian",
        "soothing whispers of the flute",
        "A clarinet piece",
        "saxophone virtuoso",
        "vibrant notes of the trumpet",
        "Let the drum beats set the rhythm",
        "piano concerto",
        "violin duet",
        "Get swept away by the deep, resonant tones of the cello.",
        "strong pedal bass",
        "acoustic guitar strumming",
        "electric guitar masterpiece",
        "Futuristic sounds of the synthesizer",
        "playful tones of the marimba",
        "An accordian masterpiece",
        "Halloween",
        "Christmas",
        "New Year",
        "Valentine's day",
        "Birthday",
        "Wedding",
        "Funeral",
        "Graduation",
        "Spooky fun with costumes and treats at Halloween.",
        "Christmas music suitable for family gathering",
        "Countdown to the New Year - celebrate with fireworks, friends, and fresh starts.",
        "romantic dinner on Valentine's day",
        "Birthday party that celebrate with cake and good times",
        "Love is in the air at weddings",
        "Pay respects and say goodbye at funerals",
        "Graduation ceremony",
        "Rainy Day",
        "Sunny Day",
        "Spring",
        "Summer",
        "Autumn",
        "Winter",
        "Evening",
        "Morning",
        # "Perfect for rainy moments",
        # "energetic music for a sunny day hike",
        # "spring sunshine",
        # "Feel the sun's warmth in summer",
        # "Autumn's cozy ambiance.",
        # "Songs to keep you warm this chilly season",
        # "Music for nighttime",
        # "Start your day right",
        # "Vlog/DailyLife",
        # "Food",
        # "Pet",
        # "Beauty/Fashion",
        # "Entertainment",
        # "babies",
        # "children",
        # "Anime",
        # "study/work",
        # "Wake up",
        # "Family time",
        # "landscape/scenery",
        # "Prank",
        # "Timelapse",
        # "Music for everyday moments",
        # "Delicious food",
        # "lovely cat",
        # "Generate a music for beauty/fashion videos",
        # "Fun time",
        # "Lullaby for babies",
        # "Kids' world",
        # "attack on titan music",
        # "Give me a music to help with focus during study or work",
        # "Good morning music",
        # "Songs for family reunion",
        # "landscape/scenery",
        # "music for prank",
        # "music for timelapse videos",
        # "Beach",
        # "Nightclub",
        # "Coffee Shop",
        # "Restaurants",
        # "Lounge",
        # "Campus",
        # "Park",
        # "Marketplace",
        # "Rainforest",
        # "Train station",
        # "Mountain",
        # "chilling on the beach while sipping margaritas",
        # "80s nightclub with an elegant vibe",
        # "Music played at Starbucks",
        # "Music played at a local restaurant",
        # "Lounge chillout music",
        # "Campus Vibes: Study Jams and Chill Beats",
        # "Stroll in the park",
        # "Music played in the marketplace",
        # "Sound from the rainforests",
        # "music played at busy train station",
        # "Mountain hikes",
        # "Sport",
        # "Dance",
        # "Game",
        # "Travel",
        # "Roadtrip",
        # "Meditation",
        # "Running",
        # "Football games",
        # "breakdancing",
        # "Music for games",
        # "Songs for the journey",
        # "Music I would listen during Roadtrip",
        # "Music for meditation",
        # "Play me some music when I am running",
        # "RnB Funk, Sorrow/Sad, Winter",
        # "EDM Techno, Shocking/magnificent",
        # "Sorrow/Sad, Coffee Shop",
        # "EDM House, Excited, Valentine's day",
        # "Bossa Nova, Funny, bass, Morning",
        # "Chinese Opera, Dramatic",
        # "Chill/Calm/Relaxing, Graduation",
        # "Dance Pop, Happy, Timelapse",
        # "Funny, Halloween",
        # "Happy, Campus",
        # "8 Bit / Chiptune, Dynamic/Energetic, Prank",
        # "K-Pop, guitar strumming, Coffee Shop",
        # "Hiphop Trap, Beach",
        # "Dreamy, violin, Meditation",
        # "Big Band Jazz, Dance",
        # "drum, Timelapse",
        # "Bossa Nova, piano, landscape/scenery",
        # "Funny, Dance",
        # "Easy listening, Game",
        # "Classical Music, babies",
        # "jazz hip hop, Romantic, marimba, ",
        # "Dreamy, strings and drums, ",
        # "K-Pop, Chill/Calm/Relaxing, Wake up",
        # "Pop, Dramatic, brass ensemble",
        # "Chinese pop, Funny, Running",
        # "morning, Chill/Calm/Relaxing, Autumn",
        # "RnB Soul, nightime, Mountain",
        # "Easy listening, Sorrow/Sad, clarinet and oboe",
        # "starlight, retrospect, Campus",
        # "trumpet, Lounge",
        # "Funny, saxophone, Anime",
        # "guitar, Valentine's day",
        # "ambient electronic, Universe",
        # "Chill/Calm/Relaxing, Vlog/DailyLife",
        # "Thrilling/Suspenseful/Tense, Prank",
        # "Chinese pop, Dynamic/Energetic, Nightclub",
        # "Easy listening, Romantic, ",
        # "sentiment, Autumn",
        # "Dynamic/Energetic, Roadtrip",
        # "Romantic, Family time",
        # "Chill/Calm/Relaxing, Funeral",
        # "Electronic Ambient, Beauty/Fashion",
        # "Shocking/magnificent, Dance",
        # "Smooth Jazz, saxophone, study/work",
        # "Funny, pets, Beach",
        # "Smooth Jazz, Dreamy, Wedding",
        # "Bossa Nova, piano, landscape",
        # "Chinese pop, piano, Beauty/Fashion",
        # "Disco, Shocking/magnificent",
        # "RnB Funk, Funny, piano, Prank",
        # "Bossa Nova, Chill/Calm/Relaxing, Food",
        # "piano, synthesizer, Entertainment",
        # "morning, strings, Sunday",
        # "Funny, animals",
        # "Bossa Nova, Beauty/Fashion",
        # "flute, piano, Vlog/DailyLife",
        # "Electronic Ambient, Dreamy, Wedding",
        # "Country Bluegrass, Train station",
        # "Funny, accordian, Travel",
        # "Chill/Calm/Relaxing, Christmas",
        # "Cute, babies",
        # "Easy listening, forest",
        # "Funny, synthesizer, Prank",
        # "jazz hip hop, upbeat, Morning",
        # "Dynamic/Energetic, Restaurants",
        # "Easy listening, Romantic, strings, Mountain",
        # "Dance Pop, brass, Prank",
        # "Easy listening, Rainy Day",
        # "Latin Pop, Chill/Calm/Relaxing, Sunny Day",
        # "K-Pop, piano",
        # "Dramatic, orchestra, Mountain",
        # "Chinese pop, Excited, Entertainment",
        # "Dramatic, Funeral",
        # "Dreamy, oboe, Restaurants",
        # "piano, guitar, Camping",
        # "Dance Pop, Dramatic, Dance",
        # "jazz hip hop, Angry/Aggressive, Roadtrip",
        # "K-Pop, dance",
        # "jazz, guitar, bass",
        # "HipHop trap, bass, bell",
        # "funky, strings ",
        # "Chinese tradition, Erhu, Pipa",
        # "Jazz, Piano",
        # "Thrilling/Suspenseful/Tense, sky, violin, drum",
        # "reggaeton, energetic, trumpet, nightclub",
        # "electronic music, party",
        # "pop, Vlog/DailyLife",
        # "happy, energetic",
        # "Classical music, piano",
        # "Jazz, Relaxing, piano",
        # "Reggaeton, Electronic ambient, dance, universe",
        # "Classical, folk, orchestrations, nature",
        # "meditation, calm, flute, guitar",
        # "violin, piano",
        # "marimba, guitar",
        # "electronic piano, organ",
        # "Create a reflective RnB Funk track that captures the melancholic essence of winter.",
        # "Develop an awe-inspiring EDM Techno composition that infuses elements of shock and magnificence.",
        # "Create a heartfelt composition that evokes emotions of sadness, suitable for a contemplative coffee shop ambiance.",
        # "Develop an exhilarating EDM House composition that captures the excitement of Valentine's Day.",
        # "Create a playful Bossa Nova track with a fun bassline, perfect for a sunny morning vibe.",
        # "Craft a compelling composition that embraces the dramatic spirit of Chinese opera.",
        # "Develop a soothing piece, capturing a sense of calm and tranquility that fits well for graduation.",
        # "Produce a lively Dance Pop tune that radiates happiness, perfect for capturing the essence of a timelapse.",
        # "Create a whimsical track, perfect for playful Halloween festivities.",
        # "a joyful tune that reflects the lively campus atmosphere.",
        # "Make a fun 8 Bit / Chiptune track with dynamic energy, perfect for a playful prank vibe.",
        # "an enjoyable K-Pop melody with rhythmic guitar, great for a cozy coffee shop mood.",
        # "Make a Hiphop Trap track that adds excitement to a beach scene",
        # "Make a calming tune with a dreamy violin for a chill meditation vibe.",
        # "Imagine the vibrant energy of a Big Band Jazz tune seamlessly blending with the rhythm, inviting you to dance along.",
        # "Imagine the rhythmic charm of hihat percussion enhancing the beauty of a captivating timelapse.",
        # "Picture a serene Bossa Nova composition, where the gentle notes of the piano paint a musical portrait of a tranquil and beautiful landscape.",
        # "Craft a lighthearted track perfect for dancing, exuding a sense of fun and humor.",
        # "Create an easy listening melody that sets a relaxed tone, making it ideal for background music during a game.",
        # "Design a piece of classical music that's soothing and harmonious, suitable for creating a calming atmosphere for babies.",
        # "Blend the rhythms of jazz and hip hop into a romantic melody, enriched by the enchanting sound of marimba.",
        # "a dreamy composition by combining the ethereal qualities of strings with the rhythmic heartbeat of drums.",
        # "Design a chill K-Pop melody that exudes a calming and relaxing ambiance, perfect for starting your day with a refreshing wake-up vibe.",
        # "Compose a dramatic pop arrangement featuring the grandeur of a brass ensemble",
        # "Blend the catchy elements of Chinese pop with a touch of humor, setting an energetic and playful tone that's perfect for a running soundtrack.",
        # "Experience a tranquil autumn morning, accompanied by calming and relaxing melodies.",
        # "Indulge in the soulful vibes of RnB Soul during nighttime amidst the mountains.",
        # "Ease into a serene atmosphere with easy listening tunes featuring clarinet and oboe.",
        # "Get lost in the magic of starlit nights, pondering fond memories on campus.",
        # "Let the trumpet's melodies take you to a relaxing lounge filled with a nostalgic charm.",
        # "Unwind to the playful saxophone tunes that transport you to an animated world.",
        # "Feel the rhythm of Dance Pop with a lively guitar, perfect for celebrating Valentine's Day.",
        # "Immerse yourself in ambient electronic sounds that evoke the vastness of the universe.",
        # "Chill and relax with captivating melodies, capturing the essence of daily vlogs.",
        # "Brace yourself for thrilling suspense with music that sets the tone for pranks.",
        # "Experience the vibrant and energetic essence of Chinese pop, perfectly suited for the electrifying atmosphere of a nightclub.",
        # "Set a romantic ambiance with easy listening tunes that speak to the heart",
        # "Embrace the sentiment of the season with autumnal melodies, reflecting the changing beauty of nature.",
        # "Feel the dynamic energy of music that sets the tone for an exciting road trip adventure.",
        # "Create a romantic atmosphere that captures the essence of cherished family time.",
        # "Ease into a calming melody, perfect for adding a sense of serenity to funeral proceedings.",
        # "Immerse in the electronic ambiance that mirrors the world of beauty and fashion.",
        # "Experience the shocking and magnificent elements of music, perfect for expressing dance through sound.",
        # "Let the soulful sound of a saxophone guide your focus during study and work sessions.",
        # "Infuse humor and lightness with melodies that capture the playful nature of pets at the beach.",
        # "Let Smooth Jazz melodies create a dreamy backdrop for the warmth of a wedding celebration.",
        # "Enjoy the easygoing rhythms of Bossa Nova paired with a groovy piano, reminiscent of the calming ocean.",
        # "Delight in the melodies of Chinese pop, featuring piano movements that resonate with beauty and fashion themes.",
        # "Let the DJ's beats take you by surprise with shocking and magnificent sounds that ignite the atmosphere.",
        # "Have a laugh with the playful RnB Funk, where untuned piano notes add humor and a touch of prank.",
        # "Soak in the relaxed vibes of Bossa Nova, creating a perfect backdrop for enjoying food.",
        # "Immerse in the world of entertainment with synthesizers and electronic pianos adding captivating allure.",
        # "Elevate the mood of a peaceful Sunday morning with the soothing melodies of a string quartet.",
        # "Indulge in the lightheartedness of funny melodies that capture the essence of playful animals.",
        # "Bossa Nova evokes a fashionable vibe.",
        # "Flute and piano suit daily vlogs.",
        # "Dreamy Electronic Ambient sets a wedding mood.",
        # "Country Bluegrass captures train station scenes.",
        # "Funny accordion tunes bring travel vibes.",
        # "Chill/Calm/Relaxing melodies for Christmas eve.",
        # "Cute melodies celebrate babies.",
        # "Easy listening conjures rainforest ambiance.",
        # "Funny synthesizer tones add a playful prank touch.",
        # "Jazz hip hop brings upbeat energy to mornings.",
        # "Feel the dynamic vibe of jazz hip hop at restaurants.",
        # "Easy listening turns romantic with soothing strings in the mountains.",
        # "Dance Pop comes alive with a vibrant brass ensemble, adding a hint of prank.",
        # "Easy listening suits rainy days with its calming melodies.",
        # "Latin Pop offers a chill vibe on sunny days.",
        # "K-Pop resonates through the melodies of a piano.",
        # "Dramatic orchestra notes enhance mountain scenes.",
        # "Feel the excitement of Chinese pop in the world of entertainment.",
        # "Feel the drama in music fitting for a funeral.",
        # "Dreamy melodies of the oboe create a restaurant ambiance.",
        # "Piano and guitar melodies resonate with camping experiences.",
        # "Dance Pop's excitement captures the thrill of the forest.",
        # "scary music in a movie where the ghost is about to appear",
        # "kpop girl group music that is catchy and makes me want to jump up and down",
        # "a fusion of strong beats and bass with jazz guitar",
        # "Hiphop trap with strong bass and bell appregios",
        # "I want a funky music played by a string quartet.",
        # "I want a music played by traditional Chinese instruments, including Erhu and Pipa.",
        # "Music in jazz style played with piano. No drum, not saxophone, no bass.",
        # "horror and suspenseful music featuring the sound of a violin and loud drum. the sky is dark and misty and werewolves are about to come out and hunt for prey ",
        # "reggaeton beats with epic trumpet line that is clearly heard through the mix. the music should be energetic, unique and sound like music you would hear in a brazilian nightclub. makes me wanna party with my besties",
        # "awesome electronic music i would listen to at tomorrowland",
        # "smooth pop music suitable for vlog",
        # "Upbeat and lively, featuring quick, cheerful melodies and rhythms that are designed to be energetic and engaging.",
        # "A beautiful and expertly crafted piano composition that is sure to resonate with fans of contemporary classical music.",
        # "The jazz influence can be heard in the delicate and smooth rhythm of the piano notes, accompanied by a subtle and relaxing background ambiance.",
        # "A fusion of reggaeton and electronic dance music, with a spacey, otherworldly sound. Induces the experience of being lost in space, and the music would be designed to evoke a sense of wonder and awe, while being danceable.",
        # "A blend of classical orchestral music and traditional folk tunes, with a grounded, earthy sound. Evokes the feeling of being rooted in nature, and the music is crafted to elicit a sense of tranquility and introspection, while remaining rhythmic.",
        # "Meditative song, calming and soothing, with flutes and guitars. The music is slow, with a focus on creating a sense of peace and tranquility.",
        # "A violin melody accompanied by piano arrangements.",
        # "Marimba melody accompaniment by guitar strumming",
        # "Sound of an instrument like electronic piano or organ, which I cannot tell.",
        # "Thrilling/Suspenseful/Tense, piano, Beauty/Fashion",
        # "8 Bit / Chiptune, Funny, violin, piano",
        # "Chinese Opera, violin, ",
        # "Dubstep, Shocking/magnificent, brass ensemble, Park",
        # "Easy listening, Dynamic/Energetic, Anime",
        # "Bossa Nova, brass, Christmas",
        # "Sorrow/Sad, drum, bass, Food",
        # "Latin Pop, Thrilling/Suspenseful/Tense, children",
        # "EDM Disco, organ solo, Meditation",
        # "EDM House, flute and marimba, ",
        # "Angry/Aggressive, trumpet, Meditation",
        # "Funny, drum, bass, Train station",
        # "EDM House, flute, Marketplace",
        # "Metal, cello, ",
        # "Excited, acoustic guitar, Funeral",
        # "Dubstep, Christmas",
        # "Angry/Aggressive, accordian",
        # "Thrilling/Suspenseful/Tense, Pet",
        # "Bossa Nova, violin, Dance",
        # "Sorrow/Sad, bass, Valentine's day",
        # "guitar, Excited, violin, Anime",
        # "classical, rock, ",
        # "accordian, Inspirational/Hopeful",
        # "Thrilling piano concerto resonates with beauty and fashion themes.",
        # "8 Bit / Chiptune pairs humor with a violin-piano duet.",
        # "Chinese Opera melodies brought to life by the violin.",
        # "Dubstep's shocking and magnificent notes blend with a brass ensemble at the park.",
        # "Easy listening takes on a dynamic and energetic vibe for anime enthusiasts.",
        # "Bossa Nova's brass ensemble celebrates Christmas with its melodies.",
        # "Sorrowful Bossa Nova notes resonate alongside drum kicks and pedal bass for a food-themed atmosphere.",
        # "Latin Pop brings a thrilling tone to scenes involving children.",
        # "EDM Disco features an organ solo for moments of meditation.",
        # "EDM House melodies, highlighted by flute and marimba.",
        # "Feel the anger through trumpet notes during meditation in EDM House.",
        # "Funny tunes feature drum kicks and pedal bass at a train station.",
        # "Flute notes bring vibrancy to the marketplace in EDM House.",
        # "Metal notes resonate with the rich sounds of the cello.",
        # "Excited vibes with the strum of an acoustic guitar at a funeral.",
        # "Dubstep creates a unique Christmas ambiance.",
        # "Accordian notes bring an angry and aggressive twist.",
        # "Thrilling suspense pairs with a pet-themed atmosphere.",
        # "Bossa Nova's violin notes invite you to dance.",
        # "Sorrowful Bossa Nova melodies embrace the bass for Valentine's day.",
        # "Exciting violin solo accompanied by guitar strumming captures the anime spirit.",
        # "psychadelic rock mixed with complex bach music",
        # "With the use of polka rhythms and lively accordion melodies, the song tells the story of a proud and beautiful black rooster that catches the attention of all who hear him crow.",
        # "Funky piece with a strong, danceable beat and a prominent walking bassline. The melody from a keyboard has many repeated notes that brings a strong motif.",
        # "The main soundtrack of an arcade game. It is fast-paced and upbeat, with a catchy electric guitar riff. The music is repetitive and easy to remember, but with unexpected sounds, like cymbal crashes or drum rolls.",
        # "The accompaniment layer employs staccato articulations, contributing to its rhythmic momentum. The melody alternates between soaring legato phrases and punctuated staccato notes, creating a dynamic contrast. There are trap-infused beats featuring syncopated rhythms and layered hi-hats. ",
        # "A rising synth is playing an arpeggio with a lot of reverb. It is backed by pads, sub bass line and soft drums. This song is full of synth sounds creating a soothing and adventurous atmosphere.",
        # "Industrial techno sounds, repetitive, hypnotic rhythms. The music is hypnotic and trance-like, and it is easy to get lost in the rhythm. The strings high-pitched notes pierce through the darkness, adding a layer of tension and suspense.",
        # "A hyped rap beat featuring heavy 808 bass. The instrumentation should be sparse, with memorable sound design, use bell sounds as a lead and comical sfx in the background to add memorable points.",
        # "A pop beat that uses marimba as the main harmonic instrument, have simple harmonic sequence, use acoustic drums at a relatively slower tempo. The feel of the music should be joyful, hopeful and it should fit well at a party.",
        # "Martikainen's expert use of strings, horns, drums, and woodwinds. The strings have sweeping arpeggios, and the woodwinds have delicate trills that add delicate nuances. ",
        # "Create a heavy progressive rock track that features Metal and Jazz genre elements. The lead instrument should be a heavy electric guitar incorporating virtuosic passages, melody runs and some cool catchy improvisation. Please also include complex harmony in the accompaniment. ",
        # "A hiphop trap with 808 bass, distinctive bell melodies, and well-placed comical sound effects. Harmonies should complement the melodies, while keeping it simple chord progression.",
        # "The track is characterized by lively chord arpeggios that create a pulsating rhythm with upbeat tempo. The melody takes center stage with notes that feature large pitch jumps, adding a sense of dynamic movement. There are also guitar strumming patterns that provide texture and depth, enhancing the overall energy of the composition.",
        # "Dynamic beat with punchy kick and snappy snare, funky bassline with playful slide. Swing-infused keyboard melody with trills. Expressive horns, well-balanced mix captures urban energy.",
        # "The saxophone takes the lead with a flurry of intricate lines that challenge your ear and keep you engaged. Responding to the sax, the guitar offers a range of tones and techniques, from gentle melodic phrases to more aggressive and angular expressions. The rhythm section has drums and bass creating complex patterns that dance around each other. ",
        # "Rich harmonic progressions drive the rhythm, infusing the composition with a pulsating energy that resonates throughout. The melody shines with expressive vibrato and ornamented grace notes, adding layers of emotion and depth to the musical narrative.",
        # "A propulsive bassline sets the rhythmic foundation, interlocking with syncopated percussion patterns to create an immersive and dynamic groove. The melody features cascading scales and rapid runs, enhancing the track's vivacious and spirited atmosphere.",
        # "The guitar's nimble fingerpicking dances gracefully, intertwining with the resonant notes of the cello. The gentle plucks and legato bowing create an intimate duet, evoking a sense of heartfelt conversation between the instruments.",
        # "The accordion's wheezing notes add a touch of nostalgia, harmonizing with the warm resonance of the acoustic guitar. More specifically, the accordion has quick, running notes that move fast, while the guitar's strumming feels gentle and comfortable. ",
        # "Luminous notes illuminating the darkness.",
        # "A river of melodies flowing through the soul.",
        # "Wolf in the moonlight",
        # "welcome to china",
        # "welcome to indonesia",
        # "music for a scene where the hero is about to be defeated by the evil lord",
        # "a wolf playing a yellow fender telecaster in an office for his bunny friends. disney animation",
        # "feel mesmerized after watching a reverse ending movie",
        # "music that will make me go viral and famous on tiktok",
        # "The rhythmic pulse of life itself.",
        # "Craft a sonic metamorphosis where melodies evolve like butterflies emerging from a cocoon.",
        # "Capture the essence of a thunderstorm where music crackles with electrifying energy.",
        # "Craft a musical time capsule preserving the emotions and experiences of the present for future listeners.",
        # "Melting boundaries between reality and imagination.",
        # "Conjure a symphony of colors where each note resonates with a unique hue and shade.",
        # "Echoes of forgotten languages spoken by instruments.",
        # "Whispers of forgotten dreams and memories.",
        # "Haunting echoes from the depths of the soul.",
        # "Stars shimmer.",
        # "Birds sing.",
        # "Snowflakes melt.",
        # "Fireflies flicker.",
        # "the meaning of life",
        # "a moment of brilliance",
        # "crumbling under pressure",
    ]
    return [text_pool_1]


def infer_text_embeds(
    mulan_module,
    text_dset: List[str],
    tokenizer: AutoTokenizer = None,
    batch_size=128,
    max_length=200,
    device="cuda:0",
):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

    embeds = []
    text_dset = [s.lower() for s in text_dset]

    with torch.no_grad():
        for xl in tqdm(range(0, len(text_dset), batch_size)):
            xr = min(len(text_dset), xl + batch_size)

            data = {}
            encodings = tokenizer(
                text_dset[xl:xr],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            data["input_ids"] = encodings["input_ids"].to(device)
            data["token_type_ids"] = encodings["token_type_ids"].to(device)
            data["attention_mask"] = encodings["attention_mask"].to(device)

            embeds.append(mulan_module.text_encoder(**data).cpu().data.numpy())
    return np.concatenate(embeds)

print("\nInferring the text embeds...\n")
text_pools = prepare_texts()

for i in range(len(text_pools)):
    text_pool = text_pools[i]
    text_embeds = infer_text_embeds(mulan_module, text_pool)
    pickle.dump(
        [text_pool, text_embeds], open(f"{text_embed_folder}/text_pool{i}.pkl", "wb")
    )
