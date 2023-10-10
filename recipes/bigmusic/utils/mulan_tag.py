from recipes.bigmusic.datasets.mix import rewrite_metadata

text_pool_1_toplevel = [
    'Blues',
    "Children's Music",
    'Classical Music',
    'Country',
    'Devotional',
    'EDM',
    'Electronic Music',
    'Experimental',
    'Folk',
    'Hip Hop',
    'Jazz',
    'Latin',
    'MC',
    'Metal',
    'New Age',
    'Pop',
    'Punk',
    'R&B',
    'Reggae',
    'Rock',
    'Sound Effect',
    'Sound Track',
    'World Music'
]


text_pool_1_subgenre = [
    'EDM',
    'Electronic Music',
    'Country',
    'Folk',
    'Hip Hop',
    'Pop',
    'Rock',
    'Metal',
    'Punk',
    'Jazz',
    'Blues',
    'R&B',
    'Reggae',
    'Classical Music',
    'Latin',
    'New Age',
    'World Music',
    'Devotional',
    "Children's Music",
    'Experimental',
    'MC',
    'Sound Track',
    'Sound Effect',
    'Acapella',
    'Techno',
    'Trance',
    'House',
    'Disco',
    'Dubstep',
    'Future Bass',
    'Reggaeton',
    'DJ',
    'Ambient',
    '8 Bit / Chiptune',
    'Chillout',
    'Bluegrass',
    'Pop Rap',
    'Trap Rap',
    'Jazz Hip Hop',
    'Hardcore Rap',
    'Hip House',
    'Boombap',
    'K-Pop',
    'Dance Pop',
    'Easy Listening',
    'Chinese Pop',
    'Hard Rock',
    'Psychedelic Rock',
    'Instrumental Rock',
    'Metalcore',
    'Swing',
    'Bebop',
    'Big Band',
    'Jazz Fusion',
    'Cool Jazz',
    'Bossa Nova',
    'Ragtime',
    'Funk',
    'Soul',
    'Symphony',
    'Chamber Music',
    'Baroque',
    'Opera',
    'Latin Pop',
    'Tango',
    'Samba'
]

text_pool_2 = [
    'Angry/Aggressive',
    'Calm/Relaxing',
    'Chill',
    'Cute/Playful',
    'Dreamy/Ethereal',
    'Dynamic/Energetic',
    'Excited',
    'Funny',
    'Groovy/Funky',
    'Happy',
    'Healing',
    'Inspirational/Hopeful',
    'Miss',
    'Mysterious',
    'No Mood',
    'Nostalgic/Memory',
    'Romantic',
    'Sentimental/Melancholic/Lonely',
    'Shocking/magnificent/epic',
    'Sorrow/Sad',
    'Thrilling/Suspenseful/Tense',
    'Weird'
]

text_pool_3 = [
    "flute",
    "clarinet",
    "oboe",
    "saxophone",
    "bassoon",
    "trumpet",
    "frenchhorn",
    "trombone",
    "tuba",
    "drum",
    "marimba",
    "bell",
    "timpani",
    "acoustic piano",
    "electronic piano",
    "accordian",
    "violin",
    "viola",
    "cello",
    "doublebass",
    "bass",
    "acoustic guitar",
    "electric guitar",
    "Di",
    "Xiao",
    "Suona",
    "Sheng",
    "Huqin",
    "Zheng",
    "Ruan",
    "Pipa",
    "Yangqin",
    "synthesizer",
]

text_pool_4 = ["Female", "Male"]

class MulanTagger:
    def __init__(self, include_subgenres=True):
        self._tag2embed = None
        self.include_subgeners = include_subgenres
        if include_subgenres:
            self._tag2text_pool = {
                "genre": text_pool_1_subgenre, 
                "mood": text_pool_2, 
                "gender": text_pool_4
            }
        else:
            self._tag2text_pool = {
                "genre": text_pool_1_toplevel, 
                "mood": text_pool_2, 
                # "instrument": text_pool_3, # skip instruments for now
                "gender": text_pool_4
            }

    def get_tag_embeds(self, requires):
        # TODO: (AS) move this out of inner function. Currently here to remove circular dependency
        from recipes.bigmusic.lightning.embedding_modules import get_mulan_embeds
        # default text pool embeddings
        if self._tag2embed is None:
            self._tag2embed = {}
            for tag_label, category_labels in self._tag2text_pool.items():
                category_embeds = get_mulan_embeds(
                    requires, category_labels, data_type='text'
                )
                self._tag2embed[tag_label] = { 'labels': category_labels, 'embeds': category_embeds }
        return self._tag2embed

    def tag_to_style_text(self, metadata):
        # rewrite metadata to mcc form
        genre, mood, gender = metadata['genre'], metadata['mood'], metadata['gender']
        mood = None if mood == 'No Mood' else mood
        metadata = { "final_genre": genre, "final_mood": mood, "merge_aed": gender }
        return rewrite_metadata(metadata)

    def get_tags(self, requires, audio_embeds):
        all_tag_embeds = self.get_tag_embeds(requires)

        item_metadata = [{} for _ in range(audio_embeds.shape[0])]
        for cat_idx, (tag_label, categories) in enumerate(all_tag_embeds.items()):
            category_labels, category_embeds = categories['labels'], categories['embeds']
            scores = audio_embeds @ category_embeds.T
            scores = scores.argmax(dim=1)
            for item_idx, score in enumerate(scores):
                item_metadata[item_idx][tag_label] = category_labels[score]
        return item_metadata
