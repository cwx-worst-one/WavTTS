from recipes.bigmusic.datasets.mix import rewrite_metadata

text_pool_1 = [
    "EDM",
    "Electronic Music",
    "Country",
    "Folk",
    "Hip Hop",
    "Pop",
    "Rock",
    "Metal",
    "Punk",
    "Jazz",
    "Blues",
    "R&B",
    "Reggae",
    "Classical Music",
    "Latin",
    "Chinese Tradition",
    "New Age",
    "World Music",
    "Devotional",
    "Children's Music",
    "Experimental",
    "MC",
    "Sound Track",
    "Sound Effect",
]
text_pool_2 = [
    "Happy",
    "Cute/Playful",
    "Excited",
    "Funny",
    "Inspirational/Hopeful",
    "Chill",
    "Calm/Relaxing",
    "Sorrow/Sad",
    "Sentimental/Melancholic/Lonely",
    "Mysterious",
    "Weird",
    "Thrilling/Suspenseful/Tense",
    "Shocking/magnificent/epic",
    "Angry/Aggressive",
    "Groovy/Funky",
    "Dynamic/Energetic",
    "Romantic",
    "Nostalgic/Memory",
    "Dreamy/Ethereal",
    "Healing",
    "Miss",
    "No Mood",
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
TAG_TO_TEXT_POOL = {
    "genre": text_pool_1, 
    "mood": text_pool_2, 
    # "instrument": text_pool_3, # skip instruments for now
    "gender": text_pool_4
}

class MulanTagger:
    def __init__(self):
        self._tag2embed = None

    def get_tag_embeds(self, requires):
        # TODO: (AS) move this out of inner function. Currently here to remove circular dependency
        from recipes.bigmusic.lightning.embedding_modules import get_mulan_embeds
        # default text pool embeddings
        if self._tag2embed is None:
            self._tag2embed = {}
            for tag_label, category_labels in TAG_TO_TEXT_POOL.items():
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
