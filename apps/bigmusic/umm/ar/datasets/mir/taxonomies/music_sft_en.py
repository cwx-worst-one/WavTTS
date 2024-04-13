from typing import Dict
from apps.bigmusic.umm.ar.datasets.tokenizers.tokenizers import NoOpOneHotTokenizer, NoOpTokenizer

## Official MusicSFT Taxonomy
THEME = {
    "nostalgia",
    "失恋与心碎",
    "其他",
    "celebration and happiness",
    "思念",
    "儿童歌曲",
    "庆典喜事",
    "美好生活",
    "儿歌",
    "青春/时光",
    "友谊/家庭",
    "生死和永恒",
    "自然风光",
    "古典/神话",
    "宗教信仰",
    "other",
    "love and romance",
    "欢乐和庆祝",
    "农民生活",
    "game",
    "挫折和迷茫",
    "希望梦想",
    "nature and environment",
    "朋友友情",
    "挑战diss",
    "sport",
    "heartbreak and loss",
    "fantasy and dream",
    "自我表达和自信",
    "peace and harmony",
    "家乡与回忆",
    "爱情/浪漫/甜蜜",
    "和平自由",
    "自我反省和人生思考",
    "激励鼓励",
    "奇幻冒险",
    "religion",
    "freedom",
    "爱国",
    "女性",
    "生活磨难",
    "回忆和怀旧",
    "friendship and family",
    "motivational",
}

SCENE = {
    "anime",
    "autumn",
    "babies",
    "bar",
    "beach",
    "beauty/fashion",
    "birthday",
    "campus",
    "children",
    "christmas",
    "cleaning and chores",
    "coffee shop",
    "commute",
    "dance",
    "danceable",
    "date",
    "entertainment",
    "evening",
    "family time",
    "flirt",
    "focus",
    "food",
    "funeral",
    "game",
    "graduation",
    "halloween",
    "landscape/scenery",
    "lounge",
    "marketplace",
    "meditation",
    "morning",
    "new year",
    "nightclub",
    "other",
    "park",
    "party",
    "pet/animals",
    "prank",
    "rainy day",
    "relaxation",
    "restaurants",
    "roadtrip",
    "running",
    "sport",
    "spring",
    "summer",
    "sunny day",
    "theater / concert hall",
    "timelapse",
    "transition (卡点)",
    "travel",
    "universe",
    "valentine's day",
    "vlog/dailylife",
    "wake up",
    "wedding",
    "winter",
}

MOOD = {
    "angry/aggressive",
    "calm/relaxing",
    "chill",
    "cute/playful",
    "dreamy/ethereal",
    "dynamic/energetic",
    "excited",
    "funny",
    "groovy/funky",
    "happy",
    "healing",
    "inspirational/hopeful",
    "miss",
    "mysterious",
    "no mood",
    "nostalgic/memory",
    "romantic",
    "sentimental/melancholic/lonely",
    "shocking/magnificent/epic",
    "sorrow/sad",
    "thrilling/suspenseful/tense",
    "weird",
}

INSTRUMENTS = {
    "accordian",
    "acoustic guitar",
    "acoustic piano",
    "bass",
    "bassoon",
    "bell",
    "brass",
    "cello",
    "chinese folk instrument",
    "clarinet",
    "di",
    "doublebass",
    "drum",
    "electric guitar",
    "electronic piano",
    "ethnic",
    "flute",
    "frenchhorn",
    "guitar",
    "harmonica",
    "huqin",
    "keys",
    "marimba",
    "oboe",
    "other",
    "percussion",
    "pipa",
    "ruan",
    "saxophone",
    "strings",
    "suona",
    "synthesizer",
    "timpani",
    "trombone",
    "trumpet",
    "tuba",
    "viola",
    "violin",
    "woodwind",
    "xiao",
    "yangqin",
    "zheng",
}

GENRES_1 = {
    "acapella",
    "alternative/indie",
    "bgm",
    "blues",
    "childhood",
    "children's music",
    "chinese style",
    "chinese tradition",
    "classical",
    "classical music",
    "country",
    "devotional",
    "easy listening",
    "edm",
    "electronic",
    "electronic music",
    "epic",
    "experimental",
    "folk",
    "hip hop",
    "jazz",
    "latin",
    "metal",
    "new age",
    "other",
    "pop",
    "punk",
    "r&b",
    "r&b/soul",
    "reggae",
    "rock",
    "sound effect",
    "sound track",
    "tuhai",
    "world music",
}

GENRES_2 = {
    "8 bit / chiptune",
    "a cappella",
    "ambient",
    "baroque",
    "bebop",
    "big band",
    "bluegrass",
    "boombap",
    "bossa nova",
    "cantopop",
    "chamber music",
    "chillout",
    "china-wave",
    "chinese pop",
    "chinoiserie electronic",
    "chinoiserie rap",
    "city pop",
    "cool jazz",
    "dance pop",
    "disco",
    "dj",
    "dubstep",
    "easy listening",
    "edm",
    "edm trap",
    "electropop",
    "funk",
    "future bass",
    "gufeng music",
    "hard rock",
    "hardcore rap",
    "hip house",
    "house",
    "indie pop",
    "instrumental rock",
    "jazz fusion",
    "jazz hip hop",
    "k-pop",
    "latin pop",
    "metalcore",
    "opera",
    "pop folk",
    "pop rap",
    "pop rock",
    "psychedelic rock",
    "ragtime",
    "reggaeton",
    "soul",
    "swing",
    "symphony",
    "synth pop",
    "taiwanese pop",
    "techno",
    "trance",
    "trap rap",
    "trip hop",
}

VOICE_CHAR = {
    "bright",
    "cute",
    "deep",
    "electrified voice",
    "ethereal",
    "extreme",
    "husky",
    "loud and sonorous",
    "low",
    "magnetic",
    "powerful",
    "raspy",
    "sexy/lazy",
    "sharp",
    "sweet",
    "warm",
}

VOICE_GENDER = {"child", "chorus", "female", "male", "neutral"}


TOP_GENRES = {
    "Blues": "blues",
    "Children's Music": "children's music",
    "Classical Music": "classical music",
    "Country": "country",
    "Devotional": "devotional",
    "EDM": "electronic music",
    "Electronic": "electronic music",
    "Electronic Music": "electronic music",
    "Experimental": "experimental",
    "Folk": "folk",
    "Hip Hop": "hip hop",
    "Jazz": "jazz",
    "Metal": "metal",
    "Pop": "pop",
    "Pop,Devotional": "pop",
    "Pop,Rock": "pop",
    "R&B": "r&b",
    "Reggae": "reggae",
    "Rock": "rock",
    "Sound Track": "sound track",
}

TOP_MOODS = {
    "Angry/Aggressive": "angry, aggressive",
    "Nostalgic/Memory": "nostalgic, memory",
    "Chill": "chill",
    "Funny": "funny",
    "Mysterious": "mysterious",
    "Sorrow/Sad": "sorrow, sad",
    "Happy": "happy",
    "Miss": "miss",
    "Healing": "healing",
    "Dynamic/Energetic": "dynamic, energetic",
    "Cute/Playful": "cute, playful",
    "Shocking/magnificent/epic": "shocking, magnificent, epic",
    "Thrilling/Suspenseful/Tense": "thrilling, suspenseful, tense",
    "Romantic": "romantic",
    "Inspirational/Hopeful": "inspirational, hopeful",
    "Calm/Relaxing": "calm, relaxing",
    "Excited": "excited",
    "Dreamy/Ethereal": "dreamy, ethereal",
    "Groovy/Funky": "groovy, funky",
    "Weird": "weird",
    "No Mood": "no mood",
}

TOP_SCENES = {
    "landscape/scenery": "Landscape/Scenery",
    "Evening": "Evening",
    "Danceable": "Danceable",
    "Relaxation": "Relaxation",
    "Running": "Running",
    "Marketplace": "Marketplace",
    "Wake up": "Wake up",
    "Game": "Game",
    "Restaurants": "Restaurants",
    "Beauty/Fashion": "Beauty/Fashion",
    "Campus": "Campus",
    "Family time": "Family time",
    "Rainy Day": "Rainy Day",
    "Graduation": "Graduation",
    "Party": "Party",
    "Coffee Shop": "Coffee Shop",
    "babies": "Babies",
    "Date": "Date",
    "Prank": "Prank",
    "Meditation": "Meditation",
    "Food": "Food",
    "Theater / Concert hall": "Theater / Concert hall",
    "Commute": "Commute",
    "Sport": "Sport",
    "National's Day": "National's Day",
    "Roadtrip": "Roadtrip",
    "Morning": "Morning",
    "children": "Children",
    "Summer": "Summer",
    "Autumn": "Autumn",
    "Focus": "Focus",
    "Valentine's day": "Valentine's day",
    "Flirt": "Flirt",
    "Mid-autumn Festival": "Mid-autumn Festival",
    "Lounge": "Lounge",
    "Spring Festival": "Spring Festival",
    "Winter": "Winter",
    "Sunny Day": "Sunny Day",
    "Nightclub": "Nightclub",
    "Anime": "Anime",
    "Spring": "Spring",
    "Qi Xi": "Qi Xi",
    "Universe": "Universe",
    "Wedding": "Wedding",
    "Beach": "Beach",
    "Vlog/DailyLife": "Vlog/Dailylife",
    "Pet/Animals": "Pet/Animals",
    "Birthday": "Birthday",
    "Transition (卡点)": "Transition (卡点)",
}

TOP_VOICE_GENDER = {
    "Male": "male",
    "Female": "female",
    "Female,Male": "female,male",
    "Chorus": "chorus",
    "Male,Chorus": "male,chorus",
    "Female,Chorus": "female,chorus"
}


def get_vocab(names):
    # used to generate above vocabulary sets from parquet files
    return sorted(
        list(filter(None, set([f.lower() for g in list(names) for f in g.split(",")])))
    )


class MusicSFTTokenizerEN(NoOpOneHotTokenizer):
    def __init__(self):
        vocab = set()
        vocab.update(GENRES_1)
        vocab.update(GENRES_2)
        vocab.update(THEME)
        vocab.update(SCENE)
        vocab.update(MOOD)
        vocab = sorted(vocab)
        super().__init__(vocab)

    def preprocess(self, name: str) -> str:
        name = name.lower()
        return name


class TranslationTokenizer(NoOpTokenizer):
    def __init__(self, translation: Dict[str, str]):
        self._translation = translation
        self._translation_keys = sorted(set(self._translation.keys()))

        vocab = set()
        vocab.update(self.translation.values())
        super().__init__(vocab)

    @property
    def translation_keys(self) -> list:
        return self._translation_keys

    @property
    def translation(self) -> list:
        return self._translation

    def translate(self, s: str) -> str:
        return self._translation[s]

    def preprocess(self, name: str) -> str:
        if name in self.translation_keys:
            name = self.translate(name)
        return name


class MusicSFTTokenizerGenresEN(TranslationTokenizer):

    def __init__(self):
        super().__init__(TOP_GENRES)

class MusicSFTTokenizerMoodsEN(TranslationTokenizer):

    def __init__(self):
        super().__init__(TOP_MOODS)

class MusicSFTTokenizerScenesEN(TranslationTokenizer):

    def __init__(self):
        super().__init__(TOP_SCENES)

class MusicSFTTokenizerVoiceGenderEN(TranslationTokenizer):

    def __init__(self):
        super().__init__(TOP_VOICE_GENDER)

class MusicSFTVocalTokenizerEN(NoOpOneHotTokenizer):
    def __init__(self):
        vocab = set()
        vocab.update(VOICE_GENDER)
        vocab.update(VOICE_CHAR)
        vocab = sorted(vocab)
        super().__init__(vocab)

    def preprocess(self, name: str) -> str:
        name = name.lower()
        return name
