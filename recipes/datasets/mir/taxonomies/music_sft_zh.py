from recipes.mi1.models.tokenizers import NoOpOneHotTokenizer, NoOpTokenizer
from recipes.datasets.mir.taxonomies.music_sft_en import TranslationTokenizer

## Official MusicSFT Taxonomy
THEME = {
    "celebrations and happy events",
    "children's songs",
    "classical/mythology",
    "farmer's life",
    "friendship/family",
    "hometown and memories",
    "hopes and dreams",
    "life's hardships",
    "love/romance/sweetness",
    "lovelorn and heartbreak",
    "missing/longing",
    "natural scenery",
    "others",
    "patriotism",
    "peace and freedom",
    "religious beliefs",
    "self-reflection and life philosophy",
    "women",
    "youth/time",
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
    "date",
    "entertainment",
    "evening",
    "family time",
    "flirt",
    "focus",
    "food",
    "game",
    "graduation",
    "halloween",
    "landscape/scenery",
    "lounge",
    "marketplace",
    "meditation",
    "mid-autumn festival",
    "morning",
    "national's day ",
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
    "bell",
    "brass",
    "chinese folk instrument",
    "clarinet",
    "di",
    "doublebass",
    "drum",
    "electric guitar",
    "electronic piano",
    "ethnic",
    "guitar",
    "harmonica",
    "keys",
    "marimba",
    "other",
    "percussion",
    "pipa",
    "plucked-string instrument",
    "saxophone",
    "strings",
    "suona",
    "synthesizer",
    "woodwind",
    "xiao",
    "yangqin",
    "zheng",
}

GENRES_1 = {
    "alternative/indie",
    "bgm",
    "childhood",
    "chinese style",
    "chinese tradition",
    "classical",
    "devotional",
    "electronic",
    "folk",
    "hip hop",
    "jazz",
    "latin",
    "metal",
    "other",
    "pop",
    "r&b/soul",
    "rock",
    "tuhai",
}

GENRES_2 = {
    "a cappella",
    "cantopop",
    "china-wave",
    "chinese pop",
    "chinoiserie electronic",
    "chinoiserie rap",
    "country pop",
    "dance pop",
    "dj",
    "edm",
    "electropop",
    "funk",
    "gufeng music",
    "house",
    "indie pop",
    "indie rock",
    "jazz fusion",
    "jazz pop",
    "pop folk",
    "pop rock",
    "r&b rap",
    "taiwanese pop",
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
    "magnetic",
    "other",
    "powerful",
    "sexy/lazy",
    "sharp",
    "sweet",
    "warm",
}

VOICE_GENDER = {"child", "chorus", "female", "male", "neutral"}

TOP_GENRES = {
    "Pop, Pop Folk",
    "Pop, Taiwanese Pop",
    "Pop, Cantopop",
    "Pop, Contemporary Pop",
    "Pop, Chinese Pop",
    "Hip Hop, Trap Rap",
    "Hip Hop, R&B Rap",
    "Hip Hop, Old School",
    "Tuhai, DJ",
    "Tuhai, MC",
    "Tuhai, VinaHouse",
    "Jazz, Jazz Pop",
    "Classical, Funk",
    "Classical, R&B/Soul",
    "Chinese Tradition, Traditional Chinese Folk",
    "Chinese Tradition, Chinese Opera",
    "Chinese Style, GuFeng Music",
    "Chinese Style, Chinoiserie Rap",
    "Chinese Style, Chinoiserie Electronic",
    "Chinese Style, China-Wave",
    "Rock",
    "Metal",
    "Childhood",
    "Devotional",
}

TOP_GENRES_1 = {
    "流行": "Pop",
    "嘻哈": "Hip Hop", 
    "下沉土嗨": "Tuhai",
    "国风音乐": "Chinese Style",
    "摇滚": "Rock",
    "古典": "Classical",
    "爵士": "Jazz",
    "中国传统": "Chinese Tradition",
    "金属": "Metal",
    "儿童音乐": "Childhood",
    "宗教": "Devotional"
}

TOP_GENRES_2 = {
    "流行民谣": "Pop Folk",
    "闽南语流行": "Taiwanese Pop",
    "DJ慢摇/土味remix": "DJ",
    "陷阱说唱": "Trap Rap",
    "流行爵士": "Jazz Pop",
    "旋律性说唱": "R&B Rap",
    "放克音乐": "Funk",
    "中国风流行音乐": "China-Wave",
    "粤语流行": "Cantopop",
    "怀旧流行": "Contemporary Pop",
    "喊麦": "MC",
    "传统民歌": "Traditional Chinese Folk",
    "古风音乐": "GuFeng Music",
    "国风嘻哈": "Chinoiserie Rap",
    "中国戏曲": "Chinese Opera",
    "节奏蓝调/灵魂": "R&B/Soul",
    "越南鼓": "VinaHouse",
    "老派说唱": "Old School",
    "国风电子": "Chinoiserie Electronic",
    "国语流行": "Chinese Pop",
}

TOP_MOODS = {
    "怀旧的/记忆": "Nostalgic/Memory",
    "悲伤": "Sorrow/Sad",
    "开心/快乐": "Happy",
    "想念": "Miss",
    "治疗": "Healing",
    "动态的/精力充沛的": "Dynamic/Energetic",
    "可爱/调皮": "Cute/Playful",
    "震撼/壮丽": "Shocking/magnificent/epicl",
    "浪漫": "Romantic",
    "鼓舞人心的/希望的": "Inspirational/Hopeful",
    "平静": "Calm/Relaxing",
    "兴奋": "Excited",
    "梦幻/超凡脱俗的": "Dreamy/Ethereal",
    "律动": "Groovy/Funky", # TODO
}

TOP_SCENES = {
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
    "Babies": "Babies",
    "Landscape/Scenery": "Landscape/Scenery",
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
    "Children": "Children",
    "Summer": "Summer",
    "Autumn": "Autumn",
    "Focus": "Focus",
    "Valentine's day": "Valentine's day",
    "Flirt": "Flirt",
    "Mid-autumn Festival": "Mid-autumn Festival",
    "Lounge": "Lounge",
    "Spring Festival": "Spring Festival",
    "Winter": " Winter",
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
}


def get_vocab(names):
    # used to generate above vocabulary sets from parquet files
    return sorted(
        list(filter(None, set([f.lower() for g in list(names) for f in g.split(",")])))
    )


class MusicSFTTokenizerZH(NoOpOneHotTokenizer):
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

class MusicSFTVocalTokenizerZH(NoOpOneHotTokenizer):
    def __init__(self):
        vocab = set()
        vocab.update(VOICE_GENDER)
        vocab.update(VOICE_CHAR)
        vocab = sorted(vocab)
        super().__init__(vocab)

    def preprocess(self, name: str) -> str:
        name = name.lower()
        return name

class MusicSFTTokenizerGenresZH(NoOpTokenizer):

    def __init__(self):
        super().__init__(TOP_GENRES)


class MusicSFTTokenizerGenresAllZH(TranslationTokenizer):

    def __init__(self):
        translations = {}
        translations.update(TOP_GENRES_1)
        translations.update(TOP_GENRES_2)
        super().__init__(translations)

class MusicSFTTokenizerMoodsZH(TranslationTokenizer):

    def __init__(self):
        super().__init__(TOP_MOODS)

class MusicSFTTokenizerScenesZH(TranslationTokenizer):

    def __init__(self):
        super().__init__(TOP_SCENES)

if __name__ == "__main__":
    import json

    import pandas as pd
    from tqdm import tqdm

    from samantha.utils.hdfs_helper import hdfs_ls

    index_files = hdfs_ls(
        "hdfs://haruna/home/byte_data_seed/lf_lq/speech/data_store/BigMusic/music_sft-en-20231221_Smcc_Mvocal_N6.2k/index_1/*.parquet"
    )
    df = pd.concat([pd.read_parquet(f) for f in tqdm(index_files)])
    df = pd.io.json.json_normalize(df["meta"].apply(json.loads))
