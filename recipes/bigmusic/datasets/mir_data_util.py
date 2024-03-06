from functools import reduce
import operator
from typing import Dict, Tuple


ARTIST_ID_MAP = {
    # En dropout
    "en_empty": 0,
    # Deepmind
    "5IH6FPUwQTxPSXurCrcIov": 1,    # Alec Benjamin
    "6VuMaDnrHyPL1p4EHjYLi7": 2,    # Charlie Puth
    "25uiPmTg16RbhZWAqwLBy5": 3,    # Charli XCX
    "6S2OmqARrzebs0tKUEyXyp": 4,    # Demi Lovato
    "5y2Xq6xcjJb2jVM54GHK3t": 5,    # John Legend
    "5WUlDfRSoLAfcVSX1WnrxN": 6,    # Sia
    "3aQeKQSyrW4qWr35idm0cy": 7,    # T-Pain
    "3WGpXCj9YhhfX11TToZcXP": 8,    # Troye Sivan
    "0mbgkaYR8KNUb5s3s1yYHG": 9,    # Papoose
    # En Artist 17
    "74ASZWbe4lXaubB36ztrGX": 11,   # Bob Dylan
    "5e4Dhzv426EvQe3aDb64jL": 12,   # Shania Twain
    "3fMbdgg4jU18AjLCKBhRSm": 13,   # Michael Jackson
    "43ZHCT0cAZBISjO8DG9PnE": 14,   # Elvis Presley
    "3WrFJ7ztbogyGnTHbHJFl2": 15,   # The Beatles
    "1dfeR4HaWDbWqFHLkxsg1d": 16,   # Queen
    "3TVXtAsR1Inumwj472S9r4": 17,   # Drake
    "4kYSro6naA4h99UJvo89HB": 18,   # Cardi B
    "5V0MlUE1Bft0mbLlND7FJz": 19,   # Ella Fitzgerald
    "2QsynagSdAqZj3U9HgDzjD": 20,   # Bob Marley
    "6vWDO969PvNqNYHIOW5v0m": 21,   # Beyonce
    "5pKCCKE2ajJHZ9KAiaK11H": 22,   # Rihanna
    "7guDJrEfX3qb6FEbdPA5qi": 23,   # Stevie Wonder
    "1uNFoZAHBGtllmzznpCI3s": 24,   # Justin Bieber
    "0du5cEVh5yTK9QJze8zA0C": 25,   # Bruno Mars
    "7dGJo4pcD2V6oG8kP0tJRR": 26,   # Eminem
    "4S9EykWXhStSc15wEx8QFK": 27,   # Celine Dion
    # Zh droput
    "zh_empty": 50,
    # Zh Artist 16
    "2elBjNSdBE2Y3f0j1mjrql": 51,
    "40tNK2YedBV2jRFAHxpifB": 52,
    "16s0YTFcyjP4kgFwt7ktrY": 53,
    "1piwhsT7JM9pBzmLcLCg8v": 54,
    "10LslMPb7P5j9L2QXGZBmM": 55,
    "3qObkSIVSfNNDL1vbwL2N2": 56,
    "7Dx7RhX0mFuXhCOUgB01uM": 57,
    "2n3uDrupL8UtFSeZhY38MS": 58,
    "1Hu58yHg2CXNfDhlPd7Tdd": 59,
    "3WYT2b8pOLsLsqSaoWYr7U": 60,
    "0SIXZXJCAhNU8sxK0qm7hn": 61,
    "3ienC90A5I1X3irDyQoqWZ": 62,
    "6jlz5QSUqbKE4vnzo2qfP1": 63,
    "3AroL2oDPiAnMpTmIQv3KP": 64,
    "1r9DuPTHiQ7hnRRZ99B8nL": 65,
    "6qzfo7jiO4OrhxrvPFPlWX": 66,
}

ARTIST_ID_MAP_V2 = {
    # Zh droput
    "zh_empty": 0,
    # Zh Artist 16
    "6681166129722824706": 1,  # 周杰伦 280	 
    "6808079824435808257": 2,  # 陶喆	157	 
    "6857838729394915329": 3,  # 五月天	199	 
    "6805087443797149697": 4,  # 林俊杰	280	 
    "6816638586171951106": 5,  # 刘德华	728	 
    "6818415107001812994": 6,  # 张学友	571	 
    "6815161293339641858": 7,  # 苏打绿	139	 
    "6841752286910220289": 8,  # 孙燕姿	180	 
    "6815165111875930113": 9,  # 邓丽君	1007	 
    "6807327071803541505": 10, # 莫文蔚	312	 
    "6807661018613811201": 11, # 王心凌	183	 
    "6754918579642042369": 12, # 蔡依林	250	 
    "6818114466299774978": 13, # 颜人中	68	 
    "6805758982225922049": 14, # 余佳运	73	 
    "6797991849819637762": 15, # 华晨宇	123	 
    "6854432959231952897": 16, # Jony J	71	 
    "6774291469118212098": 17, # VaVa毛衍七	119	 
    "6799901283898624002": 18, # 周深	261	 
    "6761698620967225345": 19, # G.E.M. 邓紫棋	157	 
    "6939351367434405889": 20, # 队长	63	 
    "6911572607473027074": 21, # 王忻辰	66	 
    "6815146974182934529": 22, # 方大同	187	 
    "6803170060908103682": 23, # Lil Ghost小鬼	56	 
    "6816676362728769538": 24, # 杨宗纬	118	 
    "6843221283203713026": 25, # 林宥嘉	164	 
    "6817686547018549250": 26, # 李荣浩	125	 
    "6782878109353019393": 27, # 张韶涵	189	 
    "6810215791951087618": 28, # 蔡健雅	209	 
    "6795468680730773505": 29, # 王以太	86	 
    "6817693369125308417": 30, # S.H.E	171	 
    "6910084011582834689": 31, # 单依纯	84	 
    "6792011207390791682": 32, # Eric周兴哲	81	 
    "6818419502233962497": 33, # 音阙诗听	205	 
    "6815151518128293889": 34, # 李健	125	 
    "6776144869279664130": 35, # 陈粒	142	 
    "6817645899284482049": 36, # 毛不易	129	 
    "6795061528409147393": 37, # 汪苏泷	239 
    # Voice tags
    "Child": 47,
    "Male": 48,
    "Female": 49,
    # En dropout
    "en_empty": 50,
    '6705198980290091009' : 51,   #    Frank Sinatra    1844
    '6693699473027127298' : 52,   #    Ella Fitzgerald    1583
    '6683719970804733953' : 53,   #    Elvis Presley    1010
    '6781886425395628034' : 54,   #    Lil Nas X    25
    '6696193284760410114' : 55,   #    Dua Lipa    59
    '6728012888994220034' : 56,   #    Nicki Minaj    248
    '6696207516872742914' : 57,   #    Céline Dion    381
    '6698575526623188994' : 58,   #    Demi Lovato    168
    '6886747810603993089' : 59,   #    Olivia Rodrigo    46
    '6887898290780637186' : 60,   #    Miley Cyrus    168
    '6816644483610839041' : 61,   #    Imagine Dragons    128
    '6815513435095189505' : 62,   #    Kanye West    343
    '6815163859960072194' : 63,   #    Eminem    341
    '6781886475391731713' : 64,   #    Drake    413
    '6759681496765696002' : 65,   #    Lana Del Rey    151
    '6698575504477263873' : 66,   #    Coldplay    172
    '6856639593832253442' : 67,   #    Stevie Wonder    426
    '6798610658091862017' : 68,   #    Lil Baby    316
    '6834041359557462018' : 69,   #    Bob Dylan    1053
    '6698575291008161793' : 70,   #    Charli XCX    138
    '6679720590640683009' : 71,   #    Maroon 5    123
    '6792447992432429057' : 72,   #    Pop Smoke    114
    '6691744684106061825' : 73,   #    Justin Bieber    155
    '6873470890248505345' : 74,   #    Shania Twain    129
    '6816722136460167169' : 75,   #    Beyoncé    219
    '6698575458008569858' : 76,   #    Bruno Mars    64
    '6683719711886153730' : 77,   #    Lady Gaga    157
    '6999682166288287746' : 78,   #    Taylor Swift    240
    '6807925639811696642' : 79,   #    The Weeknd    170
    '6707563235899365378' : 80,   #    Bob Marley & The Wailers    314
    '6699033012245370881' : 81,   #    Selena Gomez    134
    '6702330119991597058' : 82,   #    Rihanna    159
    '6790270596862183426' : 83,   #    Queen    279
    '6937767713892730882' : 84,   #    Jon Batiste    105
    '6808401506824357889' : 85,   #    Bad Bunny    222
    '6816931983969486849' : 86,   #    Ed Sheeran    180
    '6698575554284623874' : 87,   #    Sam Smith    132
    '6817978846541776898' : 88,   #    Michael Jackson    267
    '6715159620823816194' : 89,   #    Doja Cat    77
    '6691308064122869762' : 90,   #    Luke Combs    83
    '6698579714027558914' : 91,   #    Charlie Puth    63
    '6698575972590950401' : 92,   #    Ariana Grande    155
    '6696211211471558657' : 93,   #    T-Pain    408
    '6691305168102758402' : 94,   #    Post Malone    110
    '6691297211898140673' : 95,   #    Kane Brown    80
    '6789842187606558722' : 96,   #    John Legend    248
    '6698575430993057793' : 97,   #    Lizzo    49
    '6823967124738803713' : 98,   #    The Beatles    188
    '6696197177665918978' : 99,   #    Fifth Harmony    55
    '6807313077516634113' : 100,   #    The Kid LAROI    88
    '6824713158125422593' : 101,   #    Daniel Caesar    63
    '6779401946852755457' : 102,   #    Sia    234
    '6898603714542569474' : 103,   #    Alec Benjamin    64
    '6698576944671234050' : 104,   #    Cardi B    81
    '6876461055883610114' : 105,   #    JVKE    31
    '6824322210665072641' : 106,   #    SZA    78
    '6730479131982563329' : 107,   #    Joji    85
    '7058136012715509762' : 108,   #    Steve Lacy    58
    '6838473905233987586' : 109,   #    Harry Styles    35    
}

CHINESE_MIR_GENRE_TAG_MAP = {
    # genre
    "Rock":                                 "摇滚",
    "Metal":                                "金属",
    "Childhood":                            "儿童音乐",
    "Devotional":                           "宗教",
    "Pop, Pop Folk":                        "流行民谣",
    "Pop, Taiwanese Pop":                   "闽南语流行",
    "Tuhai, DJ":                            "DJ慢摇/土味remix",
    "Hip Hop, Trap Rap":                    "陷阱说唱",
    "Jazz, Jazz Pop":                       "流行爵士",
    "Hip Hop, R&B Rap":                     "旋律性说唱",
    "Classical, Funk":                      "放克音乐",
    "Chinese Style, China-Wave":            "中国风流行音乐",
    "Pop, Cantopop":                        "粤语流行",
    "Pop, Contemporary Pop":                "怀旧流行",
    "Tuhai, MC":                            "喊麦",
    "Chinese Tradition, Traditional Chinese Folk": "传统民歌",
    "Chinese Style, GuFeng Music":          "古风音乐",
    "Chinese Style, Chinoiserie Rap":       "国风嘻哈",
    "Chinese Tradition, Chinese Opera":     "中国戏曲",
    "Classical, R&B/Soul":                  "节奏蓝调/灵魂",
    "Tuhai, VinaHouse":                     "越南鼓",
    "Hip Hop, Old School":                  "老派说唱",
    "Chinese Style, Chinoiserie Electronic":   "国风电子",
    "Pop, Chinese Pop":                     "国语流行",
    # mood    
    "Nostalgic/Memory":         "怀旧的/记忆",
    "Sorrow/Sad":               "悲伤",
    "Happy":                    "开心/快乐",
    "Miss":                     "想念",
    "Healing":                  "治疗",
    "Groovy/Funky":             "律动",
    "Dynamic/Energetic":        "动态的/精力充沛的",
    "Cute/Playful":             "可爱/调皮",
    "Shocking/magnificent/epic": "震撼/壮丽",
    "Romantic":                 "浪漫",
    "Inspirational/Hopeful":    "鼓舞人心的/希望的",
    "Calm/Relaxing":            "平静",
    "Excited":                  "兴奋",
    "Dreamy/Ethereal":          "梦幻/超凡脱俗的", 
    # scene    
    # gender
    "female":                   "女声",
    'female,chorus':            "女声",
    "male":                     "男声",
    'male,chorus':              "男声",
    "chorus":                   "男声",
    # lang
    "Mandarin":                 "普通话", 
    "Cantonese":                "粤语", 
    "Hokkien":                  "闽南话",
}

CHINESE_GENRE1_VOCAB = [
    "流行",     # use genre2
    "说唱",     # use genre2。"嘻哈"
    "下沉土嗨",  # use genre2
    "国风音乐",  # use genre2
    "摇滚",     # use genre2
    "古典",     # use genre2
    "爵士",     # use genre2
    "中国传统",  # use genre2
    "金属", 
    "儿童音乐",
    "宗教",
]

CHINESE_GENRE2_VOCAB = [
    "流行民谣",
    "闽南语流行",
    "DJ慢摇/土味remix",
    "陷阱说唱",
    "流行爵士",
    "旋律性说唱",
    "放克音乐",
    "中国风流行音乐",
    "粤语流行",
    "怀旧流行",
    "喊麦",
    "传统民歌",
    "古风音乐",
    "国风嘻哈",
    "中国戏曲",
    "节奏蓝调/灵魂",
    "越南鼓",
    "老派说唱",
    "国风电子",
    "国语流行", 
]

CHINESE_MOOD_VOCAB = [
    "怀旧的/记忆",
    "悲伤",
    "开心/快乐",
    "想念",
    "治疗",
    "律动",
    "动态的/精力充沛的",
    "可爱/调皮",
    "震撼/壮丽",
    "浪漫",
    "鼓舞人心的/希望的",
    "平静",
    "兴奋",
    "梦幻/超凡脱俗的",
]

CHINESE_SCENE_VOCAB = [
    "Evening",
    "Danceable",
    "Relaxation",
    "Marketplace",
    "Running",
    "Wake up",
    "Game",
    "Restaurants",
    "Beauty/Fashion",
    "Campus",
    "Family time",
    "Rainy Day",
    "Graduation",
    "Party",
    "Coffee Shop",
    "Babies",
    "Landscape/Scenery",
    "Date",
    "Prank",
    "Meditation",
    "Food",
    "Theater / Concert hall",
    "Commute",
    "Sport",
    "National's Day",
    "Roadtrip",
    "Morning",
    "Children",
    "Summer",
    "Autumn",
    "Focus",
    "Valentine's day",
    "Flirt",
    "Mid-autumn Festival",
    "Entertainment",
    "Lounge",
    "Spring Festival",
    "Winter",
    "Sunny Day",
    "Nightclub",
    "Anime",
    "Spring",
    "Qi Xi",
    "Universe",
    "Wedding",
    "Beach",
    "Vlog/DailyLife",
    "Pet/Animals",
    "Birthday",
]

GENDER_VOCAB = ["男声", "女声"]

LANG_VOCAB = ["普通话", "粤语", "闽南话"]

CHINESE_CAT_VOCAB = [
    CHINESE_GENRE1_VOCAB, CHINESE_GENRE2_VOCAB,
    CHINESE_MOOD_VOCAB, CHINESE_SCENE_VOCAB,
    GENDER_VOCAB, LANG_VOCAB
]

# ========================================================================
 
NONE_LABEL = 'None'

CHINESE_TO_SA_MAPPING = {
    "流行":     "Pop",     # use genre2
    "说唱":     "Hip Hop/Rap",     # use genre2。"嘻哈"
    "下沉土嗨":  "DJ",  # use genre2
    "国风音乐":  "Chinese Style",  # use genre2
    "摇滚":     "Rock",     # use genre2
    "古典":     "Classical",     # use genre2
    "爵士":     "Jazz",     # use genre2
    "中国传统":  "Chinese Tradition",  # use genre2
    "金属":     "Metal", 
    "儿童音乐":  "Other genre",
    "宗教":     "Other genre",
    "流行民谣":  "Folk",
    "闽南语流行": "None",   # Remove this category
    "DJ慢摇/土味remix": "DJ",
    "陷阱说唱":     "Hip Hop/Rap",
    "流行爵士":     "Jazz",
    "旋律性说唱":   "Hip Hop/Rap",
    "放克音乐":     "R&B/Soul",
    "中国风流行音乐": "Chinese Style",
    "粤语流行":     "Pop",
    "怀旧流行":     "Pop",
    "喊麦":         "MC",
    "传统民歌":     "Chinese Tradition",
    "古风音乐":     "Chinese Style",
    "国风嘻哈":     "Hip Hop/Rap",
    "中国戏曲":     "Chinese Tradition",
    "节奏蓝调/灵魂": "R&B/Soul",
    "越南鼓":   "Electronic",
    "老派说唱": "Hip Hop/Rap",
    "国风电子": "Electronic",
    "国语流行": "Pop", 
    "怀旧的/记忆":   "Miss",
    "悲伤":         "Sorrow",
    "开心/快乐":     "Happy",
    "想念":         "Miss",
    "治疗":         "Healing",
    "律动":         "Dynamic",
    "动态的/精力充沛的": "Dynamic",
    "可爱/调皮":    "Cute",
    "震撼/壮丽":    "Inspirational",
    "浪漫":         "Romantic",
    "鼓舞人心的/希望的": "Inspirational",
    "平静":         "Calm",
    "兴奋":         "Excited",
    "梦幻/超凡脱俗的": "Mysterious",
    "Evening":      "Evening",
    "Danceable":    "Danceable",
    "Relaxation":   "Cafe",
    "Marketplace":  "Campus",
    "Running":      "Sport",
    "Wake up":      "Morning",
    "Game":         "Game",
    "Restaurants":  "Cafe",
    "Beauty/Fashion": "Dance",
    "Campus":       "Campus",
    "Family time":  "Family",
    "Rainy Day":    "Broke up",
    "Graduation":   "Campus",
    "Party":        "Party",
    "Coffee Shop":  "Cafe",
    "Babies":       "Family",
    "Landscape/Scenery": "Travel",
    "Date":          "Date",
    "Prank":        "Halloween",
    "Meditation":   "Meditation",
    "Food":         "Food",
    "Theater / Concert hall": "Wedding",
    "Commute":      "Drive",
    "Sport":        "Sport",
    "National's Day": "Sport",
    "Roadtrip":     "Drive",
    "Morning":      "Morning",
    "Children":     "Family",
    "Summer":       "Summer",
    "Autumn":       "Autumn",
    "Focus":        "Focus",
    "Valentine's day": "Valentine's day",
    "Flirt":        "Date",
    "Mid-autumn Festival": "Family",
    "Entertainment": "Game",
    "Lounge":       "Cafe",
    "Spring Festival": "Spring Festival",
    "Winter":       "Winter",
    "Sunny Day":    "Spring",
    "Nightclub":    "Dance",
    "Anime":        "Game",
    "Spring":       "Spring",
    "Qi Xi":        "Valentine's day",
    "Universe":     "Focus",
    "Wedding":      "Wedding",
    "Beach":        "Summer",
    "Vlog/DailyLife": "Travel",
    "Pet/Animals":  "Family",
    "Birthday":     "Birthday",
    "普通话":        "Chinese", 
    "粤语":          "Cantonese",
}

SA_GENRE20 = [
    "Blues",
    "Chinese Opera",
    "Chinese Style",
    "Chinese Tradition",
    "Classical",
    "Country",
    "DJ",
    "Easy Listening",
    "Electronic",
    "Folk",
    "Hip Hop/Rap",
    "Jazz",
    "Latin",
    "MC",
    "Metal",
    "Other genre",
    "Pop",
    "Punk",
    "R&B/Soul",
    "Reggae",
    "Rock",
]

SA_MOOD19 = [
    "Angry",
    "Calm",
    "Chill",
    "Cute",
    "Dynamic",
    "Excited",
    "Funny",
    "Happy",
    "Healing",
    "Inspirational",
    "Lonely",
    "Miss/Memory",
    "Mysterious",
    "Other",
    "Romantic",
    "Sorrow",
    "Sweet",
    "Tense",
    "Weird",
]

SA_THEME33 = [
    "Autumn",
    "Bedtime",
    "Birthday",
    "Broke up",
    "Cafe",
    "Campus",
    "Christmas",
    "Dance",
    "Danceable",
    "Date",
    "Dream",
    "Drive",
    "Evening",
    "Family",
    "Focus",
    "Food",
    "Friendship",
    "Game",
    "Halloween",
    "Love",
    "Meditation",
    "Morning",
    "Other",
    "Party",
    "Sport",
    "Spring",
    "Spring Festival",
    "Summer",
    "Travel",
    "Valentine's Day",
    "Wedding",
    "Winter",
    "Yoga",
]

SA_LANG = [
    "Cantonese",
    "Chinese",
    "Chinese Dialects",
    "English",
]

SA_SINKING = [
    "Sinking", 
    "non-Sinking",
]

# Certain tags in the dataset should be replaced
SA_TAGS_SPECIAL_MAP = {
    "Нарру": "Happy",  # Confusable UTF code
    "Miss, Memory": "Miss/Memory",  # Comma is not nice for CSV
    "Miss,Memory": "Miss/Memory",
    "Pop,Chinese Style": "Chinese Style",
}


MACRO_STYLE_MAP = {
    "Pop": "Pop|||non-Sinking|Chinese",
    "Hip Hop/Rap": "Hip Hop/Rap|||non-Sinking|Chinese",
    "Chinese Style": "Chinese Style|||non-Sinking|Chinese",
    "Electronic": "Electronic|||non-Sinking|Chinese",
    "DJ": "DJ|||Sinking|Chinese",
    "Rock": "Rock|||non-Sinking|Chinese",
    "Folk": "Folk|||non-Sinking|Chinese",
    "R&B/Soul": "R&B/Soul|||non-Sinking|Chinese"
}

SA_CAT_VOCAB = [SA_GENRE20, SA_MOOD19, SA_THEME33, SA_SINKING, SA_LANG]

# Call it "voice" because:
# 1. GENDER_VOCAB has been used
# 2. will add more tags in the future
VOICE_THRESHOLDS = {
    "Male": 0.6,
    "Female": 0.65,
    # "Adult": 0.5,  # not used
    "Child": 0.85
}

VOICE_VOCAB = list(VOICE_THRESHOLDS.keys())

_VOCAB_MAP = {
    "Zh": CHINESE_CAT_VOCAB,
    "SA": SA_CAT_VOCAB,
}

def get_categorical_vocab(vocab_type: str) -> Tuple[Dict[str, int], int]:
    vocab = _VOCAB_MAP[vocab_type]
    num_categories = len(vocab)
    all_values = [NONE_LABEL] + reduce(operator.add, vocab)
    vocab2id = { value: idx for idx, value in enumerate(all_values) }
    return vocab2id, num_categories

def convert_m1_tag_to_style_text(m1_tags_list):    
    style_texts = []
    for m1_tags in m1_tags_list: 
        genre = CHINESE_MIR_GENRE_TAG_MAP.get(m1_tags.get("genre", ""), "")
        mood = CHINESE_MIR_GENRE_TAG_MAP.get(m1_tags.get("mood", ""), "")
        scene = m1_tags.get("scene", "")
        gender = CHINESE_MIR_GENRE_TAG_MAP.get(m1_tags.get("vocal_gender", ""), "")
        lang = m1_tags.get("lang", "")
        if not lang:
            lang = "普通话"
            if genre == "闽南语流行":
                lang = "闽南话"
            if genre == "粤语流行":
                lang = "粤语"            
        style_text = "|".join([genre, mood, scene, gender, lang])
        style_texts.append(style_text)
    return style_texts
