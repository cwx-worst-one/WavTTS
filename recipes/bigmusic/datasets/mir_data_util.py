from functools import reduce
import logging
import operator
from typing import Dict, Tuple, Optional
import random

from recipes.bigmusic.datasets.utils.zh_vocab import (
    VOCAB2ID_SA,
    VOCAB2ID_AUDIO_V0,
    VOCAB2ID_MIX_V0,
    VOCAB2ID_AUDIO_V1,
    VOCAB2ID_MIX_V1,
)

logger = logging.getLogger(__file__)


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

MACRO_STYLE_MAP_MOOD2_SA_MOOD19 = {
    'Calm/Relaxing': 'Calm',
    'Nostalgic/Memory': 'Miss,Memory',
    #'Nostalgic/Memory': 'Miss/Memory',
    'Sorrow/Sad': 'Sorrow',
    'Inspirational/Hopeful': 'Inspirational',
    'Dynamic/Energetic': 'Dynamic',
    'Miss': 'Miss,Memory',
    #'Miss': 'Miss/Memory',
    'Sentimental/Melancholic/Lonely': 'Lonely',
    'Dreamy/Ethereal': 'Mysterious',
    'Groovy/Funky': 'Dynamic',
    'Angry/Aggressive': 'Angry',
    'Cute/Playful': 'Cute',
}

SA_MOOD19 = [
    "Angry",
    "Calm",
    "Chill",
    "Cute_SA_MOOD",  # dedup
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
    "Sweet_SA_MOOD",  # dedup
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

AUDIO_GENRE_V0 = [
    "Pop",
    "Electronic",
    "Chinese Style",
    "Rock",
    "Jazz",
    "Hip Hop/Rap",
    "Classical",
    "R&B/Soul",
    "Metal",
    "Tuhai",
    "Childhood",
    "Devotional",
    "Chinese Tradition",
    "Easy Listening",
    "New Age",
    "8 Bit",
    "Folk",
    "Latin",
    "Alternative/Indie",
    "Epic",
    "BGM",
    "Chinese Pop",
    "Cantopop",
    "Taiwanese Pop",
    "Pop Folk",
    "Contemporary Pop",
    "Teen Pop",
    "Indie Pop",
    "Dream Pop",
    "City Pop",
    "Synth Pop",
    "Celtic Pop",
    "Dance Pop",
    "Deep Dance Pop",
    "Electropop",
    "Chamber Pop",
    "A cappella",
    "Country Pop",
    "EDM",
    "House",
    "Dubstep",
    "Future Bass",
    "Chillout",
    "Trance",
    "Techno",
    "Drum&Bass",
    "Tropical House",
    "Disco",
    "Vaporwave",
    "Trip Hop",
    "Ambient",
    "Deep Pop Edm",
    "EDM Trap",
    "Future House",
    "China-Wave",
    "GuFeng Music",
    "Chinoiserie Rap",
    "Chinoiserie Electronic",
    "Hard Rock",
    "Psychedelic Rock",
    "Pop Rock",
    "Instrumental Rock",
    "Alternative Rock",
    "Indie Rock",
    "Post-Rock",
    "Lo-Fi",
    "J Rock",
    "Shoegazing",
    "Math Rock",
    "Surf Rock",
    "Progressive Rock",
    "Soft Rock",
    "Jazz Pop",
    "Jazz Fusion",
    "Bossa Nova",
    "Avant-Garde Jazz",
    "Swing",
    "Big Band",
    "Bop",
    "Post-Bop",
    "Smooth Jazz",
    "Cool Jazz",
    "Vocal Jazz",
    "Nu Jazz",
    "Free Jazz",
    "Trap Rap",
    "Old School",
    "R&B Rap",
    "Jazz Hip Hop",
    "Alternative Hip Hop",
    "Instrumental Hip Hop",
    "Pop Rap",
    "Hardcore Rap",
    "Comedy Hip Hop",
    "Hip House",
    "Chill Beats",
    "Chorus_AUDIO_GENRE",  # dedup
    "Chamber Music",
    "Symphony",
    "Funk",
    "Contemporary R&B",
    "Neo Soul",
    "Soul",
    "Pop Soul",
    "Black Metal",
    "Death Metal",
    "Glam Metal",
    "Grindcore",
    "Power Metal",
    "Progressive Metal",
    "Speed Metal",
    "DJ",
    "VinaHouse",
    "MC",
    "Gospel",
    "Holiday Music",
    "Christian Music",
    "Buddhist music",
    "Chinese Opera",
    "Traditional Chinese Folk",
    "Chinese Quyi",
    "Red Song",
    "Folk Pop",
    "Indie Folk",
    "Tango",
    "Reggaeton",
]

AUDIO_MOOD_V0 = [
    "Happy",
    "Cute/Playful",
    "Excited",
    "Funny",
    "Inspirational/Hopeful",
    "Sorrow/Sad",
    "Sentimental/Melancholic/Lonely",
    "Weird",
    "Thrilling/Suspenseful/Tense",
    "Angry/Aggressive",
    "Groovy/Funky",
    "Dynamic/Energetic",
    "Romantic",
    "Nostalgic/Memory",
    "Dreamy/Ethereal",
    "Healing",
    "Miss",
    "Chill",
    "Calm/Relaxing",
    "Shocking/magnificent/epic",
    "Mysterious",
    "No Mood",
]

AUDIO_SCENE_V0 = [
    "Halloween",
    "Christmas",
    "New Year",
    "Spring Festival",
    "Valentine's day",
    "Qi Xi",
    "Birthday",
    "Wedding",
    "Funeral",
    "Graduation",
    "National's Day",
    "Vlog/DailyLife",
    "Food",
    "Pet/Animals",
    "Beauty/Fashion",
    "Entertainment",
    "babies",
    "children",
    "Transition",
    "Anime",
    "Wake up",
    "Family time",
    "landscape/scenery",
    "Prank",
    "Timelapse",
    "Rainy Day",
    "Sunny Day",
    "Spring",
    "Summer",
    "Autumn",
    "Winter",
    "Evening",
    "Morning",
    "Beach",
    "Nightclub",
    "Coffee Shop",
    "Restaurants",
    "Lounge",
    "Campus",
    "Park",
    "Marketplace",
    "Universe",
    "Bar",
    "Theater/Concert hall",
    "Sport",
    "Dance",
    "Game",
    "Travel",
    "Focus",
    "Party",
    "Commute",
    "Roadtrip",
    "Meditation",
    "Relaxation",
    "Cleaning and Chores",
    "Running",
    "Date",
    "Danceable",
    "Flirt",
    "Other",
]

AUDIO_GENDER_V0 = [
    "Female",
    "Male",
    "Neutral",
    "Child",
    "Adult",
    "Chorus_AUDIO_GENDER",  # dedup
]

AUDIO_TIMBRE_V0 = [
    "Warm",
    "Ethereal",
    "Husky",
    "Deep",
    "Loud and sonorous",
    "Extreme",
    "Sharp",
    "Bright",
    "Sweet_AUDIO_TIMBRE",  # dedup
    "Powerful",
    "Sexy/Lazy",
    "Magnetic",
    "Cute_AUDIO_TIMBRE",  # dedup
    "Electrified voice",
]

AUDIO_GENRE_V1 = [
    "Pop",
    "Electronic",
    "Chinese Style",
    "Rock",
    "Jazz",
    "Hip Hop/Rap",
    "Classical",
    "R&B/Soul",
    "Metal",
    "Tuhai",
    "Childhood",
    "Devotional",
    "Chinese Tradition",
    "Easy Listening",
    "New Age",
    "8 Bit",
    "Folk",
    "Latin",
    "Alternative/Indie",
    "Epic",
    "BGM",
    "Chinese Pop",
    "Cantopop",
    "Taiwanese Pop",
    "Pop Folk",
    "Contemporary Pop",
    "Teen Pop",
    "Indie Pop",
    "Dream Pop",
    "City Pop",
    "Synth Pop",
    "Celtic Pop",
    "Dance Pop",
    "Deep Dance Pop",
    "Electropop",
    "Chamber Pop",
    "A cappella",
    "Country Pop",
    "EDM",
    "House",
    "Dubstep",
    "Future Bass",
    "Chillout",
    "Trance",
    "Techno",
    "Drum&Bass",
    "Tropical House",
    "Disco",
    "Vaporwave",
    "Trip Hop",
    "Ambient",
    "Deep Pop Edm",
    "EDM Trap",
    "Future House",
    "China-Wave",
    "GuFeng Music",
    "Chinoiserie Rap",
    "Chinoiserie Electronic",
    "Hard Rock",
    "Psychedelic Rock",
    "Pop Rock",
    "Instrumental Rock",
    "Alternative Rock",
    "Indie Rock",
    "Post-Rock",
    "Lo-Fi",
    "J Rock",
    "Shoegazing",
    "Math Rock",
    "Surf Rock",
    "Progressive Rock",
    "Soft Rock",
    "Jazz Pop",
    "Jazz Fusion",
    "Bossa Nova",
    "Avant-Garde Jazz",
    "Swing",
    "Big Band",
    "Bop",
    "Post-Bop",
    "Smooth Jazz",
    "Cool Jazz",
    "Vocal Jazz",
    "Nu Jazz",
    "Free Jazz",
    "Trap Rap",
    "Old School",
    "R&B Rap",
    "Jazz Hip Hop",
    "Alternative Hip Hop",
    "Instrumental Hip Hop",
    "Pop Rap",
    "Hardcore Rap",
    "Comedy Hip Hop",
    "Hip House",
    "Chill Beats",
    "Chorus_AUDIO_GENRE",  # dedup
    "Chamber Music",
    "Symphony",
    "Funk",
    "Contemporary R&B",
    "Neo Soul",
    "Soul",
    "Pop Soul",
    "Black Metal",
    "Death Metal",
    "Glam Metal",
    "Grindcore",
    "Power Metal",
    "Progressive Metal",
    "Speed Metal",
    "DJ",
    "VinaHouse",
    "MC",
    "Gospel",
    "Holiday Music",
    "Christian Music",
    "Buddhist music",
    "Chinese Opera",
    "Traditional Chinese Folk",
    "Chinese Quyi",
    "Red Song",
    "Folk Pop",
    "Indie Folk",
    "Tango",
    "Reggaeton",
    "Blues",
    "Country",
    "Punk",
    "Reggae",
    "Other genre"
]

AUDIO_MOOD_V1 = [
    "Happy",
    "Cute/Playful",
    "Excited",
    "Funny",
    "Inspirational/Hopeful",
    "Sorrow/Sad",
    "Sentimental/Melancholic/Lonely",
    "Weird",
    "Thrilling/Suspenseful/Tense",
    "Angry/Aggressive",
    "Groovy/Funky",
    "Dynamic/Energetic",
    "Romantic",
    "Nostalgic/Memory",
    "Dreamy/Ethereal",
    "Healing",
    "Miss",
    "Chill",
    "Calm/Relaxing",
    "Shocking/magnificent/epic",
    "Mysterious",
    "No Mood",
    "Sweet",
    "Other mood",
]

AUDIO_SCENE_V1 = [
    "Halloween",
    "Christmas",
    "New Year",
    "Spring Festival",
    "Valentine's day",
    "Qi Xi",
    "Birthday",
    "Wedding",
    "Funeral",
    "Graduation",
    "National's Day",
    "Vlog/DailyLife",
    "Food",
    "Pet/Animals",
    "Beauty/Fashion",
    "Entertainment",
    "babies",
    "children",
    "Transition",
    "Anime",
    "Wake up",
    "Family time",
    "landscape/scenery",
    "Prank",
    "Timelapse",
    "Rainy Day",
    "Sunny Day",
    "Spring",
    "Summer",
    "Autumn",
    "Winter",
    "Evening",
    "Morning",
    "Beach",
    "Nightclub",
    "Coffee Shop",
    "Restaurants",
    "Lounge",
    "Campus",
    "Park",
    "Marketplace",
    "Universe",
    "Bar",
    "Theater/Concert hall",
    "Sport",
    "Dance",
    "Game",
    "Travel",
    "Focus",
    "Party",
    "Commute",
    "Roadtrip",
    "Meditation",
    "Relaxation",
    "Cleaning and Chores",
    "Running",
    "Date",
    "Danceable",
    "Flirt",
    "Bedtime",
    "Broke up",
    "Dream",
    "Drive",
    "Friendship",
    "Love",
    "Yoga",
    "Other scene",
]

AUDIO_GENDER_V1 = [
    "Female",
    "Male",
    "Neutral",
    "Child",
    "Adult",
    "Chorus_AUDIO_GENDER",  # dedup
    "Unkonwn",
]

AUDIO_TIMBRE_V1 = [
    "Warm",
    "Ethereal",
    "Husky",
    "Deep",
    "Loud and sonorous",
    "Extreme",
    "Sharp",
    "Bright",
    "Sweet_AUDIO_TIMBRE",
    "Powerful",
    "Sexy/Lazy",
    "Magnetic",
    "Cute_AUDIO_TIMBRE",
    "Electrified voice",
]

AUDIO_GENRE_V2 = [
    "Pop",
    "Electronic",
    "Chinese Style",
    "Rock",
    "Jazz",
    "Hip Hop/Rap",
    "Classical",
    "R&B/Soul",
    "Metal",
    "Tuhai",
    "Childhood",
    "Devotional",
    "Chinese Tradition",
    "Easy Listening",
    "New Age",
    "8 Bit",
    "Folk",
    "Latin",
    "Alternative/Indie",
    "Epic",
    "BGM",
    "Chinese Pop",
    "Cantopop",
    "Taiwanese Pop",
    "Pop Folk",
    "Contemporary Pop",
    "Teen Pop",
    "Indie Pop",
    "Dream Pop",
    "City Pop",
    "Synth Pop",
    "Celtic Pop",
    "Dance Pop",
    "Deep Dance Pop",
    "Electropop",
    "Chamber Pop",
    "A cappella",
    "Country Pop",
    "EDM",
    "House",
    "Dubstep",
    "Future Bass",
    "Chillout",
    "Trance",
    "Techno",
    "Drum&Bass",
    "Tropical House",
    "Disco",
    "Vaporwave",
    "Trip Hop",
    "Ambient",
    "Deep Pop Edm",
    "EDM Trap",
    "Future House",
    "China-Wave",
    "GuFeng Music",
    "Chinoiserie Rap",
    "Chinoiserie Electronic",
    "Hard Rock",
    "Psychedelic Rock",
    "Pop Rock",
    "Instrumental Rock",
    "Alternative Rock",
    "Indie Rock",
    "Post-Rock",
    "Lo-Fi",
    "J Rock",
    "Shoegazing",
    "Math Rock",
    "Surf Rock",
    "Progressive Rock",
    "Soft Rock",
    "Jazz Pop",
    "Jazz Fusion",
    "Bossa Nova",
    "Avant-Garde Jazz",
    "Swing",
    "Big Band",
    "Bop",
    "Post-Bop",
    "Smooth Jazz",
    "Cool Jazz",
    "Vocal Jazz",
    "Nu Jazz",
    "Free Jazz",
    "Trap Rap",
    "Old School",
    "R&B Rap",
    "Jazz Hip Hop",
    "Alternative Hip Hop",
    "Instrumental Hip Hop",
    "Pop Rap",
    "Hardcore Rap",
    "Comedy Hip Hop",
    "Hip House",
    "Chill Beats",
    "Chorus_AUDIO_GENRE",
    "Chamber Music",
    "Symphony",
    "Funk",
    "Contemporary R&B",
    "Neo Soul",
    "Soul",
    "Pop Soul",
    "Black Metal",
    "Death Metal",
    "Glam Metal",
    "Grindcore",
    "Power Metal",
    "Progressive Metal",
    "Speed Metal",
    "DJ",
    "VinaHouse",
    "MC",
    "Gospel",
    "Holiday Music",
    "Christian Music",
    "Buddhist music",
    "Chinese Opera",
    "Traditional Chinese Folk",
    "Chinese Quyi",
    "Red Song",
    "Folk Pop",
    "Indie Folk",
    "Tango",
    "Reggaeton",
    "Blues",
    "Country",
    "Punk",
    "Reggae",
    "Other genre",
    "Boombap",
    "Chillwave",
    "Chinese Folk",
    "Contemporary Blues",
    "Contemporary Folk",
    "Country Blues",
    "Emo Rap",
    "English Folk",
    "Flamenco",
    "Gangsta Rap",
    "Hardstyle",
    "Heavy Metal",
    "HiNRG",
    "Indian Pop",
    "Italian Pop",
    "Japanese Folk",
    "J Pop",
    "KPop",
    "Latin Pop",
    "Low pop",
    "Melbourne Bounce",
    "New Chinese Folk",
    "New Wave",
    "Nostalgic Pop",
    "Pop Punk",
    "Rock Blues",
    "Russian Pop",
    "Samba",
    "Shi Dai Qu",
    "Soundtrack",
    "Thrash Metal",
    "Traditional Folk",
    "Traditional Jazz",
    "Vulgar Pop",
    "West Coast Hip Hop",
    "World Music",
]

AUDIO_MOOD_V2 = [
    "Happy",
    "Cute/Playful",
    "Excited",
    "Funny",
    "Inspirational/Hopeful",
    "Sorrow/Sad",
    "Sentimental/Melancholic/Lonely",
    "Weird",
    "Thrilling/Suspenseful/Tense",
    "Angry/Aggressive",
    "Groovy/Funky",
    "Dynamic/Energetic",
    "Romantic",
    "Nostalgic/Memory",
    "Dreamy/Ethereal",
    "Healing",
    "Miss",
    "Chill",
    "Calm/Relaxing",
    "Shocking/magnificent/epic",
    "Mysterious",
    "No Mood",
    "Other mood",
]

AUDIO_SCENE_V2 = [
    "Halloween",
    "Christmas",
    "New Year",
    "Spring Festival",
    "Valentine's day",
    "Qi Xi",
    "Birthday",
    "Wedding",
    "Funeral",
    "Graduation",
    "National's Day",
    "Vlog/DailyLife",
    "Food",
    "Pet/Animals",
    "Beauty/Fashion",
    "Entertainment",
    "babies",
    "children",
    "Transition",
    "Anime",
    "Wake up",
    "Family time",
    "landscape/scenery",
    "Prank",
    "Timelapse",
    "Rainy Day",
    "Sunny Day",
    "Spring",
    "Summer",
    "Autumn",
    "Winter",
    "Evening",
    "Morning",
    "Beach",
    "Nightclub",
    "Coffee Shop",
    "Restaurants",
    "Lounge",
    "Campus",
    "Park",
    "Marketplace",
    "Universe",
    "Bar",
    "Theater/Concert hall",
    "Sport",
    "Dance",
    "Game",
    "Travel",
    "Focus",
    "Party",
    "Commute",
    "Roadtrip",
    "Meditation",
    "Relaxation",
    "Cleaning and Chores",
    "Running",
    "Date",
    "Danceable",
    "Flirt",
    "Bedtime",
    "Broke up",
    "Dream",
    "Drive",
    "Friendship",
    "Love",
    "Yoga",
    "Other scene",
    "Mid-autumn Festival",
]

AUDIO_GENDER_V2 = AUDIO_GENDER_V1

AUDIO_TIMBRE_V2 = AUDIO_TIMBRE_V1

# Map SA genre mood scene tag to Multitag vocab
MAP_SA_TO_AUDIO_TAG = {
    "Angry": "Angry/Aggressive",
    "Calm": "Calm/Relaxing",
    "Chill": "Chill",
    "Cute_SA_MOOD": "Cute/Playful",  # dedup
    "Dynamic": "Dynamic/Energetic",
    "Excited": "Excited",
    "Funny": "Funny",
    "Happy": "Happy",
    "Healing": "Healing",
    "Inspirational": "Inspirational/Hopeful",
    "Lonely": "Sentimental/Melancholic/Lonely",
    "Miss/Memory": "Nostalgic/Memory",
    "Mysterious": "Mysterious",
    "Other": "No Mood",
    "Romantic": "Romantic",
    "Sorrow": "Sorrow/Sad",
    "Sweet_SA_MOOD": "Cute/Playful",  # dedup
    "Tense": "Thrilling/Suspenseful/Tense",
    "Weird": "Weird",
}

# Certain tags in the dataset should be replaced
SA_TAGS_SPECIAL_MAP = {
    "Нарру": "Happy",  # Confusable UTF code
    "Miss, Memory": "Miss/Memory",  # Comma is not nice for CSV
    "Miss,Memory": "Miss/Memory",
    "Pop,Chinese Style": "Chinese Style",
    # dedup
    "Cute": "Cute_SA_MOOD",
    "Sweet": "Sweet_SA_MOOD",
}

SA_TAGS_MOOD_SPECIAL_MAP = {
    "Other": "Other_SA_MOOD"
}

AUDIO_TAGS_GENRE_SPECIAL_MAP_V0 = {
    "Hip Hop": "Hip Hop/Rap",
    "Chorus": "Chorus_AUDIO_GENRE",  # dedup
}

AUDIO_TAGS_MOOD_SPECIAL_MAP_V0 = {
}

AUDIO_TAGS_SCENE_SPECIAL_MAP_V0 = {
    "Transition (卡点)": "Transition",
    "Theater / Concert hall": "Theater/Concert hall",
}

AUDIO_TAGS_GENDER_SPECIAL_MAP_V0 = {
    # dedup
    "chorus": "Chorus_AUDIO_GENDER",
    "Chorus": "Chorus_AUDIO_GENDER",
    # fix typo (if any)
    "chrous": "Chorus_AUDIO_GENDER",
    "Chrous": "Chorus_AUDIO_GENDER",
}

AUDIO_TAGS_GENRE_SPECIAL_MAP_V1 = {
    "Hip Hop": "Hip Hop/Rap",
    "Chorus": "Chorus_AUDIO_GENRE",  # dedup
}

AUDIO_TAGS_MOOD_SPECIAL_MAP_V1 = {
    "Angry": "Angry/Aggressive",
    "Calm": "Calm/Relaxing",
    "Cute": "Cute/Playful",
    "Dynamic": "Dynamic/Energetic",
    "Inspirational": "Inspirational/Hopeful",
    "Lonely": "Sentimental/Melancholic/Lonely",
    "Miss, Memory": "Miss",
    "Miss,Memory": "Miss",
    "Sorrow": "Sorrow/Sad",
    "Tense": "Thrilldding/Suspenseful/Tense",
    "Other": "Other mood",
}

AUDIO_TAGS_SCENE_SPECIAL_MAP_V1 = {
    "Transition (卡点)": "Transition",
    "Theater / Concert hall": "Theater/Concert hall",
    "Valentine's Day": "Valentine's day",
    "Cafe": "Coffee Shop",
    "Family": "Family time",
    "Other": "Other scene",
}

AUDIO_TAGS_GENDER_SPECIAL_MAP_V1 = {
    "female": "Female",
    "male": "Male",
    "child": "Child",
    "adult": "Adult",
    # dedup
    "chorus": "Chorus_AUDIO_GENDER",
    "Chorus": "Chorus_AUDIO_GENDER",
    # fix typo (if any)
    "chrous": "Chorus_AUDIO_GENDER",
    "Chrous": "Chorus_AUDIO_GENDER",
}

AUDIO_TAGS_GENRE_SPECIAL_MAP_V2 = {
    "Hip Hop": "Hip Hop/Rap",
    "Acapella": "A cappella",
    "dj": "DJ",
    "Glam Metal（Hair Metal，Pop Metal）": "Glam Metal",
    "Glam Metal\n（Hair Metal，Pop Metal）": "Glam Metal",
    "low pop": "Low pop",
    "Other": "Other genre",
    "tuhai": "Tuhai",
    "ChinaWave": "China-Wave",
    "DJDJ/remix": "DJ",
    "Progressive MetalProg Metal": "Progressive Metal",
    "Red Song/": "Red Song",
    "Rock BluesNew": "Rock Blues",
    "Chorus": "Chorus_AUDIO_GENRE",  # dedup
}

AUDIO_TAGS_MOOD_SPECIAL_MAP_V2 = {
    "Angry": "Angry/Aggressive",
    "Calm": "Calm/Relaxing",
    "Cute": "Cute/Playful",
    "Dynamic": "Dynamic/Energetic",
    "Inspirational": "Inspirational/Hopeful",
    "Lonely": "Sentimental/Melancholic/Lonely",
    "Miss, Memory": "Miss",
    "Miss,Memory": "Miss",
    "Sorrow": "Sorrow/Sad",
    "Tense": "Thrilldding/Suspenseful/Tense",
    "Other": "Other mood",
    "Sweet": "Cute/Playful", # Sweet also in vocal_timbre
}

AUDIO_TAGS_SCENE_SPECIAL_MAP_V2 = {
    "Transition (卡点)": "Transition",
    "Theater / Concert hall": "Theater/Concert hall",
    "Valentine's Day": "Valentine's day",
    "Cafe": "Coffee Shop",
    "Family": "Family time",
    "Other": "Other scene",
    "Babies": "babies",
    "Children": "children",
    "Dating": "Date",
    "Landscape/Scenery": "landscape/scenery",
    "National's Day ": "National's Day",
}

AUDIO_TAGS_GENDER_SPECIAL_MAP_V2 = AUDIO_TAGS_GENDER_SPECIAL_MAP_V1

AUDIO_TAGS_TIMBRE_SPECIAL_MAP = {  # for all versions
    # dedup
    "Cute": "Cute_AUDIO_TIMBRE",
    "Sweet": "Sweet_AUDIO_TIMBRE",
}

MACRO_STYLE_MAP = {
    "Pop": "Pop|||non-Sinking|Chinese",
    "Hip Hop/Rap": "Hip Hop/Rap|||non-Sinking|Chinese",
    "Chinese Style": "Chinese Style|||non-Sinking|Chinese",
    "Electronic": "Electronic|||non-Sinking|Chinese",
    "DJ": "DJ|||Sinking|Chinese",
    "Rock": "Rock|||non-Sinking|Chinese",
    "Folk": "Folk|||non-Sinking|Chinese",
    "R&B/Soul": "R&B/Soul|||non-Sinking|Chinese",
    "MC": "MC|||Sinking|Chinese",
    "Chinese Tradition": "Chinese Tradition|||non-Sinking|Chinese",
    "Jazz": "Jazz|||non-Sinking|Chinese",
    "Punk": "Punk|||non-Sinking|Chinese",
    "Reggae": "Reggae|||non-Sinking|Chinese",
    "empty": "||||"
}

MACRO_STYLE_MAP_GENRE_TO_MULTITAG = {
    "Pop": [
        "Pop,Chinese Pop|Nostalgic/Memory|Autumn,Rainy Day|Adult,Female|Warm", 
        # "Pop,Chinese Pop|Happy|Commute,Summer,Sunny Day|Adult,Female|Bright",
        # "Pop,Chinese Pop|Nostalgic/Memory|Roadtrip,Commute,Sport|Adult,Male|Powerful"
    ],
    "Hip Hop/Rap": [
        "Hip Hop/Rap,R&B Rap|Dynamic/Energetic|Dance,Commute,Roadtrip|Adult,Male|Powerful,Husky",
        # "Hip Hop,Trap Rap|Dynamic/Energetic|Dance,Commute,Sport|Adult,Male|Powerful"
    ],
    "Chinese Style": [
        "Chinese Style|Nostalgic/Memory|Autumn,Rainy Day|Adult,Male|Warm",
        # "Chinese Style|Sorrow/Sad|Autumn,Rainy Day|Adult,Female|Warm"
    ],
    "Electronic": [
        "Electropop,Pop|Dynamic/Energetic|Dance,Nightclub,Party|Adult,Female|Sexy/Lazy",
        # "Electronic|Dynamic/Energetic|Dance,Party,Sport|Adult,Female|Electrified voice"
    ],
    "DJ": [
        "DJ,Tuhai|Dynamic/Energetic|Dance,Roadtrip,Summer|Adult,Male|Husky",
        # "DJ,Tuhai|Dynamic/Energetic|Dance,Roadtrip,Summer|Adult,Female|Warm"
    ],
    "Rock": [
        "Rock|Excited|Commute,Sport|Adult,Male|Husky",
        # "Rock|Excited|Running,Roadtrip,Party,Sport,Sunny Day,Summer,Commute|Adult,Male|Magnetic"
    ],
    "Folk": [
        "Folk|Nostalgic/Memory|Autumn,Rainy Day|Adult,Female|Sexy/Lazy",
        # "Pop,Pop Folk|Romantic|Commute,Sunny Day,Travel|Adult,Male|Warm"
    ],
    "R&B/Soul": [
        "R&B/Soul|Chill|Relaxation,Bar,Date|Adult,Female|Magnetic",
        # "R&B/Soul|Groovy/Funky,Romantic|Roadtrip,Valentine's day,Sunny Day,Commute,Relaxation|Adult,Male|Sexy/Lazy,Husky"
    ],
    "MC": [
        "MC,Tuhai,DJ|Dynamic/Energetic|Dance,Summer|Adult,Male|Sharp,Bright",
        # "MC|||Adult,Female|Bright",
    ],
    "empty": ["||||"]
}

MACRO_STYLE_MAP_GENRE_TO_MULTITAG_V3 = {
    "Pop": [
        "Pop,Chinese Pop|Nostalgic/Memory|Autumn,Rainy Day|Adult,Female|Warm", 
    ],
    "Hip Hop/Rap": [
        "Hip Hop/Rap,R&B Rap|Dynamic/Energetic|Dance,Commute,Roadtrip|Adult,Male|Powerful,Husky",
    ],
    "Chinese Style": [
        "Chinese Style|Nostalgic/Memory|Autumn,Rainy Day|Adult,Male|Warm",
    ],
    "Electronic": [
        #"Electropop,Pop|Dynamic/Energetic|Dance,Nightclub,Party|Adult,Female|Sexy/Lazy",
        "Electronic,Deep Pop Edm|Sorrow/Sad|Nightclub,Danceable,Summer|Female,Adult|Bright",
    ],
    "DJ": [
        "DJ,Tuhai|Dynamic/Energetic|Dance,Roadtrip,Summer|Adult,Male|Husky",
    ],
    "Rock": [
        "Rock|Excited|Commute,Sport|Adult,Male|Husky",
    ],
    "Folk": [
        "Folk|Nostalgic/Memory|Autumn,Rainy Day|Adult,Female|Sexy/Lazy",
    ],
    "R&B/Soul": [
        "R&B/Soul|Chill|Relaxation,Bar,Date|Adult,Female|Magnetic",
    ],
    #"MC": [
    #    #"MC,Tuhai,DJ|Dynamic/Energetic|Dance,Summer|Adult,Male|Sharp,Bright",
    #    "MC,Tuhai|Dynamic/Energetic|Dance,Summer|Adult,Male|Sharp,Bright",
    #],
    "Chinese Tradition": [
        "Chinese Tradition|Inspirational/Hopeful|National's Day,Sunny Day|Adult,Male|Loud and sonorous",
    ],
    "Jazz": [
        "Jazz|Sorrow/Sad|Spring,Travel|Female,Adult|Magnetic",
    ],
    "Punk": [
        "Punk|Sorrow/Sad|Sunny Day,Summer|Female,Adult|Bright",
    ],
    "Reggae": [
        "Reggae|Sorrow/Sad|Love,Bar|Female,Adult|Warm",
    ],
    "empty": ["||||"]
}

COMBO_GENRE_TO_MULTITAG_V3 = {
    "Pop": [
        "Pop,Chinese Pop",
        "Pop,Nostalgic Pop",
        "Pop,Electropop",
        "Pop,Indie Pop",
        "Pop,Dance Pop",
        "Pop,Synth Pop",
        "Pop,Chinese Pop,Hip Hop/Rap",
        "Pop,Chinese Pop,Folk",
        "Pop,Chinese Pop,Chinese Style",
        "Pop,Chinese Pop,Rock",
        "Pop,Pop Folk",
    ],
    "Chinese Style": [
        "Chinese Style,China-Wave",
        "Chinese Style,Chinese Pop,Pop",
        "Chinese Style,GuFeng Music",
        "Chinese Style,Chinoiserie Electronic",
        "Chinese Style,Chinoiserie Rap",
    ],
    "Rock": [
        "Rock,Pop Rock",
        "Rock,Indie Rock",
        "Rock,Chinese Pop,Pop",
    ],
    "Hip Hop/Rap": [
        "Hip Hop/Rap,Pop Rap",
        "Hip Hop/Rap,R&B Rap",
        "Hip Hop/Rap,Trap Rap",
        "Hip Hop/Rap,Chinese Pop,Pop",
        "Hip Hop/Rap,Boombap",
        "Hip Hop/Rap,Old School",
        "Hip Hop/Rap,R&B/Soul",
        "Hip Hop/Rap,Jazz Hip Hop",
        "Hip Hop/Rap,Emo Rap",
    ],
    "R&B/Soul": [
        "R&B/Soul,Contemporary R&B",
        "R&B/Soul,Funk",
        "R&B/Soul,Chinese Pop,Pop",
        "R&B/Soul,Pop Soul",
        "R&B/Soul,Hip Hop/Rap",
    ],
    "DJ": [
        "Tuhai,DJ",
    ],
    "Folk": [
        "Folk,Chinese Folk",
        "Folk,Folk Pop",
    ],
    "Chinese Tradition": [
        "Chinese Tradition,New Chinese Folk",
        "Chinese Tradition,Traditional Chinese Folk",
        "Chinese Tradition,Red Song",
    ],
    "Electronic": [
        "Electronic,Deep Pop Edm",
        "Electronic,EDM",
        "Electronic,Pop,Chinese Pop",
    ],
    "Jazz": [
        "Jazz,Jazz Pop",
        "Jazz,Bossa Nova",
    ],
    "Punk": [
        "Punk,Pop Punk",
    ],
    "Reggae": [
        "Reggae,Reggae",
    ],
}


COMBO_MOOD_TO_MULTITAG_V3 = [
    "Nostalgic/Memory",
    "Sorrow/Sad",
    "Dynamic/Energetic",
    "Happy",
    "Romantic",
    "Miss",
    "Excited",
    "Inspirational/Hopeful",
    "Sentimental/Melancholic/Lonely",
    "Chill",
    "Cute/Playful",
    "Groovy/Funky",
    "Nostalgic/Memory,Miss",
    "Dynamic/Energetic,Happy",
    "Healing",
    "Angry/Aggressive",
    "Nostalgic/Memory,Romantic",
    "Dynamic/Energetic,Sorrow/Sad",
    "Angry/Aggressive,Dynamic/Energetic",
    "Happy,Romantic",
    "Groovy/Funky,Happy",
    "Dynamic/Energetic,Romantic",
    "Nostalgic/Memory,Dynamic/Energetic",
    "Chill,Romantic",
    "Nostalgic/Memory,Sorrow/Sad",
    "Groovy/Funky,Dynamic/Energetic",
    "Calm/Relaxing",
    "Cute/Playful,Dynamic/Energetic",
    "Nostalgic/Memory,Sentimental/Melancholic/Lonely",
    "Cute/Playful,Romantic",
    "Groovy/Funky,Romantic",
    "Excited,Inspirational/Hopeful",
    "Miss,Sentimental/Melancholic/Lonely",
    "Miss,Sorrow/Sad",
    "Funny",
    "Miss,Romantic",
    "Excited,Dynamic/Energetic",
    "Cute/Playful,Happy",
    "Groovy/Funky,Sorrow/Sad",
    "Sentimental/Melancholic/Lonely,Dynamic/Energetic",
    "Inspirational/Hopeful,Dynamic/Energetic",
    "Excited,Romantic",
    "Excited,Happy",
    "Nostalgic/Memory,Groovy/Funky",
    "Nostalgic/Memory,Excited",
    "Nostalgic/Memory,Chill",
]


COMBO_SCENE_TO_MULTITAG_V3 = [
    "Rainy Day,Autumn,Love,Broke up",
    "Rainy Day,Autumn,Love",
    "Rainy Day,Autumn",
    "Commute,Sunny Day,Love",
    "Roadtrip,Commute,Sunny Day",
    "Rainy Day,Love,Broke up",
    "Summer,Commute,Sunny Day",
    "Commute,Spring,Sunny Day",
    "Autumn,Love,Broke up",
    "Commute,Sunny Day",
    "Summer,Sunny Day,Sport",
    "Commute,Sunny Day,Dance",
    "Valentine's day,Sunny Day,Love",
    "Roadtrip,Commute,Sport",
    "Rainy Day,Autumn,Winter",
    "Summer,Roadtrip,Commute,Sunny Day",
    "Travel,Commute,Sunny Day",
    "Valentine's day,Love,Date",
    "Roadtrip,Commute",
    "Valentine's day,Sunny Day,Love,Date",
    "Autumn,Winter",
    "Commute,Autumn,Love",
    "Summer,Roadtrip,Dance",
    "Commute,Dance,Sport",
    "Rainy Day,Winter,Love,Broke up",
    "Summer,Roadtrip,Sunny Day",
    "Rainy Day,Autumn,Evening",
    "Summer,Roadtrip,Commute",
    "Roadtrip,Commute,Love",
    "Sunny Day,Love,Spring",
    "Summer,Roadtrip,Sport",
    "Commute,Rainy Day,Autumn",
    "Commute,Sunny Day,Spring",
    "Commute,Roadtrip,Sunny Day,Love",
    "Summer,Sunny Day,Love",
    "Summer,Sunny Day,Dance",
    "Roadtrip,Commute,Dance",
    "Summer,Commute,Love",
    "Sunny Day,Love,Dance",
    "Party,Dance,Sport",
    "Danceable,Nightclub,Dance",
    "Autumn,Commute",
    "Sunny Day,Love,Date",
    "Party,Commute,Dance",
    "Summer,Commute,Dance",
    "Commute,Dance",
    "Summer,Roadtrip,Sunny Day,Sport",
    "Evening,Roadtrip,Commute",
    "Commute,Sunny Day,Sport",
    "Summer,Party,Sunny Day",
    "Summer,Dance,Sport",
    "Commute,Love,Dance",
    "Summer,Sunny Day",
    "Evening,Autumn,Rainy Day",
    "Commute,Sunny Day,Love,Dance",
    "Commute,Autumn",
    "Summer,Running,Sunny Day",
    "Summer,Dance",
    "Party,Sunny Day,Dance",
    "Game,Dance,Sport",
    "Evening,Commute,Relaxation",
]


COMBO_VOICE_GENDER_TO_MULTITAG_V3 = ["Female,Adult", "Male,Adult"]

COMBO_VOICE_TIMBRE_TO_MULTITAG_V3 = [
    "Warm",
    "Bright",
    "Bright,Warm",
    "Husky",
    "Magnetic",
    "Powerful",
    "Sexy/Lazy",
    "Sweet_AUDIO_TIMBRE",
    "Husky,Warm",
    "Husky,Bright",
    "Sharp",
    "Magnetic,Warm",
    "Warm,Sexy/Lazy",
    "Bright,Magnetic",
    "Husky,Powerful",
    "Warm,Sweet_AUDIO_TIMBRE",
    "Loud and sonorous",
    "Bright,Sweet_AUDIO_TIMBRE",
    "Bright,Powerful",
    "Deep",
    "Bright,Sexy/Lazy",
    "Warm,Powerful",
    "Magnetic,Sexy/Lazy",
    "Husky,Sharp",
    "Sharp,Bright",
    "Husky,Magnetic",
    "Ethereal",
    "Warm,Ethereal",
    "Sharp,Warm",
    "Bright,Loud and sonorous",
    "Cute_AUDIO_TIMBRE",
    "Husky,Deep",
    "Bright,Ethereal",
    "Husky,Sexy/Lazy",
    "Sweet_AUDIO_TIMBRE,Sexy/Lazy",
    "Sharp,Electrified voice",
    "Deep,Warm",
    "Sharp,Powerful",
    "Deep,Magnetic",
    "Husky,Bright,Warm",
    "Sharp,Sexy/Lazy",
    "Loud and sonorous,Warm",
    "Magnetic,Powerful",
    "Bright,Sharp",
    "Deep,Bright",
    "Cute_AUDIO_TIMBRE,Warm",
    "Bright,Warm,Powerful",
    "Bright,Magnetic,Warm",
]


MAP_SUB_GENRE_2_SA_GENRE20 = {
    'Hip Hop': ['Hip Hop/Rap'],
    'Chinese Pop': ['Pop'],
    'R&B Rap': ['Hip Hop/Rap'],
    'Tuhai': ['DJ', 'MC'],
    'Trap Rap': ['Hip Hop/Rap'],
    'Pop Rock': ['Rock'],
    'Folk Pop': ['Folk'],
    'Pop Rap': ['Hip Hop/Rap'],
    'House': ['Electronic'],
    'Old School': ['Hip Hop/Rap'],
    'Funk': ['R&B/Soul'],
    }


SA_CAT_VOCAB = [SA_GENRE20, SA_MOOD19, SA_THEME33, SA_SINKING, SA_LANG]
AUDIO_CAT_VOCAB_V0 = [AUDIO_GENRE_V0, AUDIO_MOOD_V0, AUDIO_SCENE_V0, AUDIO_GENDER_V0, AUDIO_TIMBRE_V0]
AUDIO_CAT_VOCAB_V1 = [AUDIO_GENRE_V1, AUDIO_MOOD_V1, AUDIO_SCENE_V1, AUDIO_GENDER_V1, AUDIO_TIMBRE_V1]
AUDIO_CAT_VOCAB_V2 = [AUDIO_GENRE_V2, AUDIO_MOOD_V2, AUDIO_SCENE_V2, AUDIO_GENDER_V2, AUDIO_TIMBRE_V2]

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

# Deprecated
_VOCAB_MAP = {
    "Zh": CHINESE_CAT_VOCAB,
    "SA": SA_CAT_VOCAB,
    "AudioV0": AUDIO_CAT_VOCAB_V0,  # for model V1.5, initial taxonomy 
    "Audio": AUDIO_CAT_VOCAB_V1,    # unified vocab to support initial taxonomy and SA tag. # TODO rename
    "AudioV2": AUDIO_CAT_VOCAB_V2,  # unified vocab to support new taxonomy and SA tag
}

_UNIFIED_VOCAB_MAP = {
    "SA_unified": VOCAB2ID_SA,
    # backward compatible
    "Audio_unified": VOCAB2ID_AUDIO_V0,
    "Mix_unified": VOCAB2ID_MIX_V0,
    # V0 (audio tag v0)
    "Audio_unified_v0": VOCAB2ID_AUDIO_V0,
    "Mix_unified_v0": VOCAB2ID_MIX_V0,
    # V1 (audio tag v1, v2, v3)
    "Audio_unified_v1": VOCAB2ID_AUDIO_V1,
    "Mix_unified_v1": VOCAB2ID_MIX_V1,
}

def get_categorical_vocab(vocab_type: str) -> Tuple[Dict[str, int], int]:
    if vocab_type in _UNIFIED_VOCAB_MAP:
        _vocab2id = _UNIFIED_VOCAB_MAP[vocab_type]
        # Hard-code num_categories here. It's supposed to be decided by dataloader.
        # The lookup table does not know how many categories the dataloader would use.
        return _vocab2id.to_dict(), 5
    logger.warning("Non-unified vocab is deprecated.")
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

# ========================================================================

_TEMPO_LABEL_RANGES = {
    "Grave": (0, 40),
    "Largo": (40, 60),
    "Adagio": (60, 70),
    "Andante": (70, 90),
    "Moderato": (90, 110),
    "Allegro": (110, 140),
    "Vivace": (140, 160),
    "Presto": (160, 200),
}

TEMPO_RANGE = (0, 200)
TEMPO_LABELS = [""] + list(_TEMPO_LABEL_RANGES.keys())
TEMPO_LABEL_ID_MAP = {label: i for i, label in enumerate(TEMPO_LABELS)}
ID_TEMPO_LABEL_MAP = {v: k for k, v in TEMPO_LABEL_ID_MAP.items()}


def tempo_to_label(tempo: Optional[int]) -> str:
    if tempo is None:
        return ""
    for label, (low, high) in _TEMPO_LABEL_RANGES.items():
        if low <= tempo < high:
            return label
    return ""


# ========================================================================

_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_KEY_TYPES = ["Maj", "Min"]

KEYS = ["N"] + [root + ":" + key_type for root in _ROOTS for key_type in _KEY_TYPES]
KEY_ID_MAP = {label: i for i, label in enumerate(KEYS)}
ID_KEY_MAP = {v: k for k, v in KEY_ID_MAP.items()}

# ========================================================================

def _split_and_check_style_text(text: str) -> Tuple[str, str, str]:
    genre, mood, gender = text.split('|')
    if mood and mood not in AUDIO_MOOD_V2 and mood not in SA_MOOD19:
        raise ValueError(f"Unsupported mood {mood}")
    if gender and gender not in ["Male", "Female"]:
        raise ValueError(f"Unsupported gender {gender}")
    return genre, mood, gender


def rewrite_style_input_to_multi_tag_combo_v3(text: str) -> Tuple[str, str, str, int]:
    multitag_list = ['', '', '', '', '']
    genre, mood, gender = _split_and_check_style_text(text)
    # default genre
    if (not genre) or (genre not in COMBO_GENRE_TO_MULTITAG_V3):
        genre = "Pop"
    genre = random.choice(COMBO_GENRE_TO_MULTITAG_V3[genre])
    multitag_list[0] = genre
    # default mood
    if mood:
        mood = MAP_SA_TO_AUDIO_TAG[mood] if mood in SA_MOOD19 else mood
    else:
        mood = random.choice(COMBO_MOOD_TO_MULTITAG_V3)
    multitag_list[1] = mood
    # default scene
    multitag_list[2] = random.choice(COMBO_SCENE_TO_MULTITAG_V3)
    # default gender
    if not gender:
        gender = random.choice(COMBO_VOICE_GENDER_TO_MULTITAG_V3)
    if 'Adult' not in gender:
        gender = 'Adult,' + gender
    multitag_list[3] = gender
    # default timbre
    multitag_list[4] = random.choice(COMBO_VOICE_TIMBRE_TO_MULTITAG_V3)
    multitag = "|".join(multitag_list)
    
    gender = "Male" if "Male" in multitag_list[3] else "Female"
    speaker_id = ARTIST_ID_MAP_V2[gender]

    key, tempo_label = "N", ""
    print('Original style input : ' + text)
    print('Rewrite style input : ' + multitag)
    return multitag, key, tempo_label, speaker_id

def rewrite_style_input_to_multi_tag_v3(text: str) -> Tuple[str, str, str, int]:
    genre, mood, gender = _split_and_check_style_text(text)
    # default genre
    if (not genre) or (genre not in MACRO_STYLE_MAP_GENRE_TO_MULTITAG_V3):
        genre = "Pop"
    multitag = random.choice(MACRO_STYLE_MAP_GENRE_TO_MULTITAG_V3[genre])
    multitag_list = multitag.split('|')
    # default mood
    if mood:
        mood = MAP_SA_TO_AUDIO_TAG[mood] if mood in SA_MOOD19 else mood
        multitag_list[1] = mood
    # default gender
    if gender and ('Adult' not in gender):
        multitag_list[3] = 'Adult,' + gender
    multitag = "|".join(multitag_list)    

    gender = "Male" if "Male" in multitag_list[3] else "Female"
    speaker_id = ARTIST_ID_MAP_V2[gender]

    key, tempo_label = "N", ""
    print('Original style input : ' + text)
    print('Rewrite style input : ' + multitag)
    return multitag, key, tempo_label, speaker_id


def rewrite_style_input_to_multi_tag(text: str) -> Tuple[str, str, str, int]:
    genre, mood, gender = _split_and_check_style_text(text)

    # TODO(yilin) rethink the style input rewritting logic here.
    if (not genre) or (genre not in MACRO_STYLE_MAP_GENRE_TO_MULTITAG):  # default genre
        genre = "Pop"
    # Based on genre to get a complete multi-tag + key + tempo_label expansion.
    multitag = random.choice(MACRO_STYLE_MAP_GENRE_TO_MULTITAG[genre])
    # If user specifies mood and gender, overwrite the multitag field.
    multitag_list = multitag.split('|')
    if mood:
        mood = MAP_SA_TO_AUDIO_TAG[mood] if mood in SA_MOOD19 else mood # Support SA tag input.
        multitag_list[1] = mood
    if gender:
        multitag_list[3] = 'Adult,' + gender
    multitag = "|".join(multitag_list)    

    gender = "Male" if "Male" in multitag_list[3] else "Female"
    speaker_id = ARTIST_ID_MAP_V2[gender]

    # TODO(yilin) Add key, tempo_label expansion
    key, tempo_label = "N", ""
    print('Original style input : ' + text)
    print('Rewrite style input : ' + multitag)
    return multitag, key, tempo_label, speaker_id


def rewrite_style_input_to_sa_tag(text: str) -> Tuple[str, str, str, int]:
    genre, mood, gender = _split_and_check_style_text(text)

    # Obtain SA style text
    if not genre:  # default genre
        genre = "Pop"
    satag = MACRO_STYLE_MAP[genre]
    satag_list = satag.split('|')
    if mood:  # override mood if given
        if mood in MACRO_STYLE_MAP_MOOD2_SA_MOOD19:
            satag_list[1] = MACRO_STYLE_MAP_MOOD2_SA_MOOD19[mood]
        else:
            satag_list[1] = mood
    satag = "|".join(satag_list)    

    gender = gender if gender else "zh_empty"
    speaker_id = ARTIST_ID_MAP_V2[gender]

    # TODO(yilin) Add key, tempo_label expansion
    key, tempo_label = "N", ""
    print('Original style input : ' + text)
    print('Rewrite style input : ' + satag)
    return satag, key, tempo_label, speaker_id
