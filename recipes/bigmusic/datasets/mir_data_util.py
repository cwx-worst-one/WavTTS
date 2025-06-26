from functools import reduce
import logging
import operator
from typing import List, Dict, Tuple, Optional
import random

from recipes.bigmusic.datasets.utils.zh_vocab import (
    VOCAB2ID_SA,
    VOCAB2ID_AUDIO_V3,
    VOCAB2ID_MIX_V3,
    VOCAB2ID_AUDIO_V4,
    VOCAB2ID_MIX_V4,
    VOCAB2ID_INST_V1,
)

from recipes.bigmusic.datasets.combo_audio_tags import (
    COMBO_GENRE_TO_MULTITAG_V4,
    COMBO_GENRE_TO_INSTRUMENTS_V4,
    COMBO_GENRE_TO_CANT_MULTITAG_V5,
    SUB_GENRE_MAPPING
)

logger = logging.getLogger(__file__)


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
    # Inst dropout
    "inst_empty": 38,
    # Cant dropout
    "cant_empty": 39,
    # Voice tags
    "Neutral": 43,
    "Multiple": 44,
    "Chorus_AUDIO_GENDER": 45,
    "Chorus": 45,
    "Child": 46,
    "Male_Female": 47,
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

# =======================================================================
 
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

UNKNOWN_GENRE_MAP_SA_GENRE20 = {
    'Surf Rock': 'Rock',
}

MACRO_STYLE_MAP_MOOD2_SA_MOOD19 = {
    'Calm/Relaxing': 'Calm',
    # 'Nostalgic/Memory': 'Miss,Memory',
    'Nostalgic/Memory': 'Miss/Memory',
    'Sorrow/Sad': 'Sorrow',
    'Inspirational/Hopeful': 'Inspirational',
    'Dynamic/Energetic': 'Dynamic',
    # 'Miss': 'Miss,Memory',
    'Miss': 'Miss/Memory',
    'Sentimental/Melancholic/Lonely': 'Lonely',
    'Dreamy/Ethereal': 'Mysterious',
    'Groovy/Funky': 'Dynamic',
    'Angry/Aggressive': 'Angry',
    'Cute/Playful': 'Cute_SA_MOOD',
    'Shocking/magnificent/epic': '',
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

SA_LANG43 = [
    'Chinese',
    'Cantonese',
    'Taiwanese',
    'English',
    'Hindi',
    'Punjabi',
    'Indonesian',
    'Portuguese',
    'Telugu',
    'Tamil',
    'Malayalam',
    'Kannada',
    'Spanish',
    'Japanese',
    'Korean',
    'German',
    'French',
    'Italian',
    'Russian',
    'Arabic',
    'Non-vocal',
    'Malay',
    'Thai',
    'Gujarati',
    'Bengali',
    'Bhojpuri',
    'Marathi',
    'Turkish',
    'Dutch',
    'Finnish',
    'Greek',
    'Norwegian',
    'Swedish',
    'Polish',
    'Czech',
    'Javanese',
    'Rajasthani',
    'Sanskrit',
    'Sundanese',
    'Minangkabau',
    'Ambonese Malay',
    'Vietnamese',
    'Chinese Dialects',
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

AUDIO_GENRE_V3 = [
    "8 Bit",
    "Acapella",
    "A cappella",
    "Acid Jazz",
    "African Folk",
    "African Pop",
    "Alternative Hip Hop",
    "Alternative/Indie",
    "Alternative Rock",
    "Ambient",
    "Arabic Folk",
    "Arabic Pop",
    "Avant-Garde Jazz",
    "Baroque",
    "Bass House",
    "Bebop",
    "BGM",
    "Bhangra",
    "Big Band",
    "Black Metal",
    "Blue Grass",
    "Blues",
    "Blues Rock",
    "Boogie Woogie",
    "Boombap",
    "Bop",
    "Bossa Nova",
    "Brazilian Funk Style",
    "Breakbeat",
    "Brit Pop",
    "Bubblegum Bass",
    "Buddhist music",
    "Cantopop",
    "Celtic Pop",
    "Chamber Music",
    "Chamber Pop",
    "Childhood",
    "Chill Beats",
    "Chillout",
    "Chillwave",
    "China-Wave",
    "Chinese Ballad Pop",
    "Chinese Folk",
    "Chinese Opera",
    "Chinese Pop",
    "Chinese Quyi",
    "Chinese Style",
    "Chinese Tradition",
    "Chinese Traditional Instrumental Music",
    "Chinoiserie Electronic",
    "Chinoiserie Rap",
    "Chorus_AUDIO_GENRE",
    "Christian Music",
    "City Pop",
    "Classical",
    "Classical period",
    "Classic R&B / Soul",
    "Comedy Hip Hop",
    "Contemporary Blues",
    "Contemporary classical music",
    "Contemporary Folk",
    "Contemporary Pop",
    "Contemporary R&B",
    "Cool Jazz",
    "Country",
    "Country Blues",
    "Country Folk",
    "Country Pop",
    "Country Rock",
    "Dancehall",
    "Dance Pop",
    "Dance Punk",
    "Dark Ambient",
    "Deathcore",
    "Death Metal",
    "Deep Dance Pop",
    "Deep Pop Edm",
    "Devotional",
    "Disco",
    "DJ",
    "Doom Metal",
    "Doo-Wop",
    "Downtempo",
    "Dream Pop",
    "Drill Rap",
    "Drum&Bass",
    "Dubstep",
    "East Coast Hip Hop",
    "Easy Listening",
    "EDM",
    "EDM Trap",
    "Electronic",
    "Electropop",
    "Emo Punk",
    "Emo Rap",
    "English Folk",
    "Enka",
    "Epic",
    "Fado",
    "Flamenco",
    "Folk",
    "Folk Metal",
    "Folk Pop",
    "Folktronica",
    "Free Jazz",
    "French Folk",
    "French Pop",
    "Funk",
    "Future Bass",
    "Future House",
    "Gangsta Rap",
    "German Pop",
    "Glam Metal",
    "Glam Metal（Hair Metal，Pop Metal）",
    "Glam Metal\n（Hair Metal，Pop Metal）",
    "Glitch",
    "Gospel",
    "Gothic Metal",
    "Grindcore ",
    "Grindcore",
    "Grunge Rock",
    "GuFeng Music",
    "Gypsy Jazz",
    "Hard Bop",
    "Hardcore",
    "Hardcore Punk ",
    "Hardcore Rap",
    "Hard Rock",
    "Hardstyle",
    "Heavy Metal",
    "Hi-NRG",
    "HiNRG",
    "Hip Hop",
    "Hip Hop/Rap",
    "Hip House",
    "Holiday Music",
    "House",
    "IDM",
    "Impressionism",
    "Indian Pop",
    "Indie Folk",
    "Indie Pop",
    "Indie Rock",
    "Industrial Metal",
    "Instrumental Hip Hop",
    "Instrumental Rock",
    "Irish Folk",
    "Italian Pop",
    "Japanese Folk",
    "Japanese Traditional Music",
    "Jazz",
    "Jazz Blues",
    "Jazz Fusion",
    "Jazz Hip Hop",
    "Jazz Pop",
    "J Pop",
    "J Rock",
    "Korean Folk",
    "Korea Trot",
    "K-Pop",
    "KPop",
    "Latin",
    "Latin Pop",
    "Latin Rock",
    "Lo-Fi",
    "Lo-Fi House",
    "Lo-Fi Rock",
    "Low pop",
    "Math Rock",
    "MC",
    "Melbourne Bounce",
    "Metal",
    "Metalcore",
    "Midwest Hip Hop",
    "Modernism",
    "Mongolian Folk Songs",
    "Neo Funk",
    "Neo Soul",
    "New Age",
    "New Chinese Folk",
    "New Wave ",
    "New Wave",
    "Nostalgic Pop",
    "No Wave",
    "Nu Jazz",
    "Nu Metal",
    "Old School",
    "Other",
    "Other genre",
    "Pop",
    "Pop Folk",
    "Pop Punk",
    "Pop Rap",
    "Pop Rock",
    "Pop Soul",
    "Post-Bop",
    "Post-hardcore",
    "Post-Punk",
    "Post-Rock",
    "Power Metal",
    "Progressive Metal",
    "Progressive Metal（Prog Metal)",
    "Progressive R&B",
    "Progressive Rock",
    "Psychedelic Rock",
    "Punk",
    "Ragtime",
    "Rap Metal",
    "R&B Rap",
    "R&B/Soul",
    "Red Song",
    "Reggae",
    "Reggaeton",
    "Rock",
    "Rock Blues",
    "Romantic music",
    "Rumba",
    "Russian Pop",
    "Samba",
    "Score",
    "Shi Dai Qu",
    "Shima Uta",
    "Shoegaze-Rock",
    "Shoegazing",
    "Ska Punk",
    "Smooth Jazz",
    "Soft Rock",
    "Son cubano",
    "Soul",
    "Soundtrack",
    "Southern Hip Hop",
    "Speed Metal",
    "Surf Rock",
    "Swing",
    "Symphony",
    "Synth Pop",
    "Synthwave",
    "Taiwanese Pop",
    "Tango",
    "Techno",
    "Teen Pop",
    "Thrash Metal",
    "Traditional Blues",
    "Traditional Chinese Folk",
    "Traditional Folk",
    "Traditional Jazz",
    "Trance",
    "Trap Rap",
    "Trip Hop",
    "Tropical House",
    "Tuhai",
    "Turkish Pop",
    "Uygur Folk Songs",
    "Vaporwave",
    "VinaHouse",
    "Vocal Jazz",
    "Vulgar Pop",
    "West Coast Hip Hop",
    "Worldbeat",
    "World Music",
    "Yodeling",
]

AUDIO_EXTRA_V3 = [
    "Tuhai_AUDIO_EXTRA",
    "Soundtrack_AUDIO_EXTRA",
    "Nostalgic",
    "Lo-fi",
]

AUDIO_MOOD_V3 = [
    "Angry",
    "Angry/Aggressive",
    "Calm",
    "Calm/Relaxing",
    "Chill",
    "Cute/Playful",
    "Cute_SA_MOOD",
    "Dreamy/Ethereal",
    "Dynamic",
    "Dynamic/Energetic",
    "Excited",
    "Funny",
    "Groovy/Funky",
    "Happy",
    "Healing",
    "Inspirational",
    "Inspirational/Hopeful",
    "Lonely",
    "Miss",
    "Miss/Memory",
    "Mysterious",
    "No Mood",
    "Nostalgic/Memory",
    "Other mood",
    "Other_SA_MOOD",
    "Romantic",
    "Sentimental/Melancholic/Lonely",
    "Shocking/magnificent/epic",
    "Sorrow",
    "Sorrow/Sad",
    "Sweet_SA_MOOD",
    "Tense",
    "Thrilling/Suspenseful/Tense",
    "Weird",
]

AUDIO_SCENE_V3 = [
    "Anime",
    "Autumn",
    "babies",
    "Bar",
    "Beach",
    "Beauty/Fashion",
    "Bedtime",
    "Birthday",
    "Broke up",
    "Cafe",
    "Campus",
    "children",
    "Christmas",
    "Cleaning and Chores",
    "Coffee Shop",
    "Commute",
    "Dance",
    "Danceable",
    "Date",
    "Dream",
    "Drive",
    "Entertainment",
    "Evening",
    "Family",
    "Family time",
    "Flirt",
    "Focus",
    "Food",
    "Friendship",
    "Funeral",
    "Game",
    "Graduation",
    "Halloween",
    "landscape/scenery",
    "Lounge",
    "Love",
    "Marketplace",
    "Meditation",
    "Mid-autumn Festival",
    "Morning",
    "National's Day",
    "New Year",
    "Nightclub",
    "Other",
    "Other scene",
    "Park",
    "Party",
    "Pet/Animals",
    "Prank",
    "Qi Xi",
    "Rainy Day",
    "Relaxation",
    "Restaurants",
    "Roadtrip",
    "Running",
    "Sport",
    "Spring",
    "Spring Festival",
    "Summer",
    "Sunny Day",
    "Theater/Concert hall",
    "Timelapse",
    "Transition",
    "Travel",
    "Universe",
    "Valentine's day",
    "Valentine's Day",
    "Vlog/DailyLife",
    "Wake up",
    "Wedding",
    "Winter",
    "Yoga",
]

AUDIO_LANG_V3 = [
    "Cantonese",
    "Chinese",
    "Chinese Dialects",
    "English",
    "African",
    "Arabic",
    "French",
    "German",
    "Hindi",
    "Instrumental/Non-vocal",
    "Italian",
    "Japanese",
    "Korea",
    "Other",
    "Other Lang",
    "Russian",
    "Sichuanese",
    "Taiwanese",
    "Turkish",
]

AUDIO_GENDER_V3 = [
    "Female",
    "Male",
    "Neutral",
    "Child",
    "Adult",
    "Chorus_AUDIO_GENDER",
    "Unknown",
    "Multiple",
    "female",
    "male",
    "child",
    "adult",
]

AUDIO_TIMBRE_V3 = [
    "Bright",
    "Cute_AUDIO_TIMBRE",
    "Cute",
    "Deep",
    "Electrified voice",
    "Ethereal",
    "Extreme",
    "Husky",
    "Loud and sonorous",
    "Magnetic",
    "Powerful",
    "Sexy/Lazy",
    "Sharp",
    "Sweet_AUDIO_TIMBRE",
]

AUDIO_SINKING_V3 = [
    "Sinking",
    "non-Sinking",
]

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
    "Shocking/Magnificent/Epic": "Shocking/magnificent/epic",
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
    #"": "Unknown",
    "": "empty gender",
}

AUDIO_TAGS_GENRE_SPECIAL_MAP_V2 = {
    "Hip Hop": "Hip Hop/Rap",
    "Acapella": "A cappella",
    "dj": "DJ",
    "Glam Metal（Hair Metal，Pop Metal）": "Glam Metal",
    "Glam Metal\n（Hair Metal，Pop Metal）": "Glam Metal",
    "low pop": "Low pop",
    "tuhai": "Tuhai",
    "ChinaWave": "China-Wave",
    "DJDJ/remix": "DJ",
    "Progressive MetalProg Metal": "Progressive Metal",
    "Red Song/": "Red Song",
    "Rock BluesNew": "Rock Blues",
    "Chorus": "Chorus_AUDIO_GENRE",  # dedup
    "low Pop": "Low pop",
    "": "empty genre",
    "Other": "Other_AUDIO_GENRE",
    "Other genre": "Other_AUDIO_GENRE",
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
    "Other": "Other_AUDIO_MOOD",
    #"Sweet": "Cute/Playful", # Sweet also in vocal_timbre
    "Sweet": "Sweet_SA_MOOD",
    "": "empty mood",
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
    "": "empty scene",
}

AUDIO_TAGS_GENDER_SPECIAL_MAP_V2 = AUDIO_TAGS_GENDER_SPECIAL_MAP_V1

AUDIO_TAGS_TIMBRE_SPECIAL_MAP = {  # for all versions
    # dedup
    "Cute": "Cute_AUDIO_TIMBRE",
    "Sweet": "Sweet_AUDIO_TIMBRE",
    "ACG": "ACG voice",
    "Chinese Folk": "Chinese_Folk_AUDIO_TIMBRE",
    "Romantic": "Romantic_AUDIO_TIMBRE",
    "": "empty timbre",
}

AUDIO_TAGS_LANG_SPECIAL_MAP = {
    "Other": "Other Lang",
    "Chinesse": "Chinese",
    "Canto": "Cantonese",
    "": "empty lang",
}

AUDIO_TAGS_EXTRA_SPECIAL_MAP = {
    "Tuhai": "Tuhai_AUDIO_EXTRA",
    "tuhai": "Tuhai_AUDIO_EXTRA",
    "Soundtrack": "Soundstrack_AUDIO_EXTRA",
    "soundtrack": "Soundstrack_AUDIO_EXTRA",
    "": "empty extra",
}

AUDIO_TAGS_INST_SPECIAL_MAP = {
    "Frenchhorn": "French_Horn",
    "Drumset": "Drum_Set",
    "Drums": "Drum_Set",
    "Smallpercussion": "Percussive",
    "Synthdrums": "Synth_Drums",
    "Electricpiano": "Electric_piano",
    "Doublebass": "Double_Bass",
    "Contrabass": "Double_Bass",
    "Synthstrings": "Synth_Strings",
    "Synthbass": "Synth_Bass",
    "Acousticguitar": "Acoustic_Guitar",
    "Electricguitar": "Electric_Guitar",
    "Pluck": "Synth_Pluck",
    "Lead": "Synth_Lead",
    "Pad": "Synth_Pad",
    "Syntheffects": "Synth_Effects",
    "Harp": "Orchestral_Harp",
    "Soundeffects": "Sound_Effects",
    "Bassdrum": "Bass_Drum",
    "Tom": "Tom_Tom",
    "Cymbals": "Crash_Cymbal",
    "Chimes": "Tubular_Bells",
    "Others": "Other Inst",
    "": "empty instrument",
}

AUDIO_TAGS_TEMPO_SPECIAL_MAP = {
    "": "empty tempo",
}

AUDIO_TAGS_MODE_SPECIAL_MAP = {
    "Others": "Other Mode",
    "Maj": "Major",
    "Min": "Minor",
    "Blues": "Blues_AUDIO_MODE",
    "": "empty mode",
}

AUDIO_TAGS_KEY_SPECIAL_MAP = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
    "": "empty key",
}

COMBO_VOICE_GENDER_TO_MULTITAG_V3 = ["Female", "Male"]

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
#AUDIO_CAT_VOCAB_V3 = [AUDIO_GENRE_V3, AUDIO_GENRE_V3, AUDIO_EXTRA_V3, AUDIO_MOOD_V3, AUDIO_SCENE_V3, AUDIO_GENDER_V3, AUDIO_TIMBRE_V3, AUDIO_LANG_V3, AUDIO_SINKING_V3]
AUDIO_CAT_VOCAB_V3 = list(VOCAB2ID_MIX_V4.to_dict().keys())

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


_UNIFIED_VOCAB_MAP = {
    "Audio_unified_v3": VOCAB2ID_AUDIO_V3,
    "Mix_unified_v3": VOCAB2ID_MIX_V3,
    "Audio_unified_v4": VOCAB2ID_AUDIO_V4,
    "Mix_unified_v4": VOCAB2ID_MIX_V4,
}

def parse_artist_id_by_lang_gender(lang, gender) -> int:
    if 'Chinese' in lang:
        artist_id = ARTIST_ID_MAP_V2['zh_empty']
    elif 'Cantonese' in lang:
        artist_id = ARTIST_ID_MAP_V2['cant_empty']
    elif 'English' in lang:
        artist_id = ARTIST_ID_MAP_V2['en_empty']
    else:
        artist_id = ARTIST_ID_MAP_V2['inst_empty']

    if 'Neutral' in gender:
        artist_id = ARTIST_ID_MAP_V2['Neutral']
    elif 'Child' in gender:
        artist_id = ARTIST_ID_MAP_V2['Child']
    elif 'Male' in gender and 'Female' in gender:
        artist_id = ARTIST_ID_MAP_V2['Male_Female']
    elif 'Male' in gender:
        artist_id = ARTIST_ID_MAP_V2['Male']
    elif 'Female' in gender:
        artist_id = ARTIST_ID_MAP_V2['Female']
    elif 'Multiple' in gender:
        artist_id = ARTIST_ID_MAP_V2['Multiple']
    elif 'Chorus' in gender:
        artist_id = ARTIST_ID_MAP_V2['Chorus']
    elif 'Chorus_AUDIO_GENDER' in gender:
        artist_id = ARTIST_ID_MAP_V2['Chorus_AUDIO_GENDER']
    return artist_id


def get_categorical_vocab(vocab_type: str) -> Tuple[Dict[str, int], int]:
    if vocab_type in _UNIFIED_VOCAB_MAP:
        _vocab2id = _UNIFIED_VOCAB_MAP[vocab_type]
        # Hard-code num_categories here. It's supposed to be decided by dataloader.
        # The lookup table does not know how many categories the dataloader would use.
        if vocab_type in ['Audio_unified_v3', 'Mix_unified_v3']:
            return _vocab2id.to_dict(), 9
        elif vocab_type in ['Audio_unified_v4', 'Mix_unified_v4']:
            return _vocab2id.to_dict(), 13
        else:
            return _vocab2id.to_dict(), 5


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

AUDIO_INST_V3 = [
    "Accordion",
    "Acousticguitar",
    "Acoustic_Guitar",
    "Acoustic_Piano",
    "Backingvocal",
    "Bagpipe",
    "Banjo",
    "Baritone_Sax",
    "Bass",
    "Bassdrum",
    "Bass_Drum",
    "Bassoon",
    "Bells",
    "Belltree",
    "Bianzhong",
    "Brass",
    "Brass_Section",
    "Cello",
    "Chimes",
    "Chinese_Drums",
    "Chinese Traditional Instruments",
    "Choir_and_Voice",
    "Chorus",
    "Chromatic_Percussion",
    "Clap",
    "Clarinet",
    "Clean_Electric_Guitar",
    "Contrabass",
    "Crash_Cymbal",
    "Cymbals",
    "Di",
    "Distorted_Electric_Guitar",
    "Doublebass",
    "Double_Bass",
    "Drums",
    "Drumset",
    "Drum_Set",
    "Dulcimer",
    "Electric_Bass",
    "Electricguitar",
    "Electric_Guitar",
    "Electric_piano",
    "Electricpiano",
    "English_Horn",
    "Erhu",
    "Ethnic",
    "Flute",
    "Frenchhorn",
    "French_Horn",
    "Fx",
    "Glockenspiel",
    "Guitar",
    "Guqin",
    "Guzheng",
    "Hangdrum",
    "Harmonica",
    "Harp",
    "Hi_hat",
    "Hulusi",
    "Irishwhistle",
    "Keys",
    "Lead",
    "Mandolin",
    "Marimba",
    "Matouqin",
    "Musicbox",
    "Oboe",
    "Ocarina",
    "Orchestral_Harp",
    "Organ",
    "Other Inst",
    "Pad",
    "Panflute",
    "Percussion",
    "Percussive",
    "Piccolo",
    "Pipa",
    "Pipe",
    "Pluck",
    "Plucked Strings",
    "Ruan",
    "Sandhammer",
    "Sanxian",
    "Saxophone",
    "Sheng",
    "Sitar",
    "Smallpercussion",
    "Snare",
    "Soprano/Alto_Sax",
    "Soundeffects",
    "Sound_Effects",
    "String_Ensemble",
    "Strings",
    "Suona",
    "Synthbass",
    "Synth_Bass",
    "Synth_Brass",
    "Synth_Clave",
    "Synth_Cymbals",
    "Synthdrums",
    "Synth_Drums",
    "Syntheffects",
    "Synth_Effects",
    "Synthesizers",
    "Synth_Hi_hat",
    "Synth_Kick",
    "Synth_Lead",
    "Synth_Pad",
    "Synth_Pluck",
    "Synth_Snare",
    "Synthstrings",
    "Synth_Strings",
    "Synth_Tom",
    "Tambourine",
    "Tenor_Sax",
    "Timpani",
    "Tom",
    "Tom_Tom",
    "Trombone",
    "Trumpet",
    "Tuba",
    "Tubular_Bells",
    "Ukelele",
    "Vibraphone",
    "Viola",
    "Violin",
    "Vocal",
    "Vocal_Chops",
    "Whistle",
    "Woodwinds",
    "Xiao",
    "Xun",
    "Xylophone",
    "Yangqin",
]


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

_TEMPO_COARSE_LABEL_RANGES = {
    "Slow": (0, 60),
    "Moderate speed": (60, 110),
    "Fast": (110, 150),
    "Very fast": (150, 200),
}

COARSE_TEMPO_LABEL2TEMPO_LABEL = {
    "Slow": ["Grave", "Largo"],
    "Moderate speed": ["Adagio", "Andante", "Moderato"],
    "Fast": ["Allegro", "Vivace"],
    "Very fast": ["Vivace", "Presto"],
}

TEMPO_RANGE = (0, 200)
TEMPO_LABELS = [""] + list(_TEMPO_LABEL_RANGES.keys())
COARSE_TEMPO_LABELS = [""] + list(_TEMPO_COARSE_LABEL_RANGES.keys())

TEMPO_LABEL_ID_MAP = {label: i for i, label in enumerate(TEMPO_LABELS)}
ID_TEMPO_LABEL_MAP = {v: k for k, v in TEMPO_LABEL_ID_MAP.items()}
COARSE_TEMPO_LABELS_ID_MAP = {label: i for i, label in enumerate(COARSE_TEMPO_LABELS)}
ID_COARSE_TEMPO_LABEL_MAP = {v: k for k, v in COARSE_TEMPO_LABELS_ID_MAP.items()}

TIME_SIGNATURES = ["X", "2/4", "3/4", "4/4"]  # currently only support 4 quarter note
TIME_SIGNATURE_ID_MAP = {label: i for i, label in enumerate(TIME_SIGNATURES)}
ID_TIME_SIGNATURE_MAP = {v: k for k, v in TIME_SIGNATURE_ID_MAP.items()}



def tempo_to_label(tempo: Optional[int]) -> str:
    if tempo is None:
        return ""
    for label, (low, high) in _TEMPO_LABEL_RANGES.items():
        if low <= tempo < high:
            return label
    return ""

def tempo_to_coarse_label(tempo: Optional[int]) -> str:
    if tempo is None:
        return ""
    for label, (low, high) in _TEMPO_COARSE_LABEL_RANGES.items():
        if low <= tempo < high:
            return label
    return ""


# ========================================================================


# ========================================================================

_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_MODES = ["Maj", "Min", "Major", "Minor"]
_FINEGRAINED_MODES = ["Others", "Major", "Minor", "Gregorian", "Pentatonic", "Blues", # first class mode label
                      "Natural_Major", "Harmonic_Major", "Melodic_Major",    # second class mode label
                      "Natural_Minor", "Harmonic_Minor", "Melodic_Minor",
                      "Gong", "Shang", "Jue", "Zhi", "Yu", "DuJie"]

# - used by music fm plus 
ROOTS = ["X"] + _ROOTS
ROOT_ID_MAP = {label: i for i, label in enumerate(ROOTS)}
ID_ROOT_MAP = {v: k for k, v in ROOT_ID_MAP.items()}

MODES = ["X"] + _MODES
MODE_ID_MAP = {
    "X": 0,
    "Maj": 1, "Major": 1,
    "Min": 2, "Minor": 2,
}
ID_MODE_MAP = {v: k for k, v in MODE_ID_MAP.items()}

FINEGRAINED_MODES = ["X"] + _FINEGRAINED_MODES

KEYS = ["X"] + [root + ":" + key_type for root in _ROOTS for key_type in _MODES]
KEY_ID_MAP = {label: i for i, label in enumerate(KEYS)}
ID_KEY_MAP = {v: k for k, v in KEY_ID_MAP.items()}


# ========================================================================


class StyleText:
    STYLE_TEXT_TAGS_IN_ORDER = ["genre", "mood", "gender", "timbre", "scene", "genre_extra", "extra", "language", "is_sinking", "instrument", "tempo", "key", "mode"]
    _STYLE_TEXT_TAG_IDX_MAP = {tag_name: pos for pos, tag_name in enumerate(STYLE_TEXT_TAGS_IN_ORDER)}

    def __init__(self, style_text: str):
        self.style_text = style_text
        self.tags = self.style_text.split("|")

        self.genre = self.tags[self._STYLE_TEXT_TAG_IDX_MAP["genre"]] if self._STYLE_TEXT_TAG_IDX_MAP["genre"] < len(self.tags) else ""
        self.mood = self.tags[self._STYLE_TEXT_TAG_IDX_MAP["mood"]] if self._STYLE_TEXT_TAG_IDX_MAP["mood"] < len(self.tags) else ""
        self.gender = self.tags[self._STYLE_TEXT_TAG_IDX_MAP["gender"]] if self._STYLE_TEXT_TAG_IDX_MAP["gender"] < len(self.tags) else ""
        self.timbre = self.tags[self._STYLE_TEXT_TAG_IDX_MAP["timbre"]] if self._STYLE_TEXT_TAG_IDX_MAP["timbre"] < len(self.tags) else ""
        self.scene = self.tags[self._STYLE_TEXT_TAG_IDX_MAP["scene"]] if self._STYLE_TEXT_TAG_IDX_MAP["scene"] < len(self.tags) else ""
        self.genre_extra = self.tags[self._STYLE_TEXT_TAG_IDX_MAP["genre_extra"]] if self._STYLE_TEXT_TAG_IDX_MAP["genre_extra"] < len(self.tags) else ""
        self.extra = self.tags[self._STYLE_TEXT_TAG_IDX_MAP["extra"]] if self._STYLE_TEXT_TAG_IDX_MAP["extra"] < len(self.tags) else ""
        self.language = self.tags[self._STYLE_TEXT_TAG_IDX_MAP["language"]] if self._STYLE_TEXT_TAG_IDX_MAP["language"] < len(self.tags) else ""
        self.is_sinking = self.tags[self._STYLE_TEXT_TAG_IDX_MAP["is_sinking"]] if self._STYLE_TEXT_TAG_IDX_MAP["is_sinking"] < len(self.tags) else ""
        self.instrument = self.tags[self._STYLE_TEXT_TAG_IDX_MAP["instrument"]] if self._STYLE_TEXT_TAG_IDX_MAP["instrument"] < len(self.tags) else ""
        self.tempo_label = self.tags[self._STYLE_TEXT_TAG_IDX_MAP["tempo_label"]] if self._STYLE_TEXT_TAG_IDX_MAP["tempo_label"] < len(self.tags) else ""
        self.key = self.tags[self._STYLE_TEXT_TAG_IDX_MAP["key"]] if self._STYLE_TEXT_TAG_IDX_MAP["key"] < len(self.tags) else ""
        self.mode = self.tags[self._STYLE_TEXT_TAG_IDX_MAP["mode"]] if self._STYLE_TEXT_TAG_IDX_MAP["mode"] < len(self.tags) else ""


def _split_and_check_style_text_v3(text: str) -> Tuple[str, str, str, str, str]:
    style_text = StyleText(text)
    genre, mood, gender, scene, timbre, genre_extra, extra, language, is_sinking, instrument, tempo_label, key, mode = (
        style_text.genre,
        style_text.mood,
        style_text.gender,
        style_text.scene,
        style_text.timbre,
        style_text.genre_extra,
        style_text.extra,
        style_text.language,
        style_text.is_sinking,
        style_text.instrument,
        style_text.tempo_label,
        style_text.key,
        style_text.mode,
    )
    target_mood_list = []
    target_mood_list.extend(AUDIO_MOOD_V3)
    target_mood_list.extend(SA_MOOD19)
    target_mood_list.extend(list(MAP_SA_TO_AUDIO_TAG.keys()))
    new_mood = []
    if mood:
        for _mood in mood.split(','):
            _mood = SA_TAGS_SPECIAL_MAP.get(_mood, _mood)
            if _mood not in target_mood_list:
                print(target_mood_list)
                raise ValueError(f"Unsupported mood {_mood}")
            new_mood.append(_mood)
    mood = ','.join(new_mood)
    if gender and gender not in AUDIO_GENDER_V3:
        raise ValueError(f"Unsupported gender {gender}")
    return genre, mood, scene, gender, timbre, genre_extra, extra, language, is_sinking, instrument, tempo_label, key, mode


def _subgenre_to_genre_by_hierarchy(subgenre: str, supported_genres: List[str]) -> str:
    """This function is a patch for rewrite_style_input_to_multi_tag_combo_v4 to ensure \
    any genre tag, either in the vocab or not, can be mapped to the tags that are in the vocab.
    """

    def subgenre_to_genre(subgenre: str) -> Optional[str]:
        for genre, subgenres in DEFAULT_TAG_HIERARCHY["genre"].items():
            if subgenre in subgenres:
                return genre
        return "Pop"  # Set a default genre

    if not supported_genres:
        raise ValueError("supported_genres can not be empty")
    if subgenre in supported_genres:
        return subgenre
    genre = subgenre_to_genre(subgenre)
    return genre if genre in supported_genres else supported_genres[0]


def rewrite_style_input_to_multi_tag_combo_v4(
    text: str, random_seed: Optional[int] = None
) -> Tuple[str, str, str, int]:
    rnd = random.Random(random_seed)
    logger.info(f"'rewrite_style_input_to_multi_tag_combo_v4' will use seed {random_seed}")
    target_genre_list = []
    target_sub_genre_list = []
    main_genre_list = list(COMBO_GENRE_TO_MULTITAG_V4.keys())
    sub_genre_mapping = SUB_GENRE_MAPPING
    for key1 in COMBO_GENRE_TO_MULTITAG_V4:
        target_genre_list.append(key1)
        for item in COMBO_GENRE_TO_MULTITAG_V4[key1]:
            for key2 in item.split("|")[0].split(","):
                if key2 not in target_genre_list:
                    target_sub_genre_list.append(key2)
                sub_genre_mapping[key2] = key1
    target_genre_list = list(set(target_genre_list))
    target_sub_genre_list = list(set(target_sub_genre_list))

    genres, mood, scene, gender, timbre, genre_extra, extra, language, is_sinking = (
        _split_and_check_style_text_v3(text)
    )

    # Patch (Yilin):  Remap the genres to make the function work
    genres = ",".join(
        list(
            set(
                _subgenre_to_genre_by_hierarchy(g, list(sub_genre_mapping.keys()))
                for g in genres.split(",")
            )
        )
    )

    # default is Chinese
    if not language:
        language = "Chinese"
    # default genre
    format_genres = []
    main_format_genres = []
    for genre in genres.split(","):
        if genre in AUDIO_GENRE_V3:
            if genre != sub_genre_mapping[genre]:
                format_genres.append(genre)
                main_format_genres.append(sub_genre_mapping[genre])
        else:
            logger.info("%s not in AUDIO_GENRE_V3" % genre)
    for genre in genres.split(","):
        if (
            genre in AUDIO_GENRE_V3
            and genre not in format_genres
            and genre not in main_format_genres
        ):
            format_genres.append(genre)
    if len(format_genres) == 0:
        for genre in genres.split(","):
            if genre in sub_genre_mapping:
                format_genres.append(sub_genre_mapping[genre])
            else:
                logger.info("%s not in sub_genre_mapping" % genre)
    format_genres = list(set(format_genres))
    if len(format_genres) == 0:
        if timbre:
            for item in timbre.split(","):
                if item in ["Loud and sonorous", "Powerful"]:
                    format_genres.append(
                        rnd.choice(["Chinese Tradition", "Rock", "Punk"])
                    )
                elif item in ["Sexy/Lazy"]:
                    format_genres.append(rnd.choice(["Jazz", "R&B/Soul", "Reggae"]))
                elif item in ["Electrified voice"]:
                    format_genres.append(
                        rnd.choice(["Electronic", "Hip Hop/Rap", "DJ"])
                    )
        else:
            format_genres = ["Pop"]
    if len(format_genres) == 0:
        format_genres = ["Pop"]
    logger.info("format_genres: %s" % format_genres)
    expand_mood_list = [
        [
            "Miss",
            "Sorrow/Sad",
            "Nostalgic/Memory",
            "Sentimental/Melancholic/Lonely",
        ],
        [
            "Dynamic/Energetic",
            "Excited",
        ],
        [
            "Chill",
            "Calm/Relaxing",
            "Healing",
        ],
        [
            "Dreamy/Ethereal",
            "Mysterious",
        ],
    ]
    mood_type = {
        "neutral": [
            "Shocking/magnificent/epic",
            "Chill",
            "Calm/Relaxing",
            "Mysterious",
            "Groovy/Funky",
            "Dynamic/Energetic",
            "Romantic",
            "Nostalgic/Memory",
            "Miss",
            "Dreamy/Ethereal",
            "Healing",
            "No Mood",
        ],
        "positive": [
            "Happy",
            "Cute/Playful",
            "Excited",
            "Funny",
            "Inspirational/Hopeful",
            "Sweet",
        ],
        "negative": [
            "Sorrow/Sad",
            "Sentimental/Melancholic/Lonely",
            "Weird",
            "Thrilling/Suspenseful/Tense",
            "Angry/Aggressive",
        ],
    }
    mood_mapping = {}
    for genre in format_genres:
        if genre in COMBO_GENRE_TO_MULTITAG_V4:
            for item in COMBO_GENRE_TO_MULTITAG_V4[genre]:
                for key in item.split("|")[1].split(","):
                    if key not in mood_mapping:
                        mood_mapping[key] = []
                    mood_mapping[key].append(item)
    random_genre_list = []
    new_mood_list = []
    not_match_mood_list = []
    if mood:
        for key in mood.split(","):
            key = MAP_SA_TO_AUDIO_TAG[key] if key in MAP_SA_TO_AUDIO_TAG else key
            for item in expand_mood_list:
                if key in item:
                    new_mood_list.extend(item)
        for key in new_mood_list:
            if key in mood_mapping:
                random_genre_list.extend(mood_mapping[key])
            else:
                not_match_mood_list.append(key)
    if len(random_genre_list) == 0:
        for genre in format_genres:
            if genre in COMBO_GENRE_TO_MULTITAG_V4:
                random_genre_list.extend(COMBO_GENRE_TO_MULTITAG_V4[genre])
            else:
                for item in COMBO_GENRE_TO_MULTITAG_V4[sub_genre_mapping[genre]]:
                    if genre in item.split("|")[0].split(","):
                        random_genre_list.append(item)
                if len(random_genre_list) == 0:
                    random_genre_list.extend(
                        COMBO_GENRE_TO_MULTITAG_V4[sub_genre_mapping[genre]]
                    )
    if len(random_genre_list) == 0:
        random_genre_list.extend(COMBO_GENRE_TO_MULTITAG_V4["Pop"])
    logger.info("random_genre_list: %s" % random_genre_list)
    _multitag_list = rnd.choice(random_genre_list)
    _multitag_list = _multitag_list.split("|")
    # [genre, mood, scene, timbre] -> [genre, mood, scene, gender, timbre]
    multitag_list = [
        _multitag_list[0],
        _multitag_list[1],
        _multitag_list[2],
        "",
        _multitag_list[3],
    ]
    if len(format_genres) > 0:
        new_genre = []
        for genre in format_genres:
            new_genre.append(genre)
            if sub_genre_mapping[genre] in multitag_list[0].split(","):
                new_genre.append(sub_genre_mapping[genre])
        new_genre = list(set(new_genre))
        if len(new_genre) == 1:
            new_genre.extend(multitag_list[0].split(","))
            multitag_list[0] = ",".join(list(set(new_genre)))
        elif len(new_genre) > 1:
            _new_genre = []
            _new_genre_extra = []
            for g in new_genre:
                mg = sub_genre_mapping[g]
                if mg in multitag_list[0].split(",") or g in multitag_list[0].split(
                    ","
                ):
                    _new_genre.append(g)
                else:
                    _new_genre_extra.append(g)
            if len(_new_genre) == 1:
                _new_genre.extend(multitag_list[0].split(","))
            multitag_list[0] = ",".join(list(set(_new_genre)))
            genre_extra = ",".join(list(set(_new_genre_extra)))

    new_mood = mood.split(",")
    new_mood_type = []
    for m in new_mood:
        if m in mood_type["positive"]:
            new_mood_type.append("positive")
        if m in mood_type["negative"]:
            new_mood_type.append("negative")
    for m in multitag_list[1].split(","):
        if m in mood_type["neutral"] and m not in new_mood:
            new_mood.append(m)
        if m in mood_type["positive"] and "negative" not in new_mood_type and m not in new_mood:
            new_mood.append(m)
        if m in mood_type["negative"] and "positive" not in new_mood_type and m not in new_mood:
            new_mood.append(m)
    # cause tags_lyrics only read first one tag in mood, keep user choice in first place
    multitag_list[1] = ",".join(new_mood)
    if timbre:
        multitag_list[4] = timbre
    logger.info("multitag_list: %s" % multitag_list)
    if not gender:
        if "Sweet_AUDIO_TIMBRE" in multitag_list[4]:
            gender = "Female,Adult"
        elif "Cute_AUDIO_TIMBRE" in multitag_list[4]:
            gender = "Female,Adult"
        elif "Husky" in multitag_list[4]:
            gender = "Male,Adult"
        elif "Loud and sonorous" in multitag_list[4]:
            gender = "Male,Adult"
        else:
            gender = rnd.choice(COMBO_VOICE_GENDER_TO_MULTITAG_V3)
    if "Adult" not in gender:
        gender = "Adult," + gender
    multitag_list[3] = gender
    multitag_list.append(genre_extra)
    multitag_list.append(extra)
    multitag_list.append(language)
    if not is_sinking:
        is_sinking = "non-Sinking"
        for item in multitag_list[0].split(","):
            if item in ["DJ", "MC", "Tuhai", "Tuhai_AUDIO_EXTRA"]:
                is_sinking = "Sinking"
        for item in multitag_list[5].split(","):
            if item in ["DJ", "MC", "Tuhai", "Tuhai_AUDIO_EXTRA"]:
                is_sinking = "Sinking"
        for item in multitag_list[6].split(","):
            if item in ["DJ", "MC", "Tuhai", "Tuhai_AUDIO_EXTRA"]:
                is_sinking = "Sinking"
    if is_sinking == "Sinking" and "Tuhai_AUDIO_EXTRA" not in extra:
        extra = multitag_list[6].split(",")
        extra.append("Tuhai_AUDIO_EXTRA")
        multitag_list[6] = ",".join(extra)
    multitag_list.append(is_sinking)

    # check rewrite result is valid
    vocab2id = VOCAB2ID_MIX_V3.to_dict()
    new_multitag_list = []
    for item in multitag_list:
        tmp_list = []
        for _item in item.split(","):
            if _item:
                if _item in vocab2id:
                    tmp_list.append(_item)
                else:
                    logger.info("%s not in vocab2id %s" % (_item, vocab2id))
        new_multitag_list.append(",".join(tmp_list))
    # [genre, mood, scene, gender, timbre, genre_extra, extra, lang, sinking] ->
    # [genre, genre_extra, extra, mood, scene, gender, timbre, lang, sinking]
    multitag = "|".join(
        [
            # genre
            new_multitag_list[0],
            # genre_extra
            new_multitag_list[5],
            # extra
            new_multitag_list[6],
            # mood
            new_multitag_list[1],
            # scene
            new_multitag_list[2],
            # gender
            new_multitag_list[3],
            # timbre
            new_multitag_list[4],
            # lang
            new_multitag_list[7],
            # sinking
            new_multitag_list[8],
        ]
    )
    speaker_id = parse_artist_id_by_lang_gender(language, gender)

    key, tempo_label = "N", ""
    logger.info("Original style input : " + text)
    logger.info("Rewrite style input : " + multitag)
    return multitag, key, tempo_label, speaker_id


def rewrite_total_duration_to_section_duration(total_duration: float) -> Tuple[list, list]:

    patterns = [
        {
            'sections': ['verse'],
            'normed_section_durations': [[1]],
            'duration_range': [0, 15],
            'weight': 0.5,
        },
        {
            'sections': ['chorus'],
            'normed_section_durations': [[1]],
            'duration_range': [0, 15],
            'weight': 1,
        },
        {
            'sections': ['inst'],
            'normed_section_durations': [[1]],
            'duration_range': [0, 15],
            'weight': 1,
        },
        {
            'sections': ['verse', 'chorus'],
            'normed_section_durations': [[1, 2], [1, 1], [1, 1.5]],
            'duration_range': [10, 30],
            'weight': 1,
        },
        {
            'sections': ['chorus', 'chorus'],
            'normed_section_durations': [[1, 2], [1, 1], [1, 1.5]],
            'duration_range': [10, 30],
            'weight': 0.2,
        },
        {
            'sections': ['intro', 'chorus'],
            'normed_section_durations': [[1, 2], [1, 1], [1, 1.5]],
            'duration_range': [10, 30],
            'weight': 0.5,
        },
        {
            'sections': ['intro', 'verse', 'chorus'],
            'normed_section_durations': [[1, 2, 2], [1, 1, 1], [1, 2, 4]],
            'duration_range': [25, 40],
            'weight': 1,
        },
        {
            'sections': ['verse', 'inst', 'chorus'],
            'normed_section_durations': [[1, 1, 2], [1, 1, 1], [1, 2, 4]],
            'duration_range': [25, 40],
            'weight': 1,
        },
        {
            'sections': ['verse', 'verse', 'chorus'],
            'normed_section_durations': [[1, 1, 2]],
            'duration_range': [25, 40],
            'weight': 0.5,
        },
        {
            'sections': ['verse', 'chorus', 'chorus'],
            'normed_section_durations': [[1, 1, 1]],
            'duration_range': [25, 40],
            'weight': 0.5,
        },
        {
            'sections': ['verse', 'chorus', 'verse', 'chorus'],
            'normed_section_durations': [[1, 1, 1, 1], [1, 2, 1, 2]],
            'duration_range': [30, 60],
            'weight': 1,
        },
        {
            'sections': ['verse', 'chorus', 'inst', 'chorus'],
            'normed_section_durations': [[1, 1, 1, 1], [1, 2, 1, 2]],
            'duration_range': [30, 60],
            'weight': 1,
        },
        {
            'sections': ['verse', 'inst', 'chorus', 'chorus'],
            'normed_section_durations': [[1, 1, 1, 1], [1, 2, 1, 1], [1, 2, 2, 2]],
            'duration_range': [30, 60],
            'weight': 1,
        },
        {
            'sections': ['verse', 'inst', 'verse', 'inst'],
            'normed_section_durations': [[1, 2, 2, 1], [1, 2, 4, 1]],
            'duration_range': [30, 60],
            'weight': 1,
        },
        {
            'sections': ['verse', 'inst', 'verse', 'inst'],
            'normed_section_durations': [[1, 2, 2, 1], [1, 2, 4, 1]],
            'duration_range': [50, 80],
            'weight': 1,
        },
        {
            'sections': ['verse', 'inst', 'verse', 'chorus'],
            'normed_section_durations': [[1, 1, 1, 1], [1, 2, 1, 1], [1, 2, 2, 2]],
            'duration_range': [50, 80],
            'weight': 1,
        },
        {
            'sections': ['intro', 'verse', 'chorus', 'verse', 'chorus'],
            'normed_section_durations': [[1, 2, 2, 2, 2]],
            'duration_range': [50, 80],
            'weight': 1,
        },
        {
            'sections': ['intro', 'verse', 'chorus', 'inst', 'chorus'],
            'normed_section_durations': [[1, 2, 2, 2, 2], [1, 2, 2, 1, 2]],
            'duration_range': [50, 80],
            'weight': 1,
        },
        {
            'sections': ['intro', 'verse', 'chorus', 'bridge', 'verse', 'chorus'],
            'normed_section_durations': [[1, 2, 2, 1, 2, 2]],
            'duration_range': [70, 120],
            'weight': 1,
        },
        {
            'sections': ['intro', 'verse', 'chorus', 'bridge', 'chorus', 'outro'],
            'normed_section_durations': [[1, 2, 2, 1, 2, 1]],
            'duration_range': [70, 120],
            'weight': 1,
        },
        {
            'sections': ['intro', 'verse', 'chorus', 'inst', 'chorus', 'outro'],
            'normed_section_durations': [[1, 2, 2, 1, 2, 1]],
            'duration_range': [70, 120],
            'weight': 1,
        },
        {
            'sections': ['verse', 'chorus', 'inst', 'verse', 'chorus', 'outro'],
            'normed_section_durations': [[1, 1, 1, 1, 1, 0.5]],
            'duration_range': [70, 120],
            'weight': 1,
        },
        {
            'sections': ['intro', 'verse', 'chorus', 'inst', 'verse', 'chorus', 'outro'],
            'normed_section_durations': [[0.5, 1, 1, 1, 1, 1, 0.5]],
            'duration_range': [70, 120],
            'weight': 1,
        },
        {
            'sections': ['intro', 'verse', 'chorus', 'chorus', 'inst', 'verse', 'chorus', 'chorus', 'outro'],
            'normed_section_durations': [[0.5, 1, 1, 1, 1, 1, 1, 1, 0.5]],
            'duration_range': [90, 240],
            'weight': 1,
        },
        {
            'sections': ['intro', 'verse', 'chorus', 'bridge', 'chorus', 'verse', 'chorus', 'outro'],
            'normed_section_durations': [[0.5, 1, 1, 0.5, 1, 1, 1, 0.5], [1, 1, 1, 1, 1, 1, 1, 0.5],  [0.5, 1, 1, 1, 1, 1, 1, 0.5]],
            'duration_range': [90, 240],
            'weight': 1,
        },
        {
            'sections': ['intro', 'verse', 'chorus', 'inst', 'chorus', 'verse', 'chorus', 'outro'],
            'normed_section_durations': [[0.5, 1, 1, 0.5, 1, 1, 1, 0.5], [1, 1, 1, 1, 1, 1, 1, 0.5],  [0.5, 1, 1, 1, 1, 1, 1, 0.5]],
            'duration_range': [90, 240],
            'weight': 1,
        },
        {
            'sections': ['intro', 'verse', 'chorus', 'bridge', 'chorus', 'bridge', 'chorus', 'bridge', 'chorus', 'outro'],
            'normed_section_durations': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0.5, 1, 1, 0.5, 1, 0.5, 1, 0.5, 1, 0.5]],
            'duration_range': [90, 240],
            'weight': 1,
        },
        {
            'sections': ['intro', 'verse', 'chorus', 'bridge', 'verse', 'chorus', 'bridge', 'verse', 'chorus', 'bridge', 'outro'],
            'normed_section_durations': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0.5, 1, 1, 0.5, 1, 1, 0.5, 1, 1, 0.5, 0.5]],
            'duration_range': [90, 240],
            'weight': 1,
        },
    ]
    # for item in patterns:
    #     for durations in item['normed_section_durations']:
    #         assert(len(durations) == len(item['sections']))
    valid_patterns = [x for x in patterns if  x['duration_range'][0] <= total_duration <= x['duration_range'][1] ]
    weights = [x['weight'] for x in valid_patterns]
    pattern = random.choices(valid_patterns, weights=weights, k=1)[0]
    section_tags = pattern['sections']
    normed_section_durations = random.choice(pattern['normed_section_durations'])
    section_durations = [x/sum(normed_section_durations) * total_duration for x in normed_section_durations]
    message = ' + '.join(['%s (%.1fs)' % (a, b)for a, b in zip(section_tags, section_durations)])
    # logger.log(msg = f'##### rewrite section duration: {total_duration} -> {message}', level=30)
    logger.debug(f'##### rewrite section duration: {total_duration} -> {message}')
    return section_tags, section_durations


def rewrite_style_input_to_multi_tag_combo_v5(
    text: str, random_seed: Optional[int] = None
) -> Tuple[str, str, str, int]:
    genres, mood, scene, gender, timbre, genre_extra, extra, language, is_sinking, instrument, tempo_label, music_key, music_mode = (
        _split_and_check_style_text_v3(text)
    )

    rnd = random.Random(random_seed)
    logger.info(f"'rewrite_style_input_to_multi_tag_combo_v5' will use seed {random_seed}")
    target_genre_list = []
    target_sub_genre_list = []
    if language == "Cantonese":
        main_genre_list = list(COMBO_GENRE_TO_CANT_MULTITAG_V5.keys())
    else:
        main_genre_list = list(COMBO_GENRE_TO_MULTITAG_V4.keys())
    sub_genre_mapping = SUB_GENRE_MAPPING
    for key1 in COMBO_GENRE_TO_MULTITAG_V4:
        target_genre_list.append(key1)
        for item in COMBO_GENRE_TO_MULTITAG_V4[key1]:
            for key2 in item.split("|")[0].split(","):
                if key2 not in target_genre_list:
                    target_sub_genre_list.append(key2)
                sub_genre_mapping[key2] = key1
    target_genre_list = list(set(target_genre_list))
    target_sub_genre_list = list(set(target_sub_genre_list))

    # Patch (Yilin):  Remap the genres to make the function work
    genres = ",".join(
        list(
            set(
                _subgenre_to_genre_by_hierarchy(g, list(sub_genre_mapping.keys()))
                for g in genres.split(",")
            )
        )
    )

    # default is Chinese
    if not language:
        language = "Chinese"
    # default genre
    format_genres = []
    main_format_genres = []
    for genre in genres.split(","):
        if genre in AUDIO_GENRE_V3:
            if genre != sub_genre_mapping[genre]:
                format_genres.append(genre)
                main_format_genres.append(sub_genre_mapping[genre])
        else:
            logger.info("%s not in AUDIO_GENRE_V3" % genre)
    for genre in genres.split(","):
        if (
            genre in AUDIO_GENRE_V3
            and genre not in format_genres
            and genre not in main_format_genres
        ):
            format_genres.append(genre)
    if len(format_genres) == 0:
        for genre in genres.split(","):
            if genre in sub_genre_mapping:
                format_genres.append(sub_genre_mapping[genre])
            else:
                logger.info("%s not in sub_genre_mapping" % genre)
    format_genres = list(set(format_genres))
    if len(format_genres) == 0:
        if timbre:
            for item in timbre.split(","):
                if item in ["Loud and sonorous", "Powerful"]:
                    format_genres.append(
                        rnd.choice(["Chinese Tradition", "Rock", "Punk"])
                    )
                elif item in ["Sexy/Lazy"]:
                    format_genres.append(rnd.choice(["Jazz", "R&B/Soul", "Reggae"]))
                elif item in ["Electrified voice"]:
                    format_genres.append(
                        rnd.choice(["Electronic", "Hip Hop/Rap", "DJ"])
                    )
        else:
            format_genres = ["Pop"]
    if len(format_genres) == 0:
        format_genres = ["Pop"]
    logger.info("format_genres: %s" % format_genres)
    expand_mood_list = [
        [
            "Miss",
            "Sorrow/Sad",
            "Nostalgic/Memory",
            "Sentimental/Melancholic/Lonely",
        ],
        [
            "Dynamic/Energetic",
            "Excited",
        ],
        [
            "Chill",
            "Calm/Relaxing",
            "Healing",
        ],
        [
            "Dreamy/Ethereal",
            "Mysterious",
        ],
    ]
    mood_type = {
        "neutral": [
            "Shocking/magnificent/epic",
            "Chill",
            "Calm/Relaxing",
            "Mysterious",
            "Groovy/Funky",
            "Dynamic/Energetic",
            "Romantic",
            "Nostalgic/Memory",
            "Miss",
            "Dreamy/Ethereal",
            "Healing",
            "No Mood",
        ],
        "positive": [
            "Happy",
            "Cute/Playful",
            "Excited",
            "Funny",
            "Inspirational/Hopeful",
            "Sweet",
        ],
        "negative": [
            "Sorrow/Sad",
            "Sentimental/Melancholic/Lonely",
            "Weird",
            "Thrilling/Suspenseful/Tense",
            "Angry/Aggressive",
        ],
    }
    mood_mapping = {}
    for genre in format_genres:
        if genre in COMBO_GENRE_TO_MULTITAG_V4:
            for item in COMBO_GENRE_TO_MULTITAG_V4[genre]:
                for key in item.split("|")[1].split(","):
                    if key not in mood_mapping:
                        mood_mapping[key] = []
                    mood_mapping[key].append(item)
    random_genre_list = []
    new_mood_list = []
    not_match_mood_list = []
    if mood:
        for key in mood.split(","):
            key = MAP_SA_TO_AUDIO_TAG[key] if key in MAP_SA_TO_AUDIO_TAG else key
            for item in expand_mood_list:
                if key in item:
                    new_mood_list.extend(item)
        for key in new_mood_list:
            if key in mood_mapping:
                random_genre_list.extend(mood_mapping[key])
            else:
                not_match_mood_list.append(key)
    if len(random_genre_list) == 0:
        for genre in format_genres:
            if genre in COMBO_GENRE_TO_MULTITAG_V4:
                random_genre_list.extend(COMBO_GENRE_TO_MULTITAG_V4[genre])
            else:
                for item in COMBO_GENRE_TO_MULTITAG_V4[sub_genre_mapping[genre]]:
                    if genre in item.split("|")[0].split(","):
                        random_genre_list.append(item)
                if len(random_genre_list) == 0:
                    random_genre_list.extend(
                        COMBO_GENRE_TO_MULTITAG_V4[sub_genre_mapping[genre]]
                    )
    if len(random_genre_list) == 0:
        random_genre_list.extend(COMBO_GENRE_TO_MULTITAG_V4["Pop"])
    logger.info("random_genre_list: %s" % random_genre_list)
    _multitag_list = rnd.choice(random_genre_list)
    _multitag_list = _multitag_list.split("|")
    # [genre, mood, scene, timbre] -> [genre, mood, scene, gender, timbre]
    multitag_list = [
        _multitag_list[0],
        _multitag_list[1],
        _multitag_list[2],
        "",
        _multitag_list[3],
    ]
    if len(format_genres) > 0:
        new_genre = []
        for genre in format_genres:
            new_genre.append(genre)
            if sub_genre_mapping[genre] in multitag_list[0].split(","):
                new_genre.append(sub_genre_mapping[genre])
        new_genre = list(set(new_genre))
        if len(new_genre) == 1:
            new_genre.extend(multitag_list[0].split(","))
            multitag_list[0] = ",".join(list(set(new_genre)))
        elif len(new_genre) > 1:
            _new_genre = []
            _new_genre_extra = []
            for g in new_genre:
                mg = sub_genre_mapping[g]
                if mg in multitag_list[0].split(",") or g in multitag_list[0].split(
                    ","
                ):
                    _new_genre.append(g)
                else:
                    _new_genre_extra.append(g)
            if len(_new_genre) == 1:
                _new_genre.extend(multitag_list[0].split(","))
            multitag_list[0] = ",".join(list(set(_new_genre)))
            genre_extra = ",".join(list(set(_new_genre_extra)))

    new_mood = mood.split(",")
    new_mood_type = []
    for m in new_mood:
        if m in mood_type["positive"]:
            new_mood_type.append("positive")
        if m in mood_type["negative"]:
            new_mood_type.append("negative")
    for m in multitag_list[1].split(","):
        if m in mood_type["neutral"] and m not in new_mood:
            new_mood.append(m)
        if m in mood_type["positive"] and "negative" not in new_mood_type and m not in new_mood:
            new_mood.append(m)
        if m in mood_type["negative"] and "positive" not in new_mood_type and m not in new_mood:
            new_mood.append(m)
    # cause tags_lyrics only read first one tag in mood, keep user choice in first place
    multitag_list[1] = ",".join(new_mood)
    if timbre:
        multitag_list[4] = timbre
    logger.info("multitag_list: %s" % multitag_list)
    if not gender:
        if "Sweet_AUDIO_TIMBRE" in multitag_list[4]:
            gender = "Female"
        elif "Cute_AUDIO_TIMBRE" in multitag_list[4]:
            gender = "Female"
        elif "Husky" in multitag_list[4]:
            gender = "Male"
        elif "Loud and sonorous" in multitag_list[4]:
            gender = "Male"
        else:
            gender = rnd.choice(COMBO_VOICE_GENDER_TO_MULTITAG_V3)
    multitag_list[3] = gender
    multitag_list.append(genre_extra)
    multitag_list.append(extra)
    multitag_list.append(language)
    multitag_list.append(is_sinking)

    # check rewrite result is valid
    vocab2id = DEFAULT_VOCAB2ID.to_dict()
    new_multitag_list = []
    for item in multitag_list:
        tmp_list = []
        for _item in item.split(","):
            if _item:
                if _item in vocab2id:
                    tmp_list.append(_item)
                else:
                    logger.info("%s not in vocab2id %s" % (_item, vocab2id))
        new_multitag_list.append(",".join(tmp_list))
    # [genre, mood, scene, gender, timbre, genre_extra, extra, lang, sinking] ->
    # [genre, genre_extra, extra, mood, scene, gender, timbre, lang, sinking]
    inst = []
    for g in new_multitag_list[0].split(','):
        if g in COMBO_GENRE_TO_INSTRUMENTS_V4:
            inst.append(rnd.choice(COMBO_GENRE_TO_INSTRUMENTS_V4[g]))
    multitag = [
        # genre
        new_multitag_list[0],
        # genre_extra
        new_multitag_list[5],
        # extra
        new_multitag_list[6],
        # mood
        new_multitag_list[1],
        # scene
        new_multitag_list[2],
        # gender
        new_multitag_list[3],
        # timbre
        new_multitag_list[4],
        # lang
        new_multitag_list[7],
        # sinking
        new_multitag_list[8],
        # instrument
        ','.join(inst),
        # tempo
        '',
        # key
        '',
        # mode
        '',
    ]
    #multitag = [x.split(',') for x in multitag]
    #multitag = process_tags(multitag)
    #multitag = [','.join(x) for x in multitag]
    multitag = "|".join(multitag)
    speaker_id = parse_artist_id_by_lang_gender(language, gender)

    if not music_key:
        music_key = "X"
    if not tempo_label:
        tempo_label = ""
    logger.info("Original style input : " + text)
    logger.info("Rewrite style input : " + multitag)
    return multitag, music_key, tempo_label, speaker_id
