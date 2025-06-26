"""
Unified Chinese Vocabs.

CATEGORY_MAP: Strings -> Unified Tags
UNIFIED_VOCAB2ID: -> Unified Tags -> Ids

A Vocab2Id object contains one CATEGORY_MAP and one UNIFIED_VOCAB2ID,
combined they form a string-to-id mapping.

**IMPORTANT**

Instructions to add a new vocab2id lookup table:
1. Check if all your labels can be mapped to an existing UnifiedVocab enum value.
   If not, add a new enum value under the corresponding category.
2. Create a UNIFIED_VOCAB2ID dict that maps unified vocab to int ids.
   It is not necessary if you just want to reuse an existing one.
3. Create a CATEGORY_MAP that maps strings to unified tags
4. Put your CATEGORY_MAP and UNIFIED_VOCAB2ID into a Vocab2Id class
5. If you want to make compatible with VOCAB2ID_SA, register it to _sa_vocab_compatibility_check
   to pass the compatibility check.
"""

# SA:
# Genre, Mood, Theme, Sinking, Lang
# Audio:
# Genre, Mood, Scene, Gender, Timber

# Unified:
# Genre, Mood, Theme, Sinking, Lang, Gender, Timber(Artist)

from typing import Dict, List, Optional
from enum import Enum, auto


# ============================================================
# Unified Vocab
# ============================================================

class UnifiedCategory(Enum):
    GENRE = auto()
    MOOD = auto()
    THEME = auto()
    SINKING = auto()
    LANG = auto()
    GENDER = auto()
    TIMBRE = auto()
    EXTRA = auto()
    INSTRUMENT = auto()
    AUDIO_QUALITY = auto()
    POPULARITY = auto()


class UnifiedVocab(Enum):
    pass



class UnifiedDumpster(UnifiedVocab):
    NONE = auto()

#karaoke version
class UnifiedInstrument(UnifiedVocab):
    EMPTY = auto()
    WURLITZER = auto()
    FRENCH_HORN = auto()
    BONGOS = auto()
    DOUBLE_BASS = auto()
    PAN_FLUTE = auto()
    LEAD_ACOUSTIC_GUITAR = auto()
    MANDOLIN = auto()
    TALK_BOX = auto()
    TABLA = auto()
    TRADITIONAL_WIND_INSTRUMENTS = auto()
    STRING_SECTION = auto()
    VIBES = auto()
    RHYTHM_ACOUSTIC_GUITAR_ARPEGGIO = auto()
    ELECTRONIC_PERCUSSION = auto()
    SYNTH_BASS = auto()
    SYNTH_BRASS = auto()
    BACKING_VOCALS = auto()
    SEA = auto()
    CELESTA = auto()
    FEMALE_LEAD_VOCAL = auto()
    MUSIC_BOX = auto()
    MOOG_SYNTHESIZER = auto()
    UKULELE = auto()
    TENOR_SAXOPHONE = auto()
    PEDAL_STEEL_GUITAR = auto()
    RAINSTICK = auto()
    SYNTH_FLUTE = auto()
    ENGLISH_HORN = auto()
    ACOUSTIC_BASS_GUITAR = auto()
    ELECTRONIC_HI_HAT = auto()
    FINGER_CYMBALS = auto()
    STEEL_DRUMS = auto()
    DIDGERIDOO = auto()
    DRUM = auto()
    BASSOON = auto()
    KOTO = auto()
    CASTANETS = auto()
    DISTORTED_ELECTRIC_GUITAR = auto()
    SHAMISEN = auto()
    VOICE = auto()
    BASS = auto()
    KORA = auto()
    MUTED_TROMBONE = auto()
    PIANO = auto()
    RHYTHM_ELECTRIC_GUITAR = auto()
    LEAD_ELECTRIC_GUITAR = auto()
    PITCHED_PERCUSSION = auto()
    TUBA = auto()
    ARR_ACOUSTIC_GUITAR = auto()
    CP = auto()
    SURDO = auto()
    BRASS_AND_HARMONICA = auto()
    SANDPAPER = auto()
    CHARANGO = auto()
    KEYBOARD = auto()
    FIDDLE = auto()
    SOPRANO_SAXOPHONE = auto()
    POLYSYNTH = auto()
    VINYL = auto()
    SOUND_EFFECTS = auto()
    HUMAN_BEATBOX = auto()
    BREATH = auto()
    RHYTHM_ACOUSTIC_GUITAR = auto()
    HARMONICA = auto()
    BALALAIKA = auto()
    CLARINET = auto()
    TIMPANI = auto()
    XYLOPHONE = auto()
    CLAP = auto()
    TOMS = auto()
    SAMPLE = auto()
    GUTHBUINNE = auto()
    METAL_BARS = auto()
    OTHER_GUITAR = auto()
    SNAP_FINGERS = auto()
    CAVAQUINHO = auto()
    GUITAR = auto()
    MARACAS = auto()
    TAMBOURINE = auto()
    TOY_PIANO = auto()
    LAP_STEEL_GUITAR = auto()
    MALE_LEAD_VOCAL = auto()
    HAMMOND = auto()
    WHISTL = auto()
    DULCIMER = auto()
    ORCHESTRAL_PERCUSSION = auto()
    SPOONS = auto()
    GONG = auto()
    SYNTH_PAD = auto()
    ELECTRONIC_CYMBALS = auto()
    BARITONE_SAXOPHONE = auto()
    MALE_BACKING_VOCALS = auto()
    MARIMBA = auto()
    RHYTHM_ELECTRIC_GUITAR_ARPEGGIO = auto()
    ACOUSTIC_DRUM_KIT = auto()
    SAXOPHONE = auto()
    DJEMBE = auto()
    LEAD_VOCAL = auto()
    SNARE_DRUM = auto()
    CLAVES_WOODBLOCK = auto()
    WIND_CHIMES = auto()
    MUSICAL_SAW = auto()
    LATIN_PERCUSSION = auto()
    LOOP = auto()
    WOODEN_BARS = auto()
    COWBELL = auto()
    SIREN = auto()
    ORGAN = auto()
    BRASS_INSTRUMENTS = auto()
    SHAKER = auto()
    CONGAS = auto()
    ORCHESTRA = auto()
    ELECTRONIC_DRUM_KIT = auto()
    ELECTRIC_BASS = auto()
    SHEHNAI = auto()
    TRUMPET = auto()
    CYMBALS = auto()
    SYNTH_STRINGS = auto()
    FLUTE = auto()
    SCRATCH = auto()
    DIGITAL_PIANO = auto()
    CLAVINET = auto()
    AGOGO = auto()
    THEREMIN = auto()
    TAMBORA = auto()
    TAMBOURA = auto()
    GLOCKENSPIEL = auto()
    NOISE_EFFECTS = auto()
    ORGAN_AND_ACCORDION = auto()
    HARMONIUM = auto()
    ELECTRIC_PIANO = auto()
    ACOUSTIC_GUITAR = auto()
    VIOLIN = auto()
    ORCHESTRA_HIT = auto()
    RHODES = auto()
    WIND = auto()
    ALTO_SAXOPHONE = auto()
    KAZOO = auto()
    HAND_CLAP = auto()
    CAJON = auto()
    NONE = auto()
    LUTE = auto()
    SITAR = auto()
    EXPLOSION = auto()
    GUITAR_SYNTH = auto()
    ETHNIC_PERCUSSION = auto()
    MUTED_TRUMPET = auto()
    ARPEGIATOR = auto()
    ROTOTOM = auto()
    FX = auto()
    TRIANGLE = auto()
    FRETLESS_BASS = auto()
    THUMB_PIANO = auto()
    ARR_ELECTRIC_GUITAR = auto()
    HI_HAT = auto()
    BOUZOUKI = auto()
    APPALACHIAN_DULCIMER = auto()
    PICCOLO = auto()
    SAROD = auto()
    BASS_DRUM = auto()
    DRUM_KIT = auto()
    MIXED_PERCUSSION = auto()
    WASHBOARD_GUIRO = auto()
    CABASA = auto()
    PERCUSSION = auto()
    ELECTRONIC_BASS_DRUM = auto()
    FEMALE_BACKING_VOCALS = auto()
    SYNTH_AMBIANT = auto()
    MELLOTRON = auto()
    JEWS_HARP = auto()
    CROWD = auto()
    JINGLE_BELLS = auto()
    TRADITIONAL_STRINGED_INSTRUMENTS = auto()
    TROMBONE = auto()
    ELECTRONIC_TOMS = auto()
    IRISH_FLUTE = auto()
    ACCORDION = auto()
    PANDEIRO = auto()
    BELLS = auto()
    SONAR = auto()
    SYNTH_VOICE = auto()
    DRUMS_AND_PERCUSSION = auto()
    MOOG = auto()
    VIBRASLAP = auto()
    TUBULAR_BELL = auto()
    CALLIOPE = auto()
    VOCAL_BASSIST = auto()
    CLOCK = auto()
    CELLO = auto()
    SYNTHESIZER = auto()
    BIRDS = auto()
    BARITONE_GUITAR = auto()
    SYNTH_KEYS = auto()
    OBOE = auto()
    UPRIGHT_BASS = auto()
    SYNTH_LEAD = auto()
    SHAKUHACHI = auto()
    FILTER = auto()
    VOCODER = auto()
    TAIKO = auto()
    TAP_DANCE = auto()
    BANJO = auto()
    BRASS_SECTION = auto()
    VIOLA = auto()
    SUB = auto()
    DOBRO = auto()
    HARP = auto()
    MELODICA = auto()
    BAGPIPES = auto()
    OCARINA = auto()
    BOWL = auto()
    WOODWINDS = auto()
    HARPSICHORD = auto()
    ELECTRONIC_SNARE_DRUM = auto()
    BODY_PERCUSSION = auto()
    ELECTRIC_GUITAR = auto()
    FLUGELHORN = auto()
    ZITHER = auto()
    TRADITIONAL_FLUTE = auto()
    

class UnifiedGenre(UnifiedVocab):
    EMPTY = auto()
    POP = auto()
    ELECTRONIC = auto()
    CHINESE_STYLE = auto()
    ROCK = auto()
    JAZZ = auto()
    HIP_HOP = auto()
    CLASSICAL = auto()
    RNB_SOUL = auto()
    METAL = auto()
    TUHAI = auto()
    CHILDHOOD = auto()
    DEVOTIONAL = auto()
    CHINESE_TRADITION = auto()
    EASY_LISTENING = auto()
    NEW_AGE = auto()
    EIGHT_BIT = auto()
    FOLK = auto()
    LATIN = auto()
    ALTERNATIVE_INDIE = auto()
    EPIC = auto()
    BGM = auto()
    CHINESE_POP = auto()
    CANTOPOP = auto()
    TAIWANESE_POP = auto()
    POP_FOLK = auto()
    CONTEMPORARY_POP = auto()
    TEEN_POP = auto()
    INDIE_POP = auto()
    DREAM_POP = auto()
    CITY_POP = auto()
    SYNTH_POP = auto()
    CELTIC_POP = auto()
    DANCE_POP = auto()
    DEEP_DANCE_POP = auto()
    ELECTROPOP = auto()
    CHAMBER_POP = auto()
    A_CAPPELLA = auto()
    COUNTRY_POP = auto()
    EDM = auto()
    HOUSE = auto()
    DUBSTEP = auto()
    FUTURE_BASS = auto()
    CHILLOUT = auto()
    TRANCE = auto()
    TECHNO = auto()
    DRUMNBASS = auto()
    TROPICAL_HOUSE = auto()
    DISCO = auto()
    VAPORWAVE = auto()
    TRIP_HOP = auto()
    AMBIENT = auto()
    DEEP_POP_EDM = auto()
    EDM_TRAP = auto()
    FUTURE_HOUSE = auto()
    CHINA_WAVE = auto()
    GUFENG_MUSIC = auto()
    CHINOISERIE_RAP = auto()
    CHINOISERIE_ELECTRONIC = auto()
    HARD_ROCK = auto()
    PSYCHEDELIC_ROCK = auto()
    POP_ROCK = auto()
    INSTRUMENTAL_ROCK = auto()
    ALTERNATIVE_ROCK = auto()
    INDIE_ROCK = auto()
    POST_ROCK = auto()
    LO_FI = auto()
    J_ROCK = auto()
    SHOEGAZING = auto()
    MATH_ROCK = auto()
    SURF_ROCK = auto()
    PROGRESSIVE_ROCK = auto()
    SOFT_ROCK = auto()
    JAZZ_POP = auto()
    JAZZ_FUSION = auto()
    BOSSA_NOVA = auto()
    AVANT_GARDE_JAZZ = auto()
    SWING = auto()
    BIG_BAND = auto()
    BOP = auto()
    POST_BOP = auto()
    SMOOTH_JAZZ = auto()
    COOL_JAZZ = auto()
    VOCAL_JAZZ = auto()
    NU_JAZZ = auto()
    FREE_JAZZ = auto()
    TRAP_RAP = auto()
    OLD_SCHOOL = auto()
    RNB_RAP = auto()
    JAZZ_HIP_HOP = auto()
    ALTERNATIVE_HIP_HOP = auto()
    INSTRUMENTAL_HIP_HOP = auto()
    POP_RAP = auto()
    HARDCORE_RAP = auto()
    COMEDY_HIP_HOP = auto()
    HIP_HOUSE = auto()
    CHILL_BEATS = auto()
    CHORUS = auto()
    CHAMBER_MUSIC = auto()
    SYMPHONY = auto()
    FUNK = auto()
    CONTEMPORARY_RNB = auto()
    NEO_SOUL = auto()
    SOUL = auto()
    POP_SOUL = auto()
    BLACK_METAL = auto()
    DEATH_METAL = auto()
    GLAM_METAL = auto()
    GRINDCORE = auto()
    POWER_METAL = auto()
    PROGRESSIVE_METAL = auto()
    SPEED_METAL = auto()
    DJ = auto()
    VINAHOUSE = auto()
    MC = auto()
    GOSPEL = auto()
    HOLIDAY_MUSIC = auto()
    CHRISTIAN_MUSIC = auto()
    BUDDHIST_MUSIC = auto()
    CHINESE_OPERA = auto()
    TRADITIONAL_CHINESE_FOLK = auto()
    CHINESE_QUYI = auto()
    RED_SONG = auto()
    FOLK_POP = auto()
    INDIE_FOLK = auto()
    TANGO = auto()
    REGGAETON = auto()
    BLUES = auto()
    PUNK = auto()
    REGGAE = auto()
    BOOMBAP = auto()
    CHILLWAVE = auto()
    CHINESE_FOLK = auto()
    CONTEMPORARY_BLUES = auto()
    CONTEMPORARY_FOLK = auto()
    COUNTRY_BLUES = auto()
    EMO_RAP = auto()
    ENGLISH_FOLK = auto()
    FLAMENCO = auto()
    GANGSTA_RAP = auto()
    HARDSTYLE = auto()
    HEAVY_METAL = auto()
    HINRG = auto()
    INDIAN_POP = auto()
    ITALIAN_POP = auto()
    JAPANESE_FOLK = auto()
    J_POP = auto()
    KPOP = auto()
    LATIN_POP = auto()
    LOW_POP = auto()
    MELBOURNE_BOUNCE = auto()
    NEW_CHINESE_FOLK = auto()
    NEW_WAVE = auto()
    NOSTALGIC_POP = auto()
    POP_PUNK = auto()
    ROCK_BLUES = auto()
    RUSSIAN_POP = auto()
    SAMBA = auto()
    SHI_DAI_QU = auto()
    SOUNDTRACK = auto()
    THRASH_METAL = auto()
    TRADITIONAL_FOLK = auto()
    TRADITIONAL_JAZZ = auto()
    VULGAR_POP = auto()
    WEST_COAST_HIP_HOP = auto()
    WORLD_MUSIC = auto()
    BREAKBEAT = auto()
    COUNTRY_FOLK = auto()
    COUNTRY_ROCK = auto()
    JAZZ_BLUES = auto()
    ACAPELLA = auto()
    ACID_JAZZ = auto()
    AFRICAN_FOLK = auto()
    AFRICAN_POP = auto()
    ARABIC_FOLK = auto()
    ARABIC_POP = auto()
    BAROQUE = auto()
    BASS_HOUSE = auto()
    BEHOP = auto()
    BHANGRA = auto()
    BLUE_GRASS = auto()
    BLUES_ROCK = auto()
    BOOGIE_WOOGIE = auto()
    BRAZILIAN_FUNK_STYLE = auto()
    BRIT_POP = auto()
    BUBBLEGUM_BASS = auto()
    CHINESE_BALLAD_POP = auto()
    CHINESE_TRADITIONAL_INSTRUMENTAL_MUSIC = auto()
    CLASSICAL_PERIOD = auto()
    CLASSIC_RNB_SOUL = auto()
    CONTEMPORARY_CLASSICAL_MUSIC = auto()
    DANCEHALL = auto()
    DANCE_PUNK = auto()
    DARK_AMBIENT = auto()
    DEATHCORE = auto()
    DOOM_METAL = auto()
    DOO_WOP = auto()
    DOWNTEMPO = auto()
    DRILL_RAP = auto()
    EAST_COAST_HIP_HOP = auto()
    EMO_PUNK = auto()
    ENKA = auto()
    FADO = auto()
    FOLK_METAL = auto()
    FOLKTRONICA = auto()
    FRENCH_FOLK = auto()
    FRENCH_POP = auto()
    GERMAN_POP = auto()
    GLITCH = auto()
    GOTHIC_METAL = auto()
    GRUNGE_ROCK = auto()
    GYPSY_JAZZ = auto()
    HARD_BOP = auto()
    HARDCORE = auto()
    HARDCORE_PUNK = auto()
    IDM = auto()
    IMPRESSIONISM = auto()
    INDUSTRIAL_METAL = auto()
    IRISH_FOLK = auto()
    JANPANESE_TRADITIONAL_MUSIC = auto()
    KOREAN_FOLK = auto()
    KOREA_TROT = auto()
    LATIN_ROCK = auto()
    LO_FI_HOUSE = auto()
    LO_FI_ROCK = auto()
    METALCORE = auto()
    MIDWEST_HIP_HOP = auto()
    MODERNISM = auto()
    MONGOLIAN_FOLK_SONGS = auto()
    NEO_FUNK = auto()
    NO_WAVE = auto()
    NU_METAL = auto()
    POST_HARDCORE = auto()
    POST_PUNK = auto()
    PROGRESSIVE_RNB = auto()
    RAGTIME = auto()
    RAP_METAL = auto()
    ROMANTIC_MUSIC = auto()
    RUMBA = auto()
    SCORE = auto()
    SHIMA_UTA = auto()
    SHOEGAZE_ROCK = auto()
    SKA_PUNK = auto()
    SON_CUBANO = auto()
    SOUTHERN_HIP_HOP = auto()
    SYNTHWAVE = auto()
    TRADITIONAL_BLUES = auto()
    TURKISH_POP = auto()
    UYGUR_FOLK_SONGS = auto()
    WORLDBEAT = auto()
    YODELING = auto()


class UnifiedAudioQuality(UnifiedVocab):
    EMPTY = auto()
    HIGH_QUALITY = auto()
    LOW_QUALITY = auto()


class UnifiedPopularity(UnifiedVocab):
    EMPTY = auto()
    HIGH_POPULARITY = auto()
    LOW_POPULARITY = auto()

class UnifiedMood(UnifiedVocab):
    EMPTY = auto()
    HAPPY = auto()
    CUTE_PLAYFUL = auto()
    EXCITED = auto()
    FUNNY = auto()
    INSPIRATIONAL_HOPEFUL = auto()
    SORROW_SAD = auto()
    SENTIMENTAL_MELANCHOLIC_LONELY = auto()
    WEIRD = auto()
    THRILLING_SUSPENSEFUL_TENSE = auto()
    ANGRY_AGGRESSIVE = auto()
    GROOVY_FUNKY = auto()
    DYNAMIC_ENERGETIC = auto()
    ROMANTIC = auto()
    NOSTALGIC_MEMORY = auto()
    DREAMY_ETHEREAL = auto()
    HEALING = auto()
    MISS = auto()
    CHILL = auto()
    CALM_RELAXING = auto()
    SHOCKING_MAGNIFICENT_EPIC = auto()
    MYSTERIOUS = auto()
    SWEET = auto()


class UnifiedTheme(UnifiedVocab):
    EMPTY = auto()
    HALLOWEEN = auto()
    CHRISTMAS = auto()
    NEW_YEAR = auto()
    SPRING_FESTIVAL = auto()
    VALENTINES_DAY = auto()
    QI_XI = auto()
    BIRTHDAY = auto()
    WEDDING = auto()
    FUNERAL = auto()
    GRADUATION = auto()
    NATIONALS_DAY = auto()
    VLOG_DAILYLIFE = auto()
    FOOD = auto()
    PET_ANIMALS = auto()
    BEAUTY_FASHION = auto()
    ENTERTAINMENT = auto()
    BABIES = auto()
    CHILDREN = auto()
    TRANSITION = auto()
    ANIME = auto()
    WAKE_UP = auto()
    FAMILY_TIME = auto()
    LANDSCAPE_SCENERY = auto()
    PRANK = auto()
    TIMELAPSE = auto()
    RAINY_DAY = auto()
    SUNNY_DAY = auto()
    SPRING = auto()
    SUMMER = auto()
    AUTUMN = auto()
    WINTER = auto()
    EVENING = auto()
    MORNING = auto()
    BEACH = auto()
    NIGHTCLUB = auto()
    COFFEE_SHOP = auto()
    RESTAURANTS = auto()
    LOUNGE = auto()
    CAMPUS = auto()
    PARK = auto()
    MARKETPLACE = auto()
    UNIVERSE = auto()
    BAR = auto()
    THEATER_CONCERT_HALL = auto()
    SPORT = auto()
    DANCE = auto()
    GAME = auto()
    TRAVEL = auto()
    FOCUS = auto()
    PARTY = auto()
    COMMUTE = auto()
    ROADTRIP = auto()
    MEDITATION = auto()
    RELAXATION = auto()
    CLEANING_AND_CHORES = auto()
    RUNNING = auto()
    DATE = auto()
    DANCEABLE = auto()
    FLIRT = auto()

    BED_TIME = auto()
    BROKE_UP = auto()
    DREAM = auto()
    DRIVE = auto()
    FRIENDSHIP = auto()
    LOVE = auto()  # Yilin: Merge it with DATE or FLIRT?
    YOGA = auto()

    MID_AUTUMN_FESTIVAL = auto()


class UnifiedLang(UnifiedVocab):
    EMPTY = auto()
    CANTONESE = auto()
    CHINESE = auto()
    CHINESE_DIALECT = auto()
    ENGLISH = auto()
    # audio tags ver2
    AFRICAN = auto()
    ARABIC = auto()
    FRENCH = auto()
    GERMAN = auto()
    HINDI = auto()
    INSTRUMENTAL = auto()
    ITALIAN = auto()
    JAPANESE = auto()
    KOREA = auto()
    RUSSIAN = auto()
    SICHUANESE = auto()
    TAIWANESE = auto()
    TURKISH = auto()


class UnifiedSinking(UnifiedVocab):
    EMPTY = auto()
    SINKING = auto()
    NON_SINKING = auto()


class UnifiedGender(UnifiedVocab):
    EMPTY = auto()
    NEUTRAL = auto()
    FEMALE = auto()
    MALE = auto()
    CHILD = auto()
    ADULT = auto()
    CHORUS = auto()
    # audio tags ver2
    MULTIPLE = auto()


class UnifiedTimbre(UnifiedVocab):
    EMPTY = auto()
    WARM = auto()
    ETHEREAL = auto()
    HUSKY = auto()
    DEEP = auto()
    LOUD_AND_SONOROUS = auto()
    EXTREME = auto()
    SHARP = auto()
    BRIGHT = auto()
    SWEET = auto()
    POWERFUL = auto()
    SEXY_LAZY = auto()
    MAGNETIC = auto()
    CUTE = auto()
    ELECTRIFIED_VOICE = auto()

    # artist
    JieGeWenKaiYiQiang = auto() # 接个吻,开一枪
    XiaoXia = auto() # 小霞
    YaoShiSan = auto() # 尧十三
    C_BLOCK = auto() # C-BLOCK
    SongDongYe = auto() # 宋冬野
    KKLUV = auto() # kkluv
    WanNengQingNianLvDian = auto() # 万能青年旅店
    TFBOYS = auto() # TFBOYS
    GuoDing = auto() # 郭顶
    THE_CRANE = auto() # The Crane
    HuiChunDan = auto() # 回春丹
    LvYanLiang = auto() # 吕彦良
    CaoDongMeiYouPaiDui = auto() # 草东没有派对
    HuMengZhou = auto() # Corsak胡梦周
    ZhangChu = auto() # 张楚
    MaSaiKeYueDui = auto() # 马赛克乐队
    DingShiGuang = auto() # 丁世光
    ShuangSheng = auto() # 双笙
    HuangLing = auto() # 黄龄
    HuangXuan = auto() # YELLOW黄宣
    ZhaoLei = auto() # 赵雷
    AiYiLiang = auto() # 艾怡良
    J_SHEON = auto() # J.Sheon
    GARETH_T = auto() # Gareth.T
    STEP_JAD = auto() # Step.Jad
    FineYueTuan = auto() # Fine乐团
    TangChao = auto() # 唐朝
    LengJing = auto() # 棱镜
    NaQiWoFu = auto() # 那奇沃夫
    QiuDe = auto() # 裘德
    LaoLang = auto() # 老狼
    ChongSuDiaoXiangDeQuanLi = auto() # 重塑雕像的权利
    GuanHaoDe = auto() # 关浩德
    DouJingTong = auto() # 窦靖童
    LiJiuZhe = auto() # 李玖哲
    ASi = auto() # 阿肆
    NineM88 = auto() # 9m88
    LuoRiFeiChe = auto() # 落日飞车
    WangYiTai = auto() # 王以太
    PaiKeTe = auto() # 派克特
    PuShu = auto() # 朴树
    GALI = auto() # GALI
    LiJiaLong = auto() # 李佳隆
    ANTI_GENERAL = auto() # Anti-general
    GaoWuRen = auto() # 告五人
    LiuBoXin = auto() # 刘柏辛Lexie
    HuangZiTao = auto() # 黄子韬
    TIZZY_T = auto() # Tizzy T
    DuiZhang = auto() # 队长
    HIGHER_BROTHERS = auto() # Higher Brothers
    CAPPER = auto() # Capper
    ZhuJingXi = auto() # 朱婧汐
    HuaChenYu = auto() # 华晨宇
    JONY_J = auto() # Jony J
    DanBao = auto() # 蛋堡
    YuJiaYun = auto() # 余佳运
    DaBoLang = auto() # 大波浪
    YangHeSu = auto() # 杨和苏KeyNG
    AiRe = auto() # 艾热AIR
    XieChunHua = auto() # 谢春花
    WeiLiAn = auto() # 韦礼安
    YangZongWei = auto() # 杨宗纬
    HaoMeiMei = auto() # 好妹妹
    XiaZhiYu = auto() # 夏之禹
    ChenHongYu = auto() # 陈鸿宇
    AR_LiuFuYang = auto() # AR刘夫阳
    KAFE_HU = auto() # Kafe.Hu
    XueZhiQian = auto() # 薛之谦
    XinYueTuan = auto() # 信乐团
    WanNiDa = auto() # 万妮达
    KEY_L_LiuCong = auto() # KEY.L刘聪
    SuYunYing = auto() # 苏运莹
    KNOWKNOW = auto() # KnowKnow
    ERIC_ZhouXingZhe = auto() # Eric周兴哲
    XuSong = auto() # 许嵩
    GAI_ZhouYan = auto() # GAI周延
    FaLao = auto() # 法老
    SunNan = auto() # 孙楠
    VAVA_MaoYanQi = auto() # VaVa毛衍七
    LU1 = auto() # Lu1
    ErShouMeiGui = auto() # 二手玫瑰
    NINEONE_ZhaoXingYue = auto() # NINEONE赵馨玥
    TongYangYueDui = auto() # 痛仰乐队
    MaSiWei = auto() # 马思唯
    AiChen = auto() # 艾辰
    ShanYiChun = auto() # 单依纯
    LinYouJia = auto() # 林宥嘉
    ZhangLiangYing = auto() # 张靓颖
    ICE_PAPER = auto() # Ice Paper
    MaoBuYi = auto() # 毛不易
    SunShengXi = auto() # 孙盛希
    XuWei = auto() # 许巍
    LiYuGang = auto() # 李玉刚
    LiRongHao = auto() # 李荣浩
    ChenLi = auto() # 陈粒
    XuJiaYing = auto() # 徐佳莹
    WeiRuXuan = auto() # 魏如萱
    WangFeng = auto() # 汪峰
    XinKuZi = auto() # 新裤子
    TaoZhe = auto() # 陶喆
    FeiErYueDui = auto() # 飞儿乐团
    HuaZhou = auto() # 花粥
    ChenQiZhen = auto() # 陈绮贞
    CaoGe = auto() # 曹格
    WuTiaoRen = auto() # 五条人
    CiWei_HEDGEHOG = auto() # 刺猬Hedgehog
    ZhangJie = auto() # 张杰
    FangDongDeMao = auto() # 房东的猫
    ICE_YangChangQing = auto() # ICE杨长青
    CaiJianYa = auto() # 蔡健雅
    ZhangShaoHan = auto() # 张韶涵
    MC_HOTDOG = auto() # MC Hotdog
    YinLin = auto() # 银临
    XiaoJingTeng = auto() # 萧敬腾
    LiJian = auto() # 李健
    SHE = auto() # S.H.E
    PanWeiBo = auto() # 潘玮柏
    YuanYaWei = auto() # 袁娅维
    GEM_DengZiQi = auto() # G.E.M.邓紫棋
    ZhouJieLun = auto() # 周杰伦
    LiangJingRu = auto() # 梁静茹
    FangDaTong = auto() # 方大同
    SunYanZi = auto() # 孙燕姿
    ZhangZhenYue = auto() # 张震岳
    CaiYiLin = auto() # 蔡依林
    YinQueShiTing = auto() # 音阙诗听
    WangSuLong = auto() # 汪苏泷
    HeTu = auto() # 河图
    HuYanBin = auto() # 胡彦斌
    LinJunJie = auto() # 林俊杰
    ZhangHuiMei = auto() # 张惠妹
    ChenXiaoChun = auto() # 陈小春
    SuDaLv = auto() # 苏打绿
    WangLiHong = auto() # 王力宏
    ZhouShen = auto() # 周深
    ZhangXinZhe = auto() # 张信哲
    LiKeQin = auto() # 李克勤
    DouWei = auto() # 窦唯
    ChenYiXun = auto() # 陈奕迅
    ChenBaiQiang = auto() # 陈百强
    WuYueTian = auto() # 五月天
    ZhangXueYou = auto() # 张学友
    WangFei = auto() # 王菲
    LiuDeHua = auto() # 刘德华
    LinYiLian = auto() # 林忆莲
    DengLiJun = auto() # 邓丽君
    YouZhiYuanShaShou = auto() # 幼稚园杀手



class UnifiedExtra(UnifiedVocab):
    # audio tags ver2
    EMPTY = auto()
    LO_FI = auto()
    NOSTALGIC = auto()
    SOUNDTRACK = auto()
    TUHAI = auto()


# =========================================================
# Unified Vocab2id Lookup Table
# NOTE: Order matters! Do not insert, only append.
# =========================================================

def lst_to_unified_vocab2id(lst: List[UnifiedVocab]) -> Dict[UnifiedVocab, int]:
    return {v: i for i, v in enumerate(lst)}


# =========================================================
# Define string mapping 
# =======================

tag_type_dict = {
            "GENRE": UnifiedCategory.GENRE,
            "THEME": UnifiedCategory.THEME,
            "MOOD":  UnifiedCategory.MOOD,
            "LANG": UnifiedCategory.LANG,
            "TIMBRE": UnifiedCategory.TIMBRE,
            "GENDER": UnifiedCategory.GENDER,
            "INSTRUMENT": UnifiedCategory.INSTRUMENT,
            "AUDIO_QUALITY": UnifiedCategory.AUDIO_QUALITY,
            "POPULARITY": UnifiedCategory.POPULARITY,
            "ARTIST": UnifiedCategory.TIMBRE # TODO: Artist not share the same as timbre
        }
# =========================================================


# The legacy vocab2id lookup table that supports SA tags and Audio tags V0
_UNIFIED_VOCAB2ID_LEGACY_LST = [
    UnifiedDumpster.NONE,

    # SA genre
    UnifiedGenre.BLUES,
    UnifiedGenre.CHINESE_OPERA,
    UnifiedGenre.CHINESE_STYLE,
    UnifiedGenre.CHINESE_TRADITION,
    UnifiedGenre.CLASSICAL,
    UnifiedGenre.COUNTRY_POP,
    UnifiedGenre.DJ,
    UnifiedGenre.EASY_LISTENING,
    UnifiedGenre.ELECTRONIC,
    UnifiedGenre.FOLK,
    UnifiedGenre.HIP_HOP,
    UnifiedGenre.JAZZ,
    UnifiedGenre.LATIN,
    UnifiedGenre.MC,
    UnifiedGenre.METAL,
    UnifiedGenre.EMPTY,
    UnifiedGenre.POP,
    UnifiedGenre.PUNK,
    UnifiedGenre.RNB_SOUL,
    UnifiedGenre.REGGAE,
    UnifiedGenre.ROCK,

    # SA mood
    UnifiedMood.ANGRY_AGGRESSIVE,
    UnifiedMood.CALM_RELAXING,
    UnifiedMood.CHILL,
    UnifiedMood.CUTE_PLAYFUL,
    UnifiedMood.DYNAMIC_ENERGETIC,
    UnifiedMood.EXCITED,
    UnifiedMood.FUNNY,
    UnifiedMood.HAPPY,
    UnifiedMood.HEALING,
    UnifiedMood.INSPIRATIONAL_HOPEFUL,
    UnifiedMood.SENTIMENTAL_MELANCHOLIC_LONELY,
    UnifiedMood.MISS,
    UnifiedMood.MYSTERIOUS,
    UnifiedMood.EMPTY,
    UnifiedMood.ROMANTIC,
    UnifiedMood.SORROW_SAD,
    UnifiedMood.SWEET,
    UnifiedMood.THRILLING_SUSPENSEFUL_TENSE,
    UnifiedMood.WEIRD,

    # SA theme
    UnifiedTheme.AUTUMN,
    UnifiedTheme.BED_TIME,
    UnifiedTheme.BIRTHDAY,
    UnifiedTheme.BROKE_UP,
    UnifiedTheme.COFFEE_SHOP,
    UnifiedTheme.CAMPUS,
    UnifiedTheme.CHRISTMAS,
    UnifiedTheme.DANCE,
    UnifiedTheme.DANCEABLE,
    UnifiedTheme.DATE,
    UnifiedTheme.DREAM,
    UnifiedTheme.DRIVE,
    UnifiedTheme.EVENING,
    UnifiedTheme.FAMILY_TIME,
    UnifiedTheme.FOCUS,
    UnifiedTheme.FOOD,
    UnifiedTheme.FRIENDSHIP,
    UnifiedTheme.GAME,
    UnifiedTheme.HALLOWEEN,
    UnifiedTheme.LOVE,
    UnifiedTheme.MEDITATION,
    UnifiedTheme.MORNING,
    UnifiedTheme.EMPTY,
    UnifiedTheme.PARTY,
    UnifiedTheme.SPORT,
    UnifiedTheme.SPRING,
    UnifiedTheme.SPRING_FESTIVAL,
    UnifiedTheme.SUMMER,
    UnifiedTheme.TRAVEL,
    UnifiedTheme.VALENTINES_DAY,
    UnifiedTheme.WEDDING,
    UnifiedTheme.WINTER,
    UnifiedTheme.YOGA,

    # SA sinking
    UnifiedSinking.SINKING,
    UnifiedSinking.NON_SINKING,

    # SA lang
    UnifiedLang.CANTONESE,
    UnifiedLang.CHINESE,
    UnifiedLang.CHINESE_DIALECT,
    UnifiedLang.ENGLISH,

    # Audio genre
    UnifiedGenre.TUHAI,
    UnifiedGenre.CHILDHOOD,
    UnifiedGenre.DEVOTIONAL,
    UnifiedGenre.NEW_AGE,
    UnifiedGenre.EIGHT_BIT,
    UnifiedGenre.ALTERNATIVE_INDIE,
    UnifiedGenre.EPIC,
    UnifiedGenre.BGM,
    UnifiedGenre.CHINESE_POP,
    UnifiedGenre.CANTOPOP,
    UnifiedGenre.TAIWANESE_POP,
    UnifiedGenre.POP_FOLK,
    UnifiedGenre.CONTEMPORARY_POP,
    UnifiedGenre.TEEN_POP,
    UnifiedGenre.INDIE_POP,
    UnifiedGenre.DREAM_POP,
    UnifiedGenre.CITY_POP,
    UnifiedGenre.SYNTH_POP,
    UnifiedGenre.CELTIC_POP,
    UnifiedGenre.DANCE_POP,
    UnifiedGenre.DEEP_DANCE_POP,
    UnifiedGenre.ELECTROPOP,
    UnifiedGenre.CHAMBER_POP,
    UnifiedGenre.A_CAPPELLA,
    UnifiedGenre.EDM,
    UnifiedGenre.HOUSE,
    UnifiedGenre.DUBSTEP,
    UnifiedGenre.FUTURE_BASS,
    UnifiedGenre.CHILLOUT,
    UnifiedGenre.TRANCE,
    UnifiedGenre.TECHNO,
    UnifiedGenre.DRUMNBASS,
    UnifiedGenre.TROPICAL_HOUSE,
    UnifiedGenre.DISCO,
    UnifiedGenre.VAPORWAVE,
    UnifiedGenre.TRIP_HOP,
    UnifiedGenre.AMBIENT,
    UnifiedGenre.DEEP_POP_EDM,
    UnifiedGenre.EDM_TRAP,
    UnifiedGenre.FUTURE_HOUSE,
    UnifiedGenre.CHINA_WAVE,
    UnifiedGenre.GUFENG_MUSIC,
    UnifiedGenre.CHINOISERIE_RAP,
    UnifiedGenre.CHINOISERIE_ELECTRONIC,
    UnifiedGenre.HARD_ROCK,
    UnifiedGenre.PSYCHEDELIC_ROCK,
    UnifiedGenre.POP_ROCK,
    UnifiedGenre.INSTRUMENTAL_ROCK,
    UnifiedGenre.ALTERNATIVE_ROCK,
    UnifiedGenre.INDIE_ROCK,
    UnifiedGenre.POST_ROCK,
    UnifiedGenre.LO_FI,
    UnifiedGenre.J_ROCK,
    UnifiedGenre.SHOEGAZING,
    UnifiedGenre.MATH_ROCK,
    UnifiedGenre.SURF_ROCK,
    UnifiedGenre.PROGRESSIVE_ROCK,
    UnifiedGenre.SOFT_ROCK,
    UnifiedGenre.JAZZ_POP,
    UnifiedGenre.JAZZ_FUSION,
    UnifiedGenre.BOSSA_NOVA,
    UnifiedGenre.AVANT_GARDE_JAZZ,
    UnifiedGenre.SWING,
    UnifiedGenre.BIG_BAND,
    UnifiedGenre.BOP,
    UnifiedGenre.POST_BOP,
    UnifiedGenre.SMOOTH_JAZZ,
    UnifiedGenre.COOL_JAZZ,
    UnifiedGenre.VOCAL_JAZZ,
    UnifiedGenre.NU_JAZZ,
    UnifiedGenre.FREE_JAZZ,
    UnifiedGenre.TRAP_RAP,
    UnifiedGenre.OLD_SCHOOL,
    UnifiedGenre.RNB_RAP,
    UnifiedGenre.JAZZ_HIP_HOP,
    UnifiedGenre.ALTERNATIVE_HIP_HOP,
    UnifiedGenre.INSTRUMENTAL_HIP_HOP,
    UnifiedGenre.POP_RAP,
    UnifiedGenre.HARDCORE_RAP,
    UnifiedGenre.COMEDY_HIP_HOP,
    UnifiedGenre.HIP_HOUSE,
    UnifiedGenre.CHILL_BEATS,
    UnifiedGenre.CHORUS,
    UnifiedGenre.CHAMBER_MUSIC,
    UnifiedGenre.SYMPHONY,
    UnifiedGenre.FUNK,
    UnifiedGenre.CONTEMPORARY_RNB,
    UnifiedGenre.NEO_SOUL,
    UnifiedGenre.SOUL,
    UnifiedGenre.POP_SOUL,
    UnifiedGenre.BLACK_METAL,
    UnifiedGenre.DEATH_METAL,
    UnifiedGenre.GLAM_METAL,
    UnifiedGenre.GRINDCORE,
    UnifiedGenre.POWER_METAL,
    UnifiedGenre.PROGRESSIVE_METAL,
    UnifiedGenre.SPEED_METAL,
    UnifiedGenre.VINAHOUSE,
    UnifiedGenre.GOSPEL,
    UnifiedGenre.HOLIDAY_MUSIC,
    UnifiedGenre.CHRISTIAN_MUSIC,
    UnifiedGenre.BUDDHIST_MUSIC,
    UnifiedGenre.TRADITIONAL_CHINESE_FOLK,
    UnifiedGenre.CHINESE_QUYI,
    UnifiedGenre.RED_SONG,
    UnifiedGenre.FOLK_POP,
    UnifiedGenre.INDIE_FOLK,
    UnifiedGenre.TANGO,
    UnifiedGenre.REGGAETON,

    # Audio mood
    UnifiedMood.GROOVY_FUNKY,
    UnifiedMood.NOSTALGIC_MEMORY,
    UnifiedMood.DREAMY_ETHEREAL,
    UnifiedMood.SHOCKING_MAGNIFICENT_EPIC,

    # Audio theme
    UnifiedTheme.NEW_YEAR,
    UnifiedTheme.QI_XI,
    UnifiedTheme.FUNERAL,
    UnifiedTheme.GRADUATION,
    UnifiedTheme.NATIONALS_DAY,
    UnifiedTheme.VLOG_DAILYLIFE,
    UnifiedTheme.PET_ANIMALS,
    UnifiedTheme.BEAUTY_FASHION,
    UnifiedTheme.ENTERTAINMENT,
    UnifiedTheme.BABIES,
    UnifiedTheme.CHILDREN,
    UnifiedTheme.TRANSITION,
    UnifiedTheme.ANIME,
    UnifiedTheme.WAKE_UP,
    UnifiedTheme.LANDSCAPE_SCENERY,
    UnifiedTheme.PRANK,
    UnifiedTheme.TIMELAPSE,
    UnifiedTheme.RAINY_DAY,
    UnifiedTheme.SUNNY_DAY,
    UnifiedTheme.BEACH,
    UnifiedTheme.NIGHTCLUB,
    UnifiedTheme.RESTAURANTS,
    UnifiedTheme.LOUNGE,
    UnifiedTheme.PARK,
    UnifiedTheme.MARKETPLACE,
    UnifiedTheme.UNIVERSE,
    UnifiedTheme.BAR,
    UnifiedTheme.THEATER_CONCERT_HALL,
    UnifiedTheme.COMMUTE,
    UnifiedTheme.ROADTRIP,
    UnifiedTheme.RELAXATION,
    UnifiedTheme.CLEANING_AND_CHORES,
    UnifiedTheme.RUNNING,
    UnifiedTheme.FLIRT,

    # Audio gender
    UnifiedGender.FEMALE,
    UnifiedGender.MALE,
    UnifiedGender.NEUTRAL,
    UnifiedGender.CHILD,
    UnifiedGender.ADULT,
    UnifiedGender.CHORUS,

    # Audio timbre
    UnifiedTimbre.WARM,
    UnifiedTimbre.ETHEREAL,
    UnifiedTimbre.HUSKY,
    UnifiedTimbre.DEEP,
    UnifiedTimbre.LOUD_AND_SONOROUS,
    UnifiedTimbre.EXTREME,
    UnifiedTimbre.SHARP,
    UnifiedTimbre.BRIGHT,
    UnifiedTimbre.SWEET,
    UnifiedTimbre.POWERFUL,
    UnifiedTimbre.SEXY_LAZY,
    UnifiedTimbre.MAGNETIC,
    UnifiedTimbre.CUTE,
    UnifiedTimbre.ELECTRIFIED_VOICE,

]


# A legacy table that is compatible with the SA vocab2id
UNIFIED_VOCAB2ID_LEGACY = lst_to_unified_vocab2id(_UNIFIED_VOCAB2ID_LEGACY_LST)


# =========================================================

# A vocab2id lookup table that supports SA tags and Audio tags V0, V1, and V2
_UNIFIED_VOCAB2ID_V1_LST = [
    UnifiedDumpster.NONE, 

    # SA genre
    UnifiedGenre.BLUES, 
    UnifiedGenre.CHINESE_OPERA, 
    UnifiedGenre.CHINESE_STYLE, 
    UnifiedGenre.CHINESE_TRADITION, 
    UnifiedGenre.CLASSICAL, 
    UnifiedGenre.COUNTRY_POP, 
    UnifiedGenre.DJ, 
    UnifiedGenre.EASY_LISTENING, 
    UnifiedGenre.ELECTRONIC, 
    UnifiedGenre.FOLK, 
    UnifiedGenre.HIP_HOP, 
    UnifiedGenre.JAZZ, 
    UnifiedGenre.LATIN, 
    UnifiedGenre.MC, 
    UnifiedGenre.METAL, 
    UnifiedGenre.EMPTY, 
    UnifiedGenre.POP, 
    UnifiedGenre.PUNK, 
    UnifiedGenre.RNB_SOUL, 
    UnifiedGenre.REGGAE, 
    UnifiedGenre.ROCK, 

    # SA mood
    UnifiedMood.ANGRY_AGGRESSIVE, 
    UnifiedMood.CALM_RELAXING, 
    UnifiedMood.CHILL, 
    UnifiedMood.CUTE_PLAYFUL, 
    UnifiedMood.DYNAMIC_ENERGETIC, 
    UnifiedMood.EXCITED, 
    UnifiedMood.FUNNY, 
    UnifiedMood.HAPPY, 
    UnifiedMood.HEALING, 
    UnifiedMood.INSPIRATIONAL_HOPEFUL, 
    UnifiedMood.SENTIMENTAL_MELANCHOLIC_LONELY, 
    UnifiedMood.MISS, 
    UnifiedMood.MYSTERIOUS, 
    UnifiedMood.EMPTY, 
    UnifiedMood.ROMANTIC, 
    UnifiedMood.SORROW_SAD, 
    UnifiedMood.SWEET, 
    UnifiedMood.THRILLING_SUSPENSEFUL_TENSE, 
    UnifiedMood.WEIRD,

    # SA theme
    UnifiedTheme.AUTUMN, 
    UnifiedTheme.BED_TIME, 
    UnifiedTheme.BIRTHDAY, 
    UnifiedTheme.BROKE_UP, 
    UnifiedTheme.COFFEE_SHOP, 
    UnifiedTheme.CAMPUS, 
    UnifiedTheme.CHRISTMAS, 
    UnifiedTheme.DANCE, 
    UnifiedTheme.DANCEABLE, 
    UnifiedTheme.DATE, 
    UnifiedTheme.DREAM, 
    UnifiedTheme.DRIVE, 
    UnifiedTheme.EVENING, 
    UnifiedTheme.FAMILY_TIME, 
    UnifiedTheme.FOCUS, 
    UnifiedTheme.FOOD, 
    UnifiedTheme.FRIENDSHIP, 
    UnifiedTheme.GAME, 
    UnifiedTheme.HALLOWEEN, 
    UnifiedTheme.LOVE, 
    UnifiedTheme.MEDITATION, 
    UnifiedTheme.MORNING, 
    UnifiedTheme.EMPTY, 
    UnifiedTheme.PARTY, 
    UnifiedTheme.SPORT, 
    UnifiedTheme.SPRING, 
    UnifiedTheme.SPRING_FESTIVAL, 
    UnifiedTheme.SUMMER, 
    UnifiedTheme.TRAVEL, 
    UnifiedTheme.VALENTINES_DAY, 
    UnifiedTheme.WEDDING, 
    UnifiedTheme.WINTER, 
    UnifiedTheme.YOGA, 

    # SA sinking
    UnifiedSinking.SINKING, 
    UnifiedSinking.NON_SINKING, 

    # SA lang
    UnifiedLang.CANTONESE, 
    UnifiedLang.CHINESE, 
    UnifiedLang.CHINESE_DIALECT, 
    UnifiedLang.ENGLISH, 

    # Audio genre V2
    UnifiedGenre.TUHAI, 
    UnifiedGenre.CHILDHOOD, 
    UnifiedGenre.DEVOTIONAL, 
    UnifiedGenre.NEW_AGE, 
    UnifiedGenre.EIGHT_BIT, 
    UnifiedGenre.ALTERNATIVE_INDIE, 
    UnifiedGenre.EPIC, 
    UnifiedGenre.BGM, 
    UnifiedGenre.CHINESE_POP, 
    UnifiedGenre.CANTOPOP, 
    UnifiedGenre.TAIWANESE_POP, 
    UnifiedGenre.POP_FOLK, 
    UnifiedGenre.CONTEMPORARY_POP, 
    UnifiedGenre.TEEN_POP, 
    UnifiedGenre.INDIE_POP, 
    UnifiedGenre.DREAM_POP, 
    UnifiedGenre.CITY_POP, 
    UnifiedGenre.SYNTH_POP, 
    UnifiedGenre.CELTIC_POP, 
    UnifiedGenre.DANCE_POP, 
    UnifiedGenre.DEEP_DANCE_POP, 
    UnifiedGenre.ELECTROPOP, 
    UnifiedGenre.CHAMBER_POP, 
    UnifiedGenre.A_CAPPELLA, 
    UnifiedGenre.EDM, 
    UnifiedGenre.HOUSE, 
    UnifiedGenre.DUBSTEP, 
    UnifiedGenre.FUTURE_BASS, 
    UnifiedGenre.CHILLOUT, 
    UnifiedGenre.TRANCE, 
    UnifiedGenre.TECHNO, 
    UnifiedGenre.DRUMNBASS, 
    UnifiedGenre.TROPICAL_HOUSE, 
    UnifiedGenre.DISCO, 
    UnifiedGenre.VAPORWAVE, 
    UnifiedGenre.TRIP_HOP, 
    UnifiedGenre.AMBIENT, 
    UnifiedGenre.DEEP_POP_EDM, 
    UnifiedGenre.EDM_TRAP, 
    UnifiedGenre.FUTURE_HOUSE, 
    UnifiedGenre.CHINA_WAVE, 
    UnifiedGenre.GUFENG_MUSIC, 
    UnifiedGenre.CHINOISERIE_RAP, 
    UnifiedGenre.CHINOISERIE_ELECTRONIC, 
    UnifiedGenre.HARD_ROCK, 
    UnifiedGenre.PSYCHEDELIC_ROCK, 
    UnifiedGenre.POP_ROCK, 
    UnifiedGenre.INSTRUMENTAL_ROCK, 
    UnifiedGenre.ALTERNATIVE_ROCK, 
    UnifiedGenre.INDIE_ROCK, 
    UnifiedGenre.POST_ROCK, 
    UnifiedGenre.LO_FI, 
    UnifiedGenre.J_ROCK, 
    UnifiedGenre.SHOEGAZING, 
    UnifiedGenre.MATH_ROCK, 
    UnifiedGenre.SURF_ROCK, 
    UnifiedGenre.PROGRESSIVE_ROCK, 
    UnifiedGenre.SOFT_ROCK, 
    UnifiedGenre.JAZZ_POP, 
    UnifiedGenre.JAZZ_FUSION, 
    UnifiedGenre.BOSSA_NOVA, 
    UnifiedGenre.AVANT_GARDE_JAZZ, 
    UnifiedGenre.SWING, 
    UnifiedGenre.BIG_BAND, 
    UnifiedGenre.BOP, 
    UnifiedGenre.POST_BOP, 
    UnifiedGenre.SMOOTH_JAZZ, 
    UnifiedGenre.COOL_JAZZ, 
    UnifiedGenre.VOCAL_JAZZ, 
    UnifiedGenre.NU_JAZZ, 
    UnifiedGenre.FREE_JAZZ, 
    UnifiedGenre.TRAP_RAP, 
    UnifiedGenre.OLD_SCHOOL, 
    UnifiedGenre.RNB_RAP, 
    UnifiedGenre.JAZZ_HIP_HOP, 
    UnifiedGenre.ALTERNATIVE_HIP_HOP, 
    UnifiedGenre.INSTRUMENTAL_HIP_HOP, 
    UnifiedGenre.POP_RAP, 
    UnifiedGenre.HARDCORE_RAP, 
    UnifiedGenre.COMEDY_HIP_HOP, 
    UnifiedGenre.HIP_HOUSE, 
    UnifiedGenre.CHILL_BEATS, 
    UnifiedGenre.CHORUS, 
    UnifiedGenre.CHAMBER_MUSIC, 
    UnifiedGenre.SYMPHONY, 
    UnifiedGenre.FUNK, 
    UnifiedGenre.CONTEMPORARY_RNB, 
    UnifiedGenre.NEO_SOUL, 
    UnifiedGenre.SOUL, 
    UnifiedGenre.POP_SOUL, 
    UnifiedGenre.BLACK_METAL, 
    UnifiedGenre.DEATH_METAL, 
    UnifiedGenre.GLAM_METAL, 
    UnifiedGenre.GRINDCORE, 
    UnifiedGenre.POWER_METAL, 
    UnifiedGenre.PROGRESSIVE_METAL, 
    UnifiedGenre.SPEED_METAL, 
    UnifiedGenre.VINAHOUSE, 
    UnifiedGenre.GOSPEL, 
    UnifiedGenre.HOLIDAY_MUSIC, 
    UnifiedGenre.CHRISTIAN_MUSIC, 
    UnifiedGenre.BUDDHIST_MUSIC, 
    UnifiedGenre.TRADITIONAL_CHINESE_FOLK, 
    UnifiedGenre.CHINESE_QUYI, 
    UnifiedGenre.RED_SONG, 
    UnifiedGenre.FOLK_POP, 
    UnifiedGenre.INDIE_FOLK, 
    UnifiedGenre.TANGO, 
    UnifiedGenre.REGGAETON, 
    UnifiedGenre.BOOMBAP, 
    UnifiedGenre.CHILLWAVE, 
    UnifiedGenre.CHINESE_FOLK, 
    UnifiedGenre.CONTEMPORARY_BLUES, 
    UnifiedGenre.CONTEMPORARY_FOLK, 
    UnifiedGenre.COUNTRY_BLUES, 
    UnifiedGenre.EMO_RAP, 
    UnifiedGenre.ENGLISH_FOLK, 
    UnifiedGenre.FLAMENCO, 
    UnifiedGenre.GANGSTA_RAP, 
    UnifiedGenre.HARDSTYLE, 
    UnifiedGenre.HEAVY_METAL, 
    UnifiedGenre.HINRG, 
    UnifiedGenre.INDIAN_POP, 
    UnifiedGenre.ITALIAN_POP, 
    UnifiedGenre.JAPANESE_FOLK, 
    UnifiedGenre.J_POP, 
    UnifiedGenre.KPOP, 
    UnifiedGenre.LATIN_POP, 
    UnifiedGenre.LOW_POP, 
    UnifiedGenre.MELBOURNE_BOUNCE, 
    UnifiedGenre.NEW_CHINESE_FOLK, 
    UnifiedGenre.NEW_WAVE, 
    UnifiedGenre.NOSTALGIC_POP, 
    UnifiedGenre.POP_PUNK, 
    UnifiedGenre.ROCK_BLUES, 
    UnifiedGenre.RUSSIAN_POP, 
    UnifiedGenre.SAMBA, 
    UnifiedGenre.SHI_DAI_QU, 
    UnifiedGenre.SOUNDTRACK, 
    UnifiedGenre.THRASH_METAL, 
    UnifiedGenre.TRADITIONAL_FOLK, 
    UnifiedGenre.TRADITIONAL_JAZZ, 
    UnifiedGenre.VULGAR_POP, 
    UnifiedGenre.WEST_COAST_HIP_HOP, 
    UnifiedGenre.WORLD_MUSIC, 

    # Audio mood V2
    UnifiedMood.GROOVY_FUNKY, 
    UnifiedMood.NOSTALGIC_MEMORY, 
    UnifiedMood.DREAMY_ETHEREAL, 
    UnifiedMood.SHOCKING_MAGNIFICENT_EPIC, 

    # Audio theme V2
    UnifiedTheme.NEW_YEAR, 
    UnifiedTheme.QI_XI, 
    UnifiedTheme.FUNERAL, 
    UnifiedTheme.GRADUATION, 
    UnifiedTheme.NATIONALS_DAY, 
    UnifiedTheme.VLOG_DAILYLIFE, 
    UnifiedTheme.PET_ANIMALS, 
    UnifiedTheme.BEAUTY_FASHION, 
    UnifiedTheme.ENTERTAINMENT, 
    UnifiedTheme.BABIES, 
    UnifiedTheme.CHILDREN, 
    UnifiedTheme.TRANSITION, 
    UnifiedTheme.ANIME, 
    UnifiedTheme.WAKE_UP, 
    UnifiedTheme.LANDSCAPE_SCENERY, 
    UnifiedTheme.PRANK, 
    UnifiedTheme.TIMELAPSE, 
    UnifiedTheme.RAINY_DAY, 
    UnifiedTheme.SUNNY_DAY, 
    UnifiedTheme.BEACH, 
    UnifiedTheme.NIGHTCLUB, 
    UnifiedTheme.RESTAURANTS, 
    UnifiedTheme.LOUNGE, 
    UnifiedTheme.PARK, 
    UnifiedTheme.MARKETPLACE, 
    UnifiedTheme.UNIVERSE, 
    UnifiedTheme.BAR, 
    UnifiedTheme.THEATER_CONCERT_HALL, 
    UnifiedTheme.COMMUTE, 
    UnifiedTheme.ROADTRIP, 
    UnifiedTheme.RELAXATION, 
    UnifiedTheme.CLEANING_AND_CHORES, 
    UnifiedTheme.RUNNING, 
    UnifiedTheme.FLIRT, 
    UnifiedTheme.MID_AUTUMN_FESTIVAL, 

    # Audio gender V2
    UnifiedGender.EMPTY, 
    UnifiedGender.NEUTRAL, 
    UnifiedGender.FEMALE, 
    UnifiedGender.MALE, 
    UnifiedGender.CHILD, 
    UnifiedGender.ADULT, 
    UnifiedGender.CHORUS, 

    # Audio timbre V2
    UnifiedTimbre.EMPTY, 
    UnifiedTimbre.WARM, 
    UnifiedTimbre.ETHEREAL, 
    UnifiedTimbre.HUSKY, 
    UnifiedTimbre.DEEP, 
    UnifiedTimbre.LOUD_AND_SONOROUS, 
    UnifiedTimbre.EXTREME, 
    UnifiedTimbre.SHARP, 
    UnifiedTimbre.BRIGHT, 
    UnifiedTimbre.SWEET, 
    UnifiedTimbre.POWERFUL, 
    UnifiedTimbre.SEXY_LAZY, 
    UnifiedTimbre.MAGNETIC, 
    UnifiedTimbre.CUTE, 
    UnifiedTimbre.ELECTRIFIED_VOICE, 
]


UNIFIED_VOCAB2ID_V1 = lst_to_unified_vocab2id(_UNIFIED_VOCAB2ID_V1_LST)

_UNIFIED_VOCAB2ID_V2_LST = [
    UnifiedDumpster.NONE,

    # SA genre
    UnifiedGenre.BLUES,
    UnifiedGenre.CHINESE_OPERA,
    UnifiedGenre.CHINESE_STYLE,
    UnifiedGenre.CHINESE_TRADITION,
    UnifiedGenre.CLASSICAL,
    UnifiedGenre.COUNTRY_POP,
    UnifiedGenre.DJ,
    UnifiedGenre.EASY_LISTENING,
    UnifiedGenre.ELECTRONIC,
    UnifiedGenre.FOLK,
    UnifiedGenre.HIP_HOP,
    UnifiedGenre.JAZZ,
    UnifiedGenre.LATIN,
    UnifiedGenre.MC,
    UnifiedGenre.METAL,
    UnifiedGenre.EMPTY,
    UnifiedGenre.POP,
    UnifiedGenre.PUNK,
    UnifiedGenre.RNB_SOUL,
    UnifiedGenre.REGGAE,
    UnifiedGenre.ROCK,

    # SA mood
    UnifiedMood.ANGRY_AGGRESSIVE,
    UnifiedMood.CALM_RELAXING,
    UnifiedMood.CHILL,
    UnifiedMood.CUTE_PLAYFUL,
    UnifiedMood.DYNAMIC_ENERGETIC,
    UnifiedMood.EXCITED,
    UnifiedMood.FUNNY,
    UnifiedMood.HAPPY,
    UnifiedMood.HEALING,
    UnifiedMood.INSPIRATIONAL_HOPEFUL,
    UnifiedMood.SENTIMENTAL_MELANCHOLIC_LONELY,
    UnifiedMood.MISS,
    UnifiedMood.MYSTERIOUS,
    UnifiedMood.EMPTY,
    UnifiedMood.ROMANTIC,
    UnifiedMood.SORROW_SAD,
    UnifiedMood.SWEET,
    UnifiedMood.THRILLING_SUSPENSEFUL_TENSE,
    UnifiedMood.WEIRD,

    # SA theme
    UnifiedTheme.AUTUMN,
    UnifiedTheme.BED_TIME,
    UnifiedTheme.BIRTHDAY,
    UnifiedTheme.BROKE_UP,
    UnifiedTheme.COFFEE_SHOP,
    UnifiedTheme.CAMPUS,
    UnifiedTheme.CHRISTMAS,
    UnifiedTheme.DANCE,
    UnifiedTheme.DANCEABLE,
    UnifiedTheme.DATE,
    UnifiedTheme.DREAM,
    UnifiedTheme.DRIVE,
    UnifiedTheme.EVENING,
    UnifiedTheme.FAMILY_TIME,
    UnifiedTheme.FOCUS,
    UnifiedTheme.FOOD,
    UnifiedTheme.FRIENDSHIP,
    UnifiedTheme.GAME,
    UnifiedTheme.HALLOWEEN,
    UnifiedTheme.LOVE,
    UnifiedTheme.MEDITATION,
    UnifiedTheme.MORNING,
    UnifiedTheme.EMPTY,
    UnifiedTheme.PARTY,
    UnifiedTheme.SPORT,
    UnifiedTheme.SPRING,
    UnifiedTheme.SPRING_FESTIVAL,
    UnifiedTheme.SUMMER,
    UnifiedTheme.TRAVEL,
    UnifiedTheme.VALENTINES_DAY,
    UnifiedTheme.WEDDING,
    UnifiedTheme.WINTER,
    UnifiedTheme.YOGA,

    # SA sinking
    UnifiedSinking.SINKING,
    UnifiedSinking.NON_SINKING,

    # SA lang
    UnifiedLang.CANTONESE,
    UnifiedLang.CHINESE,
    UnifiedLang.CHINESE_DIALECT,
    UnifiedLang.ENGLISH,

    # Audio genre V2
    UnifiedGenre.TUHAI,
    UnifiedGenre.CHILDHOOD,
    UnifiedGenre.DEVOTIONAL,
    UnifiedGenre.NEW_AGE,
    UnifiedGenre.EIGHT_BIT,
    UnifiedGenre.ALTERNATIVE_INDIE,
    UnifiedGenre.EPIC,
    UnifiedGenre.BGM,
    UnifiedGenre.CHINESE_POP,
    UnifiedGenre.CANTOPOP,
    UnifiedGenre.TAIWANESE_POP,
    UnifiedGenre.POP_FOLK,
    UnifiedGenre.CONTEMPORARY_POP,
    UnifiedGenre.TEEN_POP,
    UnifiedGenre.INDIE_POP,
    UnifiedGenre.DREAM_POP,
    UnifiedGenre.CITY_POP,
    UnifiedGenre.SYNTH_POP,
    UnifiedGenre.CELTIC_POP,
    UnifiedGenre.DANCE_POP,
    UnifiedGenre.DEEP_DANCE_POP,
    UnifiedGenre.ELECTROPOP,
    UnifiedGenre.CHAMBER_POP,
    UnifiedGenre.A_CAPPELLA,
    UnifiedGenre.EDM,
    UnifiedGenre.HOUSE,
    UnifiedGenre.DUBSTEP,
    UnifiedGenre.FUTURE_BASS,
    UnifiedGenre.CHILLOUT,
    UnifiedGenre.TRANCE,
    UnifiedGenre.TECHNO,
    UnifiedGenre.DRUMNBASS,
    UnifiedGenre.TROPICAL_HOUSE,
    UnifiedGenre.DISCO,
    UnifiedGenre.VAPORWAVE,
    UnifiedGenre.TRIP_HOP,
    UnifiedGenre.AMBIENT,
    UnifiedGenre.DEEP_POP_EDM,
    UnifiedGenre.EDM_TRAP,
    UnifiedGenre.FUTURE_HOUSE,
    UnifiedGenre.CHINA_WAVE,
    UnifiedGenre.GUFENG_MUSIC,
    UnifiedGenre.CHINOISERIE_RAP,
    UnifiedGenre.CHINOISERIE_ELECTRONIC,
    UnifiedGenre.HARD_ROCK,
    UnifiedGenre.PSYCHEDELIC_ROCK,
    UnifiedGenre.POP_ROCK,
    UnifiedGenre.INSTRUMENTAL_ROCK,
    UnifiedGenre.ALTERNATIVE_ROCK,
    UnifiedGenre.INDIE_ROCK,
    UnifiedGenre.POST_ROCK,
    UnifiedGenre.LO_FI,
    UnifiedGenre.J_ROCK,
    UnifiedGenre.SHOEGAZING,
    UnifiedGenre.MATH_ROCK,
    UnifiedGenre.SURF_ROCK,
    UnifiedGenre.PROGRESSIVE_ROCK,
    UnifiedGenre.SOFT_ROCK,
    UnifiedGenre.JAZZ_POP,
    UnifiedGenre.JAZZ_FUSION,
    UnifiedGenre.BOSSA_NOVA,
    UnifiedGenre.AVANT_GARDE_JAZZ,
    UnifiedGenre.SWING,
    UnifiedGenre.BIG_BAND,
    UnifiedGenre.BOP,
    UnifiedGenre.POST_BOP,
    UnifiedGenre.SMOOTH_JAZZ,
    UnifiedGenre.COOL_JAZZ,
    UnifiedGenre.VOCAL_JAZZ,
    UnifiedGenre.NU_JAZZ,
    UnifiedGenre.FREE_JAZZ,
    UnifiedGenre.TRAP_RAP,
    UnifiedGenre.OLD_SCHOOL,
    UnifiedGenre.RNB_RAP,
    UnifiedGenre.JAZZ_HIP_HOP,
    UnifiedGenre.ALTERNATIVE_HIP_HOP,
    UnifiedGenre.INSTRUMENTAL_HIP_HOP,
    UnifiedGenre.POP_RAP,
    UnifiedGenre.HARDCORE_RAP,
    UnifiedGenre.COMEDY_HIP_HOP,
    UnifiedGenre.HIP_HOUSE,
    UnifiedGenre.CHILL_BEATS,
    UnifiedGenre.CHORUS,
    UnifiedGenre.CHAMBER_MUSIC,
    UnifiedGenre.SYMPHONY,
    UnifiedGenre.FUNK,
    UnifiedGenre.CONTEMPORARY_RNB,
    UnifiedGenre.NEO_SOUL,
    UnifiedGenre.SOUL,
    UnifiedGenre.POP_SOUL,
    UnifiedGenre.BLACK_METAL,
    UnifiedGenre.DEATH_METAL,
    UnifiedGenre.GLAM_METAL,
    UnifiedGenre.GRINDCORE,
    UnifiedGenre.POWER_METAL,
    UnifiedGenre.PROGRESSIVE_METAL,
    UnifiedGenre.SPEED_METAL,
    UnifiedGenre.VINAHOUSE,
    UnifiedGenre.GOSPEL,
    UnifiedGenre.HOLIDAY_MUSIC,
    UnifiedGenre.CHRISTIAN_MUSIC,
    UnifiedGenre.BUDDHIST_MUSIC,
    UnifiedGenre.TRADITIONAL_CHINESE_FOLK,
    UnifiedGenre.CHINESE_QUYI,
    UnifiedGenre.RED_SONG,
    UnifiedGenre.FOLK_POP,
    UnifiedGenre.INDIE_FOLK,
    UnifiedGenre.TANGO,
    UnifiedGenre.REGGAETON,
    UnifiedGenre.BOOMBAP,
    UnifiedGenre.CHILLWAVE,
    UnifiedGenre.CHINESE_FOLK,
    UnifiedGenre.CONTEMPORARY_BLUES,
    UnifiedGenre.CONTEMPORARY_FOLK,
    UnifiedGenre.COUNTRY_BLUES,
    UnifiedGenre.EMO_RAP,
    UnifiedGenre.ENGLISH_FOLK,
    UnifiedGenre.FLAMENCO,
    UnifiedGenre.GANGSTA_RAP,
    UnifiedGenre.HARDSTYLE,
    UnifiedGenre.HEAVY_METAL,
    UnifiedGenre.HINRG,
    UnifiedGenre.INDIAN_POP,
    UnifiedGenre.ITALIAN_POP,
    UnifiedGenre.JAPANESE_FOLK,
    UnifiedGenre.J_POP,
    UnifiedGenre.KPOP,
    UnifiedGenre.LATIN_POP,
    UnifiedGenre.LOW_POP,
    UnifiedGenre.MELBOURNE_BOUNCE,
    UnifiedGenre.NEW_CHINESE_FOLK,
    UnifiedGenre.NEW_WAVE,
    UnifiedGenre.NOSTALGIC_POP,
    UnifiedGenre.POP_PUNK,
    UnifiedGenre.ROCK_BLUES,
    UnifiedGenre.RUSSIAN_POP,
    UnifiedGenre.SAMBA,
    UnifiedGenre.SHI_DAI_QU,
    UnifiedGenre.SOUNDTRACK,
    UnifiedGenre.THRASH_METAL,
    UnifiedGenre.TRADITIONAL_FOLK,
    UnifiedGenre.TRADITIONAL_JAZZ,
    UnifiedGenre.VULGAR_POP,
    UnifiedGenre.WEST_COAST_HIP_HOP,
    UnifiedGenre.WORLD_MUSIC,

    # Audio mood V2
    UnifiedMood.GROOVY_FUNKY,
    UnifiedMood.NOSTALGIC_MEMORY,
    UnifiedMood.DREAMY_ETHEREAL,
    UnifiedMood.SHOCKING_MAGNIFICENT_EPIC,

    # Audio theme V2
    UnifiedTheme.NEW_YEAR,
    UnifiedTheme.QI_XI,
    UnifiedTheme.FUNERAL,
    UnifiedTheme.GRADUATION,
    UnifiedTheme.NATIONALS_DAY,
    UnifiedTheme.VLOG_DAILYLIFE,
    UnifiedTheme.PET_ANIMALS,
    UnifiedTheme.BEAUTY_FASHION,
    UnifiedTheme.ENTERTAINMENT,
    UnifiedTheme.BABIES,
    UnifiedTheme.CHILDREN,
    UnifiedTheme.TRANSITION,
    UnifiedTheme.ANIME,
    UnifiedTheme.WAKE_UP,
    UnifiedTheme.LANDSCAPE_SCENERY,
    UnifiedTheme.PRANK,
    UnifiedTheme.TIMELAPSE,
    UnifiedTheme.RAINY_DAY,
    UnifiedTheme.SUNNY_DAY,
    UnifiedTheme.BEACH,
    UnifiedTheme.NIGHTCLUB,
    UnifiedTheme.RESTAURANTS,
    UnifiedTheme.LOUNGE,
    UnifiedTheme.PARK,
    UnifiedTheme.MARKETPLACE,
    UnifiedTheme.UNIVERSE,
    UnifiedTheme.BAR,
    UnifiedTheme.THEATER_CONCERT_HALL,
    UnifiedTheme.COMMUTE,
    UnifiedTheme.ROADTRIP,
    UnifiedTheme.RELAXATION,
    UnifiedTheme.CLEANING_AND_CHORES,
    UnifiedTheme.RUNNING,
    UnifiedTheme.FLIRT,
    UnifiedTheme.MID_AUTUMN_FESTIVAL,

    # Audio gender V2
    UnifiedGender.EMPTY,
    UnifiedGender.NEUTRAL,
    UnifiedGender.FEMALE,
    UnifiedGender.MALE,
    UnifiedGender.CHILD,
    UnifiedGender.ADULT,
    UnifiedGender.CHORUS,

    # Audio timbre V2
    UnifiedTimbre.EMPTY,
    UnifiedTimbre.WARM,
    UnifiedTimbre.ETHEREAL,
    UnifiedTimbre.HUSKY,
    UnifiedTimbre.DEEP,
    UnifiedTimbre.LOUD_AND_SONOROUS,
    UnifiedTimbre.EXTREME,
    UnifiedTimbre.SHARP,
    UnifiedTimbre.BRIGHT,
    UnifiedTimbre.SWEET,
    UnifiedTimbre.POWERFUL,
    UnifiedTimbre.SEXY_LAZY,
    UnifiedTimbre.MAGNETIC,
    UnifiedTimbre.CUTE,
    UnifiedTimbre.ELECTRIFIED_VOICE,

    # Audio genre V3.5
    UnifiedGenre.BREAKBEAT,
    UnifiedGenre.COUNTRY_FOLK,
    UnifiedGenre.COUNTRY_ROCK,
    UnifiedGenre.JAZZ_BLUES,
]


UNIFIED_VOCAB2ID_V2 = lst_to_unified_vocab2id(_UNIFIED_VOCAB2ID_V2_LST)

_UNIFIED_VOCAB2ID_V3_LST = [
    UnifiedDumpster.NONE,

    # SA genre
    UnifiedGenre.BLUES,
    UnifiedGenre.CHINESE_OPERA,
    UnifiedGenre.CHINESE_STYLE,
    UnifiedGenre.CHINESE_TRADITION,
    UnifiedGenre.CLASSICAL,
    UnifiedGenre.COUNTRY_POP,
    UnifiedGenre.DJ,
    UnifiedGenre.EASY_LISTENING,
    UnifiedGenre.ELECTRONIC,
    UnifiedGenre.FOLK,
    UnifiedGenre.HIP_HOP,
    UnifiedGenre.JAZZ,
    UnifiedGenre.LATIN,
    UnifiedGenre.MC,
    UnifiedGenre.METAL,
    UnifiedGenre.EMPTY,
    UnifiedGenre.POP,
    UnifiedGenre.PUNK,
    UnifiedGenre.RNB_SOUL,
    UnifiedGenre.REGGAE,
    UnifiedGenre.ROCK,

    # SA mood
    UnifiedMood.ANGRY_AGGRESSIVE,
    UnifiedMood.CALM_RELAXING,
    UnifiedMood.CHILL,
    UnifiedMood.CUTE_PLAYFUL,
    UnifiedMood.DYNAMIC_ENERGETIC,
    UnifiedMood.EXCITED,
    UnifiedMood.FUNNY,
    UnifiedMood.HAPPY,
    UnifiedMood.HEALING,
    UnifiedMood.INSPIRATIONAL_HOPEFUL,
    UnifiedMood.SENTIMENTAL_MELANCHOLIC_LONELY,
    UnifiedMood.MISS,
    UnifiedMood.MYSTERIOUS,
    UnifiedMood.EMPTY,
    UnifiedMood.ROMANTIC,
    UnifiedMood.SORROW_SAD,
    UnifiedMood.SWEET,
    UnifiedMood.THRILLING_SUSPENSEFUL_TENSE,
    UnifiedMood.WEIRD,

    # SA theme
    UnifiedTheme.AUTUMN,
    UnifiedTheme.BED_TIME,
    UnifiedTheme.BIRTHDAY,
    UnifiedTheme.BROKE_UP,
    UnifiedTheme.COFFEE_SHOP,
    UnifiedTheme.CAMPUS,
    UnifiedTheme.CHRISTMAS,
    UnifiedTheme.DANCE,
    UnifiedTheme.DANCEABLE,
    UnifiedTheme.DATE,
    UnifiedTheme.DREAM,
    UnifiedTheme.DRIVE,
    UnifiedTheme.EVENING,
    UnifiedTheme.FAMILY_TIME,
    UnifiedTheme.FOCUS,
    UnifiedTheme.FOOD,
    UnifiedTheme.FRIENDSHIP,
    UnifiedTheme.GAME,
    UnifiedTheme.HALLOWEEN,
    UnifiedTheme.LOVE,
    UnifiedTheme.MEDITATION,
    UnifiedTheme.MORNING,
    UnifiedTheme.EMPTY,
    UnifiedTheme.PARTY,
    UnifiedTheme.SPORT,
    UnifiedTheme.SPRING,
    UnifiedTheme.SPRING_FESTIVAL,
    UnifiedTheme.SUMMER,
    UnifiedTheme.TRAVEL,
    UnifiedTheme.VALENTINES_DAY,
    UnifiedTheme.WEDDING,
    UnifiedTheme.WINTER,
    UnifiedTheme.YOGA,

    # SA sinking
    UnifiedSinking.SINKING,
    UnifiedSinking.NON_SINKING,

    # SA lang
    UnifiedLang.CANTONESE,
    UnifiedLang.CHINESE,
    UnifiedLang.CHINESE_DIALECT,
    UnifiedLang.ENGLISH,

    # Audio genre V2
    UnifiedGenre.TUHAI,
    UnifiedGenre.CHILDHOOD,
    UnifiedGenre.DEVOTIONAL,
    UnifiedGenre.NEW_AGE,
    UnifiedGenre.EIGHT_BIT,
    UnifiedGenre.ALTERNATIVE_INDIE,
    UnifiedGenre.EPIC,
    UnifiedGenre.BGM,
    UnifiedGenre.CHINESE_POP,
    UnifiedGenre.CANTOPOP,
    UnifiedGenre.TAIWANESE_POP,
    UnifiedGenre.POP_FOLK,
    UnifiedGenre.CONTEMPORARY_POP,
    UnifiedGenre.TEEN_POP,
    UnifiedGenre.INDIE_POP,
    UnifiedGenre.DREAM_POP,
    UnifiedGenre.CITY_POP,
    UnifiedGenre.SYNTH_POP,
    UnifiedGenre.CELTIC_POP,
    UnifiedGenre.DANCE_POP,
    UnifiedGenre.DEEP_DANCE_POP,
    UnifiedGenre.ELECTROPOP,
    UnifiedGenre.CHAMBER_POP,
    UnifiedGenre.A_CAPPELLA,
    UnifiedGenre.EDM,
    UnifiedGenre.HOUSE,
    UnifiedGenre.DUBSTEP,
    UnifiedGenre.FUTURE_BASS,
    UnifiedGenre.CHILLOUT,
    UnifiedGenre.TRANCE,
    UnifiedGenre.TECHNO,
    UnifiedGenre.DRUMNBASS,
    UnifiedGenre.TROPICAL_HOUSE,
    UnifiedGenre.DISCO,
    UnifiedGenre.VAPORWAVE,
    UnifiedGenre.TRIP_HOP,
    UnifiedGenre.AMBIENT,
    UnifiedGenre.DEEP_POP_EDM,
    UnifiedGenre.EDM_TRAP,
    UnifiedGenre.FUTURE_HOUSE,
    UnifiedGenre.CHINA_WAVE,
    UnifiedGenre.GUFENG_MUSIC,
    UnifiedGenre.CHINOISERIE_RAP,
    UnifiedGenre.CHINOISERIE_ELECTRONIC,
    UnifiedGenre.HARD_ROCK,
    UnifiedGenre.PSYCHEDELIC_ROCK,
    UnifiedGenre.POP_ROCK,
    UnifiedGenre.INSTRUMENTAL_ROCK,
    UnifiedGenre.ALTERNATIVE_ROCK,
    UnifiedGenre.INDIE_ROCK,
    UnifiedGenre.POST_ROCK,
    UnifiedGenre.LO_FI,
    UnifiedGenre.J_ROCK,
    UnifiedGenre.SHOEGAZING,
    UnifiedGenre.MATH_ROCK,
    UnifiedGenre.SURF_ROCK,
    UnifiedGenre.PROGRESSIVE_ROCK,
    UnifiedGenre.SOFT_ROCK,
    UnifiedGenre.JAZZ_POP,
    UnifiedGenre.JAZZ_FUSION,
    UnifiedGenre.BOSSA_NOVA,
    UnifiedGenre.AVANT_GARDE_JAZZ,
    UnifiedGenre.SWING,
    UnifiedGenre.BIG_BAND,
    UnifiedGenre.BOP,
    UnifiedGenre.POST_BOP,
    UnifiedGenre.SMOOTH_JAZZ,
    UnifiedGenre.COOL_JAZZ,
    UnifiedGenre.VOCAL_JAZZ,
    UnifiedGenre.NU_JAZZ,
    UnifiedGenre.FREE_JAZZ,
    UnifiedGenre.TRAP_RAP,
    UnifiedGenre.OLD_SCHOOL,
    UnifiedGenre.RNB_RAP,
    UnifiedGenre.JAZZ_HIP_HOP,
    UnifiedGenre.ALTERNATIVE_HIP_HOP,
    UnifiedGenre.INSTRUMENTAL_HIP_HOP,
    UnifiedGenre.POP_RAP,
    UnifiedGenre.HARDCORE_RAP,
    UnifiedGenre.COMEDY_HIP_HOP,
    UnifiedGenre.HIP_HOUSE,
    UnifiedGenre.CHILL_BEATS,
    UnifiedGenre.CHORUS,
    UnifiedGenre.CHAMBER_MUSIC,
    UnifiedGenre.SYMPHONY,
    UnifiedGenre.FUNK,
    UnifiedGenre.CONTEMPORARY_RNB,
    UnifiedGenre.NEO_SOUL,
    UnifiedGenre.SOUL,
    UnifiedGenre.POP_SOUL,
    UnifiedGenre.BLACK_METAL,
    UnifiedGenre.DEATH_METAL,
    UnifiedGenre.GLAM_METAL,
    UnifiedGenre.GRINDCORE,
    UnifiedGenre.POWER_METAL,
    UnifiedGenre.PROGRESSIVE_METAL,
    UnifiedGenre.SPEED_METAL,
    UnifiedGenre.VINAHOUSE,
    UnifiedGenre.GOSPEL,
    UnifiedGenre.HOLIDAY_MUSIC,
    UnifiedGenre.CHRISTIAN_MUSIC,
    UnifiedGenre.BUDDHIST_MUSIC,
    UnifiedGenre.TRADITIONAL_CHINESE_FOLK,
    UnifiedGenre.CHINESE_QUYI,
    UnifiedGenre.RED_SONG,
    UnifiedGenre.FOLK_POP,
    UnifiedGenre.INDIE_FOLK,
    UnifiedGenre.TANGO,
    UnifiedGenre.REGGAETON,
    UnifiedGenre.BOOMBAP,
    UnifiedGenre.CHILLWAVE,
    UnifiedGenre.CHINESE_FOLK,
    UnifiedGenre.CONTEMPORARY_BLUES,
    UnifiedGenre.CONTEMPORARY_FOLK,
    UnifiedGenre.COUNTRY_BLUES,
    UnifiedGenre.EMO_RAP,
    UnifiedGenre.ENGLISH_FOLK,
    UnifiedGenre.FLAMENCO,
    UnifiedGenre.GANGSTA_RAP,
    UnifiedGenre.HARDSTYLE,
    UnifiedGenre.HEAVY_METAL,
    UnifiedGenre.HINRG,
    UnifiedGenre.INDIAN_POP,
    UnifiedGenre.ITALIAN_POP,
    UnifiedGenre.JAPANESE_FOLK,
    UnifiedGenre.J_POP,
    UnifiedGenre.KPOP,
    UnifiedGenre.LATIN_POP,
    UnifiedGenre.LOW_POP,
    UnifiedGenre.MELBOURNE_BOUNCE,
    UnifiedGenre.NEW_CHINESE_FOLK,
    UnifiedGenre.NEW_WAVE,
    UnifiedGenre.NOSTALGIC_POP,
    UnifiedGenre.POP_PUNK,
    UnifiedGenre.ROCK_BLUES,
    UnifiedGenre.RUSSIAN_POP,
    UnifiedGenre.SAMBA,
    UnifiedGenre.SHI_DAI_QU,
    UnifiedGenre.SOUNDTRACK,
    UnifiedGenre.THRASH_METAL,
    UnifiedGenre.TRADITIONAL_FOLK,
    UnifiedGenre.TRADITIONAL_JAZZ,
    UnifiedGenre.VULGAR_POP,
    UnifiedGenre.WEST_COAST_HIP_HOP,
    UnifiedGenre.WORLD_MUSIC,

    # Audio mood V2
    UnifiedMood.GROOVY_FUNKY,
    UnifiedMood.NOSTALGIC_MEMORY,
    UnifiedMood.DREAMY_ETHEREAL,
    UnifiedMood.SHOCKING_MAGNIFICENT_EPIC,

    # Audio theme V2
    UnifiedTheme.NEW_YEAR,
    UnifiedTheme.QI_XI,
    UnifiedTheme.FUNERAL,
    UnifiedTheme.GRADUATION,
    UnifiedTheme.NATIONALS_DAY,
    UnifiedTheme.VLOG_DAILYLIFE,
    UnifiedTheme.PET_ANIMALS,
    UnifiedTheme.BEAUTY_FASHION,
    UnifiedTheme.ENTERTAINMENT,
    UnifiedTheme.BABIES,
    UnifiedTheme.CHILDREN,
    UnifiedTheme.TRANSITION,
    UnifiedTheme.ANIME,
    UnifiedTheme.WAKE_UP,
    UnifiedTheme.LANDSCAPE_SCENERY,
    UnifiedTheme.PRANK,
    UnifiedTheme.TIMELAPSE,
    UnifiedTheme.RAINY_DAY,
    UnifiedTheme.SUNNY_DAY,
    UnifiedTheme.BEACH,
    UnifiedTheme.NIGHTCLUB,
    UnifiedTheme.RESTAURANTS,
    UnifiedTheme.LOUNGE,
    UnifiedTheme.PARK,
    UnifiedTheme.MARKETPLACE,
    UnifiedTheme.UNIVERSE,
    UnifiedTheme.BAR,
    UnifiedTheme.THEATER_CONCERT_HALL,
    UnifiedTheme.COMMUTE,
    UnifiedTheme.ROADTRIP,
    UnifiedTheme.RELAXATION,
    UnifiedTheme.CLEANING_AND_CHORES,
    UnifiedTheme.RUNNING,
    UnifiedTheme.FLIRT,
    UnifiedTheme.MID_AUTUMN_FESTIVAL,

    # Audio gender V2
    UnifiedGender.EMPTY,
    UnifiedGender.NEUTRAL,
    UnifiedGender.FEMALE,
    UnifiedGender.MALE,
    UnifiedGender.CHILD,
    UnifiedGender.ADULT,
    UnifiedGender.CHORUS,

    # Audio timbre V2
    UnifiedTimbre.EMPTY,
    UnifiedTimbre.WARM,
    UnifiedTimbre.ETHEREAL,
    UnifiedTimbre.HUSKY,
    UnifiedTimbre.DEEP,
    UnifiedTimbre.LOUD_AND_SONOROUS,
    UnifiedTimbre.EXTREME,
    UnifiedTimbre.SHARP,
    UnifiedTimbre.BRIGHT,
    UnifiedTimbre.SWEET,
    UnifiedTimbre.POWERFUL,
    UnifiedTimbre.SEXY_LAZY,
    UnifiedTimbre.MAGNETIC,
    UnifiedTimbre.CUTE,
    UnifiedTimbre.ELECTRIFIED_VOICE,

    # Audio genre V3.5
    UnifiedGenre.BREAKBEAT,
    UnifiedGenre.COUNTRY_FOLK,
    UnifiedGenre.COUNTRY_ROCK,
    UnifiedGenre.JAZZ_BLUES,

    # Audio genre V4
    UnifiedGenre.ACAPELLA,
    UnifiedGenre.ACID_JAZZ,
    UnifiedGenre.AFRICAN_FOLK,
    UnifiedGenre.AFRICAN_POP,
    UnifiedGenre.ARABIC_FOLK,
    UnifiedGenre.ARABIC_POP,
    UnifiedGenre.BAROQUE,
    UnifiedGenre.BASS_HOUSE,
    UnifiedGenre.BEHOP,
    UnifiedGenre.BHANGRA,
    UnifiedGenre.BLUE_GRASS,
    UnifiedGenre.BLUES_ROCK,
    UnifiedGenre.BOOGIE_WOOGIE,
    UnifiedGenre.BRAZILIAN_FUNK_STYLE,
    UnifiedGenre.BRIT_POP,
    UnifiedGenre.BUBBLEGUM_BASS,
    UnifiedGenre.CHINESE_BALLAD_POP,
    UnifiedGenre.CHINESE_TRADITIONAL_INSTRUMENTAL_MUSIC,
    UnifiedGenre.CLASSICAL_PERIOD,
    UnifiedGenre.CLASSIC_RNB_SOUL,
    UnifiedGenre.CONTEMPORARY_CLASSICAL_MUSIC,
    UnifiedGenre.DANCEHALL,
    UnifiedGenre.DANCE_PUNK,
    UnifiedGenre.DARK_AMBIENT,
    UnifiedGenre.DEATHCORE,
    UnifiedGenre.DOOM_METAL,
    UnifiedGenre.DOO_WOP,
    UnifiedGenre.DOWNTEMPO,
    UnifiedGenre.DRILL_RAP,
    UnifiedGenre.EAST_COAST_HIP_HOP,
    UnifiedGenre.EMO_PUNK,
    UnifiedGenre.ENKA,
    UnifiedGenre.FADO,
    UnifiedGenre.FOLK_METAL,
    UnifiedGenre.FOLKTRONICA,
    UnifiedGenre.FRENCH_FOLK,
    UnifiedGenre.FRENCH_POP,
    UnifiedGenre.GERMAN_POP,
    UnifiedGenre.GLITCH,
    UnifiedGenre.GOTHIC_METAL,
    UnifiedGenre.GRUNGE_ROCK,
    UnifiedGenre.GYPSY_JAZZ,
    UnifiedGenre.HARD_BOP,
    UnifiedGenre.HARDCORE,
    UnifiedGenre.HARDCORE_PUNK,
    UnifiedGenre.IDM,
    UnifiedGenre.IMPRESSIONISM,
    UnifiedGenre.INDUSTRIAL_METAL,
    UnifiedGenre.IRISH_FOLK,
    UnifiedGenre.JANPANESE_TRADITIONAL_MUSIC,
    UnifiedGenre.KOREAN_FOLK,
    UnifiedGenre.KOREA_TROT,
    UnifiedGenre.LATIN_ROCK,
    UnifiedGenre.LO_FI_HOUSE,
    UnifiedGenre.LO_FI_ROCK,
    UnifiedGenre.METALCORE,
    UnifiedGenre.MIDWEST_HIP_HOP,
    UnifiedGenre.MODERNISM,
    UnifiedGenre.MONGOLIAN_FOLK_SONGS,
    UnifiedGenre.NEO_FUNK,
    UnifiedGenre.NO_WAVE,
    UnifiedGenre.NU_METAL,
    UnifiedGenre.POST_HARDCORE,
    UnifiedGenre.POST_PUNK,
    UnifiedGenre.PROGRESSIVE_RNB,
    UnifiedGenre.RAGTIME,
    UnifiedGenre.RAP_METAL,
    UnifiedGenre.ROMANTIC_MUSIC,
    UnifiedGenre.RUMBA,
    UnifiedGenre.SCORE,
    UnifiedGenre.SHIMA_UTA,
    UnifiedGenre.SHOEGAZE_ROCK,
    UnifiedGenre.SKA_PUNK,
    UnifiedGenre.SON_CUBANO,
    UnifiedGenre.SOUTHERN_HIP_HOP,
    UnifiedGenre.SYNTHWAVE,
    UnifiedGenre.TRADITIONAL_BLUES,
    UnifiedGenre.TURKISH_POP,
    UnifiedGenre.UYGUR_FOLK_SONGS,
    UnifiedGenre.WORLDBEAT,
    UnifiedGenre.YODELING,

    # Audio lang V4
    UnifiedLang.EMPTY,
    UnifiedLang.AFRICAN,
    UnifiedLang.ARABIC,
    UnifiedLang.FRENCH,
    UnifiedLang.GERMAN,
    UnifiedLang.HINDI,
    UnifiedLang.INSTRUMENTAL,
    UnifiedLang.ITALIAN,
    UnifiedLang.JAPANESE,
    UnifiedLang.KOREA,
    UnifiedLang.RUSSIAN,
    UnifiedLang.SICHUANESE,
    UnifiedLang.TAIWANESE,
    UnifiedLang.TURKISH,

    # Audio gender V4
    UnifiedGender.MULTIPLE,

    # Audio extra V4
    UnifiedExtra.EMPTY,
    UnifiedExtra.LO_FI,
    UnifiedExtra.NOSTALGIC,
    UnifiedExtra.SOUNDTRACK,
    UnifiedExtra.TUHAI,
]


UNIFIED_VOCAB2ID_V3 = lst_to_unified_vocab2id(_UNIFIED_VOCAB2ID_V3_LST)


_UNIFIED_VOCAB2ID_V4_LST = [
    UnifiedDumpster.NONE,

    # SA genre
    UnifiedGenre.BLUES,
    UnifiedGenre.CHINESE_OPERA,
    UnifiedGenre.CHINESE_STYLE,
    UnifiedGenre.CHINESE_TRADITION,
    UnifiedGenre.CLASSICAL,
    UnifiedGenre.COUNTRY_POP,
    UnifiedGenre.DJ,
    UnifiedGenre.EASY_LISTENING,
    UnifiedGenre.ELECTRONIC,
    UnifiedGenre.FOLK,
    UnifiedGenre.HIP_HOP,
    UnifiedGenre.JAZZ,
    UnifiedGenre.LATIN,
    UnifiedGenre.MC,
    UnifiedGenre.METAL,
    UnifiedGenre.EMPTY,
    UnifiedGenre.POP,
    UnifiedGenre.PUNK,
    UnifiedGenre.RNB_SOUL,
    UnifiedGenre.REGGAE,
    UnifiedGenre.ROCK,

    # SA mood
    UnifiedMood.ANGRY_AGGRESSIVE,
    UnifiedMood.CALM_RELAXING,
    UnifiedMood.CHILL,
    UnifiedMood.CUTE_PLAYFUL,
    UnifiedMood.DYNAMIC_ENERGETIC,
    UnifiedMood.EXCITED,
    UnifiedMood.FUNNY,
    UnifiedMood.HAPPY,
    UnifiedMood.HEALING,
    UnifiedMood.INSPIRATIONAL_HOPEFUL,
    UnifiedMood.SENTIMENTAL_MELANCHOLIC_LONELY,
    UnifiedMood.MISS,
    UnifiedMood.MYSTERIOUS,
    UnifiedMood.EMPTY,
    UnifiedMood.ROMANTIC,
    UnifiedMood.SORROW_SAD,
    UnifiedMood.SWEET,
    UnifiedMood.THRILLING_SUSPENSEFUL_TENSE,
    UnifiedMood.WEIRD,

    # SA theme
    UnifiedTheme.AUTUMN,
    UnifiedTheme.BED_TIME,
    UnifiedTheme.BIRTHDAY,
    UnifiedTheme.BROKE_UP,
    UnifiedTheme.COFFEE_SHOP,
    UnifiedTheme.CAMPUS,
    UnifiedTheme.CHRISTMAS,
    UnifiedTheme.DANCE,
    UnifiedTheme.DANCEABLE,
    UnifiedTheme.DATE,
    UnifiedTheme.DREAM,
    UnifiedTheme.DRIVE,
    UnifiedTheme.EVENING,
    UnifiedTheme.FAMILY_TIME,
    UnifiedTheme.FOCUS,
    UnifiedTheme.FOOD,
    UnifiedTheme.FRIENDSHIP,
    UnifiedTheme.GAME,
    UnifiedTheme.HALLOWEEN,
    UnifiedTheme.LOVE,
    UnifiedTheme.MEDITATION,
    UnifiedTheme.MORNING,
    UnifiedTheme.EMPTY,
    UnifiedTheme.PARTY,
    UnifiedTheme.SPORT,
    UnifiedTheme.SPRING,
    UnifiedTheme.SPRING_FESTIVAL,
    UnifiedTheme.SUMMER,
    UnifiedTheme.TRAVEL,
    UnifiedTheme.VALENTINES_DAY,
    UnifiedTheme.WEDDING,
    UnifiedTheme.WINTER,
    UnifiedTheme.YOGA,

    # SA sinking
    UnifiedSinking.SINKING,
    UnifiedSinking.NON_SINKING,

    # SA lang
    UnifiedLang.CANTONESE,
    UnifiedLang.CHINESE,
    UnifiedLang.CHINESE_DIALECT,
    UnifiedLang.ENGLISH,

    # Audio genre V2
    UnifiedGenre.TUHAI,
    UnifiedGenre.CHILDHOOD,
    UnifiedGenre.DEVOTIONAL,
    UnifiedGenre.NEW_AGE,
    UnifiedGenre.EIGHT_BIT,
    UnifiedGenre.ALTERNATIVE_INDIE,
    UnifiedGenre.EPIC,
    UnifiedGenre.BGM,
    UnifiedGenre.CHINESE_POP,
    UnifiedGenre.CANTOPOP,
    UnifiedGenre.TAIWANESE_POP,
    UnifiedGenre.POP_FOLK,
    UnifiedGenre.CONTEMPORARY_POP,
    UnifiedGenre.TEEN_POP,
    UnifiedGenre.INDIE_POP,
    UnifiedGenre.DREAM_POP,
    UnifiedGenre.CITY_POP,
    UnifiedGenre.SYNTH_POP,
    UnifiedGenre.CELTIC_POP,
    UnifiedGenre.DANCE_POP,
    UnifiedGenre.DEEP_DANCE_POP,
    UnifiedGenre.ELECTROPOP,
    UnifiedGenre.CHAMBER_POP,
    UnifiedGenre.A_CAPPELLA,
    UnifiedGenre.EDM,
    UnifiedGenre.HOUSE,
    UnifiedGenre.DUBSTEP,
    UnifiedGenre.FUTURE_BASS,
    UnifiedGenre.CHILLOUT,
    UnifiedGenre.TRANCE,
    UnifiedGenre.TECHNO,
    UnifiedGenre.DRUMNBASS,
    UnifiedGenre.TROPICAL_HOUSE,
    UnifiedGenre.DISCO,
    UnifiedGenre.VAPORWAVE,
    UnifiedGenre.TRIP_HOP,
    UnifiedGenre.AMBIENT,
    UnifiedGenre.DEEP_POP_EDM,
    UnifiedGenre.EDM_TRAP,
    UnifiedGenre.FUTURE_HOUSE,
    UnifiedGenre.CHINA_WAVE,
    UnifiedGenre.GUFENG_MUSIC,
    UnifiedGenre.CHINOISERIE_RAP,
    UnifiedGenre.CHINOISERIE_ELECTRONIC,
    UnifiedGenre.HARD_ROCK,
    UnifiedGenre.PSYCHEDELIC_ROCK,
    UnifiedGenre.POP_ROCK,
    UnifiedGenre.INSTRUMENTAL_ROCK,
    UnifiedGenre.ALTERNATIVE_ROCK,
    UnifiedGenre.INDIE_ROCK,
    UnifiedGenre.POST_ROCK,
    UnifiedGenre.LO_FI,
    UnifiedGenre.J_ROCK,
    UnifiedGenre.SHOEGAZING,
    UnifiedGenre.MATH_ROCK,
    UnifiedGenre.SURF_ROCK,
    UnifiedGenre.PROGRESSIVE_ROCK,
    UnifiedGenre.SOFT_ROCK,
    UnifiedGenre.JAZZ_POP,
    UnifiedGenre.JAZZ_FUSION,
    UnifiedGenre.BOSSA_NOVA,
    UnifiedGenre.AVANT_GARDE_JAZZ,
    UnifiedGenre.SWING,
    UnifiedGenre.BIG_BAND,
    UnifiedGenre.BOP,
    UnifiedGenre.POST_BOP,
    UnifiedGenre.SMOOTH_JAZZ,
    UnifiedGenre.COOL_JAZZ,
    UnifiedGenre.VOCAL_JAZZ,
    UnifiedGenre.NU_JAZZ,
    UnifiedGenre.FREE_JAZZ,
    UnifiedGenre.TRAP_RAP,
    UnifiedGenre.OLD_SCHOOL,
    UnifiedGenre.RNB_RAP,
    UnifiedGenre.JAZZ_HIP_HOP,
    UnifiedGenre.ALTERNATIVE_HIP_HOP,
    UnifiedGenre.INSTRUMENTAL_HIP_HOP,
    UnifiedGenre.POP_RAP,
    UnifiedGenre.HARDCORE_RAP,
    UnifiedGenre.COMEDY_HIP_HOP,
    UnifiedGenre.HIP_HOUSE,
    UnifiedGenre.CHILL_BEATS,
    UnifiedGenre.CHORUS,
    UnifiedGenre.CHAMBER_MUSIC,
    UnifiedGenre.SYMPHONY,
    UnifiedGenre.FUNK,
    UnifiedGenre.CONTEMPORARY_RNB,
    UnifiedGenre.NEO_SOUL,
    UnifiedGenre.SOUL,
    UnifiedGenre.POP_SOUL,
    UnifiedGenre.BLACK_METAL,
    UnifiedGenre.DEATH_METAL,
    UnifiedGenre.GLAM_METAL,
    UnifiedGenre.GRINDCORE,
    UnifiedGenre.POWER_METAL,
    UnifiedGenre.PROGRESSIVE_METAL,
    UnifiedGenre.SPEED_METAL,
    UnifiedGenre.VINAHOUSE,
    UnifiedGenre.GOSPEL,
    UnifiedGenre.HOLIDAY_MUSIC,
    UnifiedGenre.CHRISTIAN_MUSIC,
    UnifiedGenre.BUDDHIST_MUSIC,
    UnifiedGenre.TRADITIONAL_CHINESE_FOLK,
    UnifiedGenre.CHINESE_QUYI,
    UnifiedGenre.RED_SONG,
    UnifiedGenre.FOLK_POP,
    UnifiedGenre.INDIE_FOLK,
    UnifiedGenre.TANGO,
    UnifiedGenre.REGGAETON,
    UnifiedGenre.BOOMBAP,
    UnifiedGenre.CHILLWAVE,
    UnifiedGenre.CHINESE_FOLK,
    UnifiedGenre.CONTEMPORARY_BLUES,
    UnifiedGenre.CONTEMPORARY_FOLK,
    UnifiedGenre.COUNTRY_BLUES,
    UnifiedGenre.EMO_RAP,
    UnifiedGenre.ENGLISH_FOLK,
    UnifiedGenre.FLAMENCO,
    UnifiedGenre.GANGSTA_RAP,
    UnifiedGenre.HARDSTYLE,
    UnifiedGenre.HEAVY_METAL,
    UnifiedGenre.HINRG,
    UnifiedGenre.INDIAN_POP,
    UnifiedGenre.ITALIAN_POP,
    UnifiedGenre.JAPANESE_FOLK,
    UnifiedGenre.J_POP,
    UnifiedGenre.KPOP,
    UnifiedGenre.LATIN_POP,
    UnifiedGenre.LOW_POP,
    UnifiedGenre.MELBOURNE_BOUNCE,
    UnifiedGenre.NEW_CHINESE_FOLK,
    UnifiedGenre.NEW_WAVE,
    UnifiedGenre.NOSTALGIC_POP,
    UnifiedGenre.POP_PUNK,
    UnifiedGenre.ROCK_BLUES,
    UnifiedGenre.RUSSIAN_POP,
    UnifiedGenre.SAMBA,
    UnifiedGenre.SHI_DAI_QU,
    UnifiedGenre.SOUNDTRACK,
    UnifiedGenre.THRASH_METAL,
    UnifiedGenre.TRADITIONAL_FOLK,
    UnifiedGenre.TRADITIONAL_JAZZ,
    UnifiedGenre.VULGAR_POP,
    UnifiedGenre.WEST_COAST_HIP_HOP,
    UnifiedGenre.WORLD_MUSIC,

    # Audio mood V2
    UnifiedMood.GROOVY_FUNKY,
    UnifiedMood.NOSTALGIC_MEMORY,
    UnifiedMood.DREAMY_ETHEREAL,
    UnifiedMood.SHOCKING_MAGNIFICENT_EPIC,

    # Audio theme V2
    UnifiedTheme.NEW_YEAR,
    UnifiedTheme.QI_XI,
    UnifiedTheme.FUNERAL,
    UnifiedTheme.GRADUATION,
    UnifiedTheme.NATIONALS_DAY,
    UnifiedTheme.VLOG_DAILYLIFE,
    UnifiedTheme.PET_ANIMALS,
    UnifiedTheme.BEAUTY_FASHION,
    UnifiedTheme.ENTERTAINMENT,
    UnifiedTheme.BABIES,
    UnifiedTheme.CHILDREN,
    UnifiedTheme.TRANSITION,
    UnifiedTheme.ANIME,
    UnifiedTheme.WAKE_UP,
    UnifiedTheme.LANDSCAPE_SCENERY,
    UnifiedTheme.PRANK,
    UnifiedTheme.TIMELAPSE,
    UnifiedTheme.RAINY_DAY,
    UnifiedTheme.SUNNY_DAY,
    UnifiedTheme.BEACH,
    UnifiedTheme.NIGHTCLUB,
    UnifiedTheme.RESTAURANTS,
    UnifiedTheme.LOUNGE,
    UnifiedTheme.PARK,
    UnifiedTheme.MARKETPLACE,
    UnifiedTheme.UNIVERSE,
    UnifiedTheme.BAR,
    UnifiedTheme.THEATER_CONCERT_HALL,
    UnifiedTheme.COMMUTE,
    UnifiedTheme.ROADTRIP,
    UnifiedTheme.RELAXATION,
    UnifiedTheme.CLEANING_AND_CHORES,
    UnifiedTheme.RUNNING,
    UnifiedTheme.FLIRT,
    UnifiedTheme.MID_AUTUMN_FESTIVAL,

    # Audio gender V2
    UnifiedGender.EMPTY,
    UnifiedGender.NEUTRAL,
    UnifiedGender.FEMALE,
    UnifiedGender.MALE,
    UnifiedGender.CHILD,
    UnifiedGender.ADULT,
    UnifiedGender.CHORUS,

    # Audio timbre V2
    UnifiedTimbre.EMPTY,
    UnifiedTimbre.WARM,
    UnifiedTimbre.ETHEREAL,
    UnifiedTimbre.HUSKY,
    UnifiedTimbre.DEEP,
    UnifiedTimbre.LOUD_AND_SONOROUS,
    UnifiedTimbre.EXTREME,
    UnifiedTimbre.SHARP,
    UnifiedTimbre.BRIGHT,
    UnifiedTimbre.SWEET,
    UnifiedTimbre.POWERFUL,
    UnifiedTimbre.SEXY_LAZY,
    UnifiedTimbre.MAGNETIC,
    UnifiedTimbre.CUTE,
    UnifiedTimbre.ELECTRIFIED_VOICE,

    # Audio genre V3.5
    UnifiedGenre.BREAKBEAT,
    UnifiedGenre.COUNTRY_FOLK,
    UnifiedGenre.COUNTRY_ROCK,
    UnifiedGenre.JAZZ_BLUES,

    # Audio genre V4
    UnifiedGenre.ACAPELLA,
    UnifiedGenre.ACID_JAZZ,
    UnifiedGenre.AFRICAN_FOLK,
    UnifiedGenre.AFRICAN_POP,
    UnifiedGenre.ARABIC_FOLK,
    UnifiedGenre.ARABIC_POP,
    UnifiedGenre.BAROQUE,
    UnifiedGenre.BASS_HOUSE,
    UnifiedGenre.BEHOP,
    UnifiedGenre.BHANGRA,
    UnifiedGenre.BLUE_GRASS,
    UnifiedGenre.BLUES_ROCK,
    UnifiedGenre.BOOGIE_WOOGIE,
    UnifiedGenre.BRAZILIAN_FUNK_STYLE,
    UnifiedGenre.BRIT_POP,
    UnifiedGenre.BUBBLEGUM_BASS,
    UnifiedGenre.CHINESE_BALLAD_POP,
    UnifiedGenre.CHINESE_TRADITIONAL_INSTRUMENTAL_MUSIC,
    UnifiedGenre.CLASSICAL_PERIOD,
    UnifiedGenre.CLASSIC_RNB_SOUL,
    UnifiedGenre.CONTEMPORARY_CLASSICAL_MUSIC,
    UnifiedGenre.DANCEHALL,
    UnifiedGenre.DANCE_PUNK,
    UnifiedGenre.DARK_AMBIENT,
    UnifiedGenre.DEATHCORE,
    UnifiedGenre.DOOM_METAL,
    UnifiedGenre.DOO_WOP,
    UnifiedGenre.DOWNTEMPO,
    UnifiedGenre.DRILL_RAP,
    UnifiedGenre.EAST_COAST_HIP_HOP,
    UnifiedGenre.EMO_PUNK,
    UnifiedGenre.ENKA,
    UnifiedGenre.FADO,
    UnifiedGenre.FOLK_METAL,
    UnifiedGenre.FOLKTRONICA,
    UnifiedGenre.FRENCH_FOLK,
    UnifiedGenre.FRENCH_POP,
    UnifiedGenre.GERMAN_POP,
    UnifiedGenre.GLITCH,
    UnifiedGenre.GOTHIC_METAL,
    UnifiedGenre.GRUNGE_ROCK,
    UnifiedGenre.GYPSY_JAZZ,
    UnifiedGenre.HARD_BOP,
    UnifiedGenre.HARDCORE,
    UnifiedGenre.HARDCORE_PUNK,
    UnifiedGenre.IDM,
    UnifiedGenre.IMPRESSIONISM,
    UnifiedGenre.INDUSTRIAL_METAL,
    UnifiedGenre.IRISH_FOLK,
    UnifiedGenre.JANPANESE_TRADITIONAL_MUSIC,
    UnifiedGenre.KOREAN_FOLK,
    UnifiedGenre.KOREA_TROT,
    UnifiedGenre.LATIN_ROCK,
    UnifiedGenre.LO_FI_HOUSE,
    UnifiedGenre.LO_FI_ROCK,
    UnifiedGenre.METALCORE,
    UnifiedGenre.MIDWEST_HIP_HOP,
    UnifiedGenre.MODERNISM,
    UnifiedGenre.MONGOLIAN_FOLK_SONGS,
    UnifiedGenre.NEO_FUNK,
    UnifiedGenre.NO_WAVE,
    UnifiedGenre.NU_METAL,
    UnifiedGenre.POST_HARDCORE,
    UnifiedGenre.POST_PUNK,
    UnifiedGenre.PROGRESSIVE_RNB,
    UnifiedGenre.RAGTIME,
    UnifiedGenre.RAP_METAL,
    UnifiedGenre.ROMANTIC_MUSIC,
    UnifiedGenre.RUMBA,
    UnifiedGenre.SCORE,
    UnifiedGenre.SHIMA_UTA,
    UnifiedGenre.SHOEGAZE_ROCK,
    UnifiedGenre.SKA_PUNK,
    UnifiedGenre.SON_CUBANO,
    UnifiedGenre.SOUTHERN_HIP_HOP,
    UnifiedGenre.SYNTHWAVE,
    UnifiedGenre.TRADITIONAL_BLUES,
    UnifiedGenre.TURKISH_POP,
    UnifiedGenre.UYGUR_FOLK_SONGS,
    UnifiedGenre.WORLDBEAT,
    UnifiedGenre.YODELING,

    # Audio lang V4
    UnifiedLang.EMPTY,
    UnifiedLang.AFRICAN,
    UnifiedLang.ARABIC,
    UnifiedLang.FRENCH,
    UnifiedLang.GERMAN,
    UnifiedLang.HINDI,
    UnifiedLang.INSTRUMENTAL,
    UnifiedLang.ITALIAN,
    UnifiedLang.JAPANESE,
    UnifiedLang.KOREA,
    UnifiedLang.RUSSIAN,
    UnifiedLang.SICHUANESE,
    UnifiedLang.TAIWANESE,
    UnifiedLang.TURKISH,

    # Audio gender V4
    UnifiedGender.MULTIPLE,

    # Audio extra V4
    UnifiedExtra.EMPTY,
    UnifiedExtra.LO_FI,
    UnifiedExtra.NOSTALGIC,
    UnifiedExtra.SOUNDTRACK,
    UnifiedExtra.TUHAI,


    # Instrument Karok Taxonomy @rui.xia need to change to unified taxnonmy in the future
    UnifiedInstrument.EMPTY,
    UnifiedInstrument.WURLITZER,
    UnifiedInstrument.FRENCH_HORN,
    UnifiedInstrument.BONGOS,
    UnifiedInstrument.DOUBLE_BASS,
    UnifiedInstrument.PAN_FLUTE,
    UnifiedInstrument.LEAD_ACOUSTIC_GUITAR,
    UnifiedInstrument.MANDOLIN,
    UnifiedInstrument.TALK_BOX,
    UnifiedInstrument.TABLA,
    UnifiedInstrument.TRADITIONAL_WIND_INSTRUMENTS,
    UnifiedInstrument.STRING_SECTION,
    UnifiedInstrument.VIBES,
    UnifiedInstrument.RHYTHM_ACOUSTIC_GUITAR_ARPEGGIO,
    UnifiedInstrument.ELECTRONIC_PERCUSSION,
    UnifiedInstrument.SYNTH_BASS,
    UnifiedInstrument.SYNTH_BRASS,
    UnifiedInstrument.BACKING_VOCALS,
    UnifiedInstrument.SEA,
    UnifiedInstrument.CELESTA,
    UnifiedInstrument.FEMALE_LEAD_VOCAL,
    UnifiedInstrument.MUSIC_BOX,
    UnifiedInstrument.MOOG_SYNTHESIZER,
    UnifiedInstrument.UKULELE,
    UnifiedInstrument.TENOR_SAXOPHONE,
    UnifiedInstrument.PEDAL_STEEL_GUITAR,
    UnifiedInstrument.RAINSTICK,
    UnifiedInstrument.SYNTH_FLUTE,
    UnifiedInstrument.ENGLISH_HORN,
    UnifiedInstrument.ACOUSTIC_BASS_GUITAR,
    UnifiedInstrument.ELECTRONIC_HI_HAT,
    UnifiedInstrument.FINGER_CYMBALS,
    UnifiedInstrument.STEEL_DRUMS,
    UnifiedInstrument.DIDGERIDOO,
    UnifiedInstrument.DRUM,
    UnifiedInstrument.BASSOON,
    UnifiedInstrument.KOTO,
    UnifiedInstrument.CASTANETS,
    UnifiedInstrument.DISTORTED_ELECTRIC_GUITAR,
    UnifiedInstrument.SHAMISEN,
    UnifiedInstrument.VOICE,
    UnifiedInstrument.BASS,
    UnifiedInstrument.KORA,
    UnifiedInstrument.MUTED_TROMBONE,
    UnifiedInstrument.PIANO,
    UnifiedInstrument.RHYTHM_ELECTRIC_GUITAR,
    UnifiedInstrument.LEAD_ELECTRIC_GUITAR,
    UnifiedInstrument.PITCHED_PERCUSSION,
    UnifiedInstrument.TUBA,
    UnifiedInstrument.ARR_ACOUSTIC_GUITAR,
    UnifiedInstrument.CP,
    UnifiedInstrument.SURDO,
    UnifiedInstrument.BRASS_AND_HARMONICA,
    UnifiedInstrument.SANDPAPER,
    UnifiedInstrument.CHARANGO,
    UnifiedInstrument.KEYBOARD,
    UnifiedInstrument.FIDDLE,
    UnifiedInstrument.SOPRANO_SAXOPHONE,
    UnifiedInstrument.POLYSYNTH,
    UnifiedInstrument.VINYL,
    UnifiedInstrument.SOUND_EFFECTS,
    UnifiedInstrument.HUMAN_BEATBOX,
    UnifiedInstrument.BREATH,
    UnifiedInstrument.RHYTHM_ACOUSTIC_GUITAR,
    UnifiedInstrument.HARMONICA,
    UnifiedInstrument.BALALAIKA,
    UnifiedInstrument.CLARINET,
    UnifiedInstrument.TIMPANI,
    UnifiedInstrument.XYLOPHONE,
    UnifiedInstrument.CLAP,
    UnifiedInstrument.TOMS,
    UnifiedInstrument.SAMPLE,
    UnifiedInstrument.GUTHBUINNE,
    UnifiedInstrument.METAL_BARS,
    UnifiedInstrument.OTHER_GUITAR,
    UnifiedInstrument.SNAP_FINGERS,
    UnifiedInstrument.CAVAQUINHO,
    UnifiedInstrument.GUITAR,
    UnifiedInstrument.MARACAS,
    UnifiedInstrument.TAMBOURINE,
    UnifiedInstrument.TOY_PIANO,
    UnifiedInstrument.LAP_STEEL_GUITAR,
    UnifiedInstrument.MALE_LEAD_VOCAL,
    UnifiedInstrument.HAMMOND,
    UnifiedInstrument.WHISTL,
    UnifiedInstrument.DULCIMER,
    UnifiedInstrument.ORCHESTRAL_PERCUSSION,
    UnifiedInstrument.SPOONS,
    UnifiedInstrument.GONG,
    UnifiedInstrument.SYNTH_PAD,
    UnifiedInstrument.ELECTRONIC_CYMBALS,
    UnifiedInstrument.BARITONE_SAXOPHONE,
    UnifiedInstrument.MALE_BACKING_VOCALS,
    UnifiedInstrument.MARIMBA,
    UnifiedInstrument.RHYTHM_ELECTRIC_GUITAR_ARPEGGIO,
    UnifiedInstrument.ACOUSTIC_DRUM_KIT,
    UnifiedInstrument.SAXOPHONE,
    UnifiedInstrument.DJEMBE,
    UnifiedInstrument.LEAD_VOCAL,
    UnifiedInstrument.SNARE_DRUM,
    UnifiedInstrument.CLAVES_WOODBLOCK,
    UnifiedInstrument.WIND_CHIMES,
    UnifiedInstrument.MUSICAL_SAW,
    UnifiedInstrument.LATIN_PERCUSSION,
    UnifiedInstrument.LOOP,
    UnifiedInstrument.WOODEN_BARS,
    UnifiedInstrument.COWBELL,
    UnifiedInstrument.SIREN,
    UnifiedInstrument.ORGAN,
    UnifiedInstrument.BRASS_INSTRUMENTS,
    UnifiedInstrument.SHAKER,
    UnifiedInstrument.CONGAS,
    UnifiedInstrument.ORCHESTRA,
    UnifiedInstrument.ELECTRONIC_DRUM_KIT,
    UnifiedInstrument.ELECTRIC_BASS,
    UnifiedInstrument.SHEHNAI,
    UnifiedInstrument.TRUMPET,
    UnifiedInstrument.CYMBALS,
    UnifiedInstrument.SYNTH_STRINGS,
    UnifiedInstrument.FLUTE,
    UnifiedInstrument.SCRATCH,
    UnifiedInstrument.DIGITAL_PIANO,
    UnifiedInstrument.CLAVINET,
    UnifiedInstrument.AGOGO,
    UnifiedInstrument.THEREMIN,
    UnifiedInstrument.TAMBORA,
    UnifiedInstrument.TAMBOURA,
    UnifiedInstrument.GLOCKENSPIEL,
    UnifiedInstrument.NOISE_EFFECTS,
    UnifiedInstrument.ORGAN_AND_ACCORDION,
    UnifiedInstrument.HARMONIUM,
    UnifiedInstrument.ELECTRIC_PIANO,
    UnifiedInstrument.ACOUSTIC_GUITAR,
    UnifiedInstrument.VIOLIN,
    UnifiedInstrument.ORCHESTRA_HIT,
    UnifiedInstrument.RHODES,
    UnifiedInstrument.WIND,
    UnifiedInstrument.ALTO_SAXOPHONE,
    UnifiedInstrument.KAZOO,
    UnifiedInstrument.HAND_CLAP,
    UnifiedInstrument.CAJON,
    UnifiedInstrument.NONE,
    UnifiedInstrument.LUTE,
    UnifiedInstrument.SITAR,
    UnifiedInstrument.EXPLOSION,
    UnifiedInstrument.GUITAR_SYNTH,
    UnifiedInstrument.ETHNIC_PERCUSSION,
    UnifiedInstrument.MUTED_TRUMPET,
    UnifiedInstrument.ARPEGIATOR,
    UnifiedInstrument.ROTOTOM,
    UnifiedInstrument.FX,
    UnifiedInstrument.TRIANGLE,
    UnifiedInstrument.FRETLESS_BASS,
    UnifiedInstrument.THUMB_PIANO,
    UnifiedInstrument.ARR_ELECTRIC_GUITAR,
    UnifiedInstrument.HI_HAT,
    UnifiedInstrument.BOUZOUKI,
    UnifiedInstrument.APPALACHIAN_DULCIMER,
    UnifiedInstrument.PICCOLO,
    UnifiedInstrument.SAROD,
    UnifiedInstrument.BASS_DRUM,
    UnifiedInstrument.DRUM_KIT,
    UnifiedInstrument.MIXED_PERCUSSION,
    UnifiedInstrument.WASHBOARD_GUIRO,
    UnifiedInstrument.CABASA,
    UnifiedInstrument.PERCUSSION,
    UnifiedInstrument.ELECTRONIC_BASS_DRUM,
    UnifiedInstrument.FEMALE_BACKING_VOCALS,
    UnifiedInstrument.SYNTH_AMBIANT,
    UnifiedInstrument.MELLOTRON,
    UnifiedInstrument.JEWS_HARP,
    UnifiedInstrument.CROWD,
    UnifiedInstrument.JINGLE_BELLS,
    UnifiedInstrument.TRADITIONAL_STRINGED_INSTRUMENTS,
    UnifiedInstrument.TROMBONE,
    UnifiedInstrument.ELECTRONIC_TOMS,
    UnifiedInstrument.IRISH_FLUTE,
    UnifiedInstrument.ACCORDION,
    UnifiedInstrument.PANDEIRO,
    UnifiedInstrument.BELLS,
    UnifiedInstrument.SONAR,
    UnifiedInstrument.SYNTH_VOICE,
    UnifiedInstrument.DRUMS_AND_PERCUSSION,
    UnifiedInstrument.MOOG,
    UnifiedInstrument.VIBRASLAP,
    UnifiedInstrument.TUBULAR_BELL,
    UnifiedInstrument.CALLIOPE,
    UnifiedInstrument.VOCAL_BASSIST,
    UnifiedInstrument.CLOCK,
    UnifiedInstrument.CELLO,
    UnifiedInstrument.SYNTHESIZER,
    UnifiedInstrument.BIRDS,
    UnifiedInstrument.BARITONE_GUITAR,
    UnifiedInstrument.SYNTH_KEYS,
    UnifiedInstrument.OBOE,
    UnifiedInstrument.UPRIGHT_BASS,
    UnifiedInstrument.SYNTH_LEAD,
    UnifiedInstrument.SHAKUHACHI,
    UnifiedInstrument.FILTER,
    UnifiedInstrument.VOCODER,
    UnifiedInstrument.TAIKO,
    UnifiedInstrument.TAP_DANCE,
    UnifiedInstrument.BANJO,
    UnifiedInstrument.BRASS_SECTION,
    UnifiedInstrument.VIOLA,
    UnifiedInstrument.SUB,
    UnifiedInstrument.DOBRO,
    UnifiedInstrument.HARP,
    UnifiedInstrument.MELODICA,
    UnifiedInstrument.BAGPIPES,
    UnifiedInstrument.OCARINA,
    UnifiedInstrument.BOWL,
    UnifiedInstrument.WOODWINDS,
    UnifiedInstrument.HARPSICHORD,
    UnifiedInstrument.ELECTRONIC_SNARE_DRUM,
    UnifiedInstrument.BODY_PERCUSSION,
    UnifiedInstrument.ELECTRIC_GUITAR,
    UnifiedInstrument.FLUGELHORN,
    UnifiedInstrument.ZITHER,
    UnifiedInstrument.TRADITIONAL_FLUTE,

    # Audio Timbre(artist) V5
    UnifiedTimbre.JieGeWenKaiYiQiang,
    UnifiedTimbre.XiaoXia,
    UnifiedTimbre.YaoShiSan,
    UnifiedTimbre.C_BLOCK,
    UnifiedTimbre.SongDongYe,
    UnifiedTimbre.KKLUV,
    UnifiedTimbre.WanNengQingNianLvDian,
    UnifiedTimbre.TFBOYS,
    UnifiedTimbre.GuoDing,
    UnifiedTimbre.THE_CRANE,
    UnifiedTimbre.HuiChunDan,
    UnifiedTimbre.LvYanLiang,
    UnifiedTimbre.CaoDongMeiYouPaiDui,
    UnifiedTimbre.HuMengZhou,
    UnifiedTimbre.ZhangChu,
    UnifiedTimbre.MaSaiKeYueDui,
    UnifiedTimbre.DingShiGuang,
    UnifiedTimbre.ShuangSheng,
    UnifiedTimbre.HuangLing,
    UnifiedTimbre.HuangXuan,
    UnifiedTimbre.ZhaoLei,
    UnifiedTimbre.AiYiLiang,
    UnifiedTimbre.J_SHEON,
    UnifiedTimbre.GARETH_T,
    UnifiedTimbre.STEP_JAD,
    UnifiedTimbre.FineYueTuan,
    UnifiedTimbre.TangChao,
    UnifiedTimbre.LengJing,
    UnifiedTimbre.NaQiWoFu,
    UnifiedTimbre.QiuDe,
    UnifiedTimbre.LaoLang,
    UnifiedTimbre.ChongSuDiaoXiangDeQuanLi,
    UnifiedTimbre.GuanHaoDe,
    UnifiedTimbre.DouJingTong,
    UnifiedTimbre.LiJiuZhe,
    UnifiedTimbre.ASi,
    UnifiedTimbre.NineM88,
    UnifiedTimbre.LuoRiFeiChe,
    UnifiedTimbre.WangYiTai,
    UnifiedTimbre.PaiKeTe,
    UnifiedTimbre.PuShu,
    UnifiedTimbre.GALI,
    UnifiedTimbre.LiJiaLong,
    UnifiedTimbre.ANTI_GENERAL,
    UnifiedTimbre.GaoWuRen,
    UnifiedTimbre.LiuBoXin,
    UnifiedTimbre.HuangZiTao,
    UnifiedTimbre.TIZZY_T,
    UnifiedTimbre.DuiZhang,
    UnifiedTimbre.HIGHER_BROTHERS,
    UnifiedTimbre.CAPPER,
    UnifiedTimbre.ZhuJingXi,
    UnifiedTimbre.HuaChenYu,
    UnifiedTimbre.JONY_J,
    UnifiedTimbre.DanBao,
    UnifiedTimbre.YuJiaYun,
    UnifiedTimbre.DaBoLang,
    UnifiedTimbre.YangHeSu,
    UnifiedTimbre.AiRe,
    UnifiedTimbre.XieChunHua,
    UnifiedTimbre.WeiLiAn,
    UnifiedTimbre.YangZongWei,
    UnifiedTimbre.HaoMeiMei,
    UnifiedTimbre.XiaZhiYu,
    UnifiedTimbre.ChenHongYu,
    UnifiedTimbre.AR_LiuFuYang,
    UnifiedTimbre.KAFE_HU,
    UnifiedTimbre.XueZhiQian,
    UnifiedTimbre.XinYueTuan,
    UnifiedTimbre.WanNiDa,
    UnifiedTimbre.KEY_L_LiuCong,
    UnifiedTimbre.SuYunYing,
    UnifiedTimbre.KNOWKNOW,
    UnifiedTimbre.ERIC_ZhouXingZhe,
    UnifiedTimbre.XuSong,
    UnifiedTimbre.GAI_ZhouYan,
    UnifiedTimbre.FaLao,
    UnifiedTimbre.SunNan,
    UnifiedTimbre.VAVA_MaoYanQi,
    UnifiedTimbre.LU1,
    UnifiedTimbre.ErShouMeiGui,
    UnifiedTimbre.NINEONE_ZhaoXingYue,
    UnifiedTimbre.TongYangYueDui,
    UnifiedTimbre.MaSiWei,
    UnifiedTimbre.AiChen,
    UnifiedTimbre.ShanYiChun,
    UnifiedTimbre.LinYouJia,
    UnifiedTimbre.ZhangLiangYing,
    UnifiedTimbre.ICE_PAPER,
    UnifiedTimbre.MaoBuYi,
    UnifiedTimbre.SunShengXi,
    UnifiedTimbre.XuWei,
    UnifiedTimbre.LiYuGang,
    UnifiedTimbre.LiRongHao,
    UnifiedTimbre.ChenLi,
    UnifiedTimbre.XuJiaYing,
    UnifiedTimbre.WeiRuXuan,
    UnifiedTimbre.WangFeng,
    UnifiedTimbre.XinKuZi,
    UnifiedTimbre.TaoZhe,
    UnifiedTimbre.FeiErYueDui,
    UnifiedTimbre.HuaZhou,
    UnifiedTimbre.ChenQiZhen,
    UnifiedTimbre.CaoGe,
    UnifiedTimbre.WuTiaoRen,
    UnifiedTimbre.CiWei_HEDGEHOG,
    UnifiedTimbre.ZhangJie,
    UnifiedTimbre.FangDongDeMao,
    UnifiedTimbre.ICE_YangChangQing,
    UnifiedTimbre.CaiJianYa,
    UnifiedTimbre.ZhangShaoHan,
    UnifiedTimbre.MC_HOTDOG,
    UnifiedTimbre.YinLin,
    UnifiedTimbre.XiaoJingTeng,
    UnifiedTimbre.LiJian,
    UnifiedTimbre.SHE,
    UnifiedTimbre.PanWeiBo,
    UnifiedTimbre.YuanYaWei,
    UnifiedTimbre.GEM_DengZiQi,
    UnifiedTimbre.ZhouJieLun,
    UnifiedTimbre.LiangJingRu,
    UnifiedTimbre.FangDaTong,
    UnifiedTimbre.SunYanZi,
    UnifiedTimbre.ZhangZhenYue,
    UnifiedTimbre.CaiYiLin,
    UnifiedTimbre.YinQueShiTing,
    UnifiedTimbre.WangSuLong,
    UnifiedTimbre.HeTu,
    UnifiedTimbre.HuYanBin,
    UnifiedTimbre.LinJunJie,
    UnifiedTimbre.ZhangHuiMei,
    UnifiedTimbre.ChenXiaoChun,
    UnifiedTimbre.SuDaLv,
    UnifiedTimbre.WangLiHong,
    UnifiedTimbre.ZhouShen,
    UnifiedTimbre.ZhangXinZhe,
    UnifiedTimbre.LiKeQin,
    UnifiedTimbre.DouWei,
    UnifiedTimbre.ChenYiXun,
    UnifiedTimbre.ChenBaiQiang,
    UnifiedTimbre.WuYueTian,
    UnifiedTimbre.ZhangXueYou,
    UnifiedTimbre.WangFei,
    UnifiedTimbre.LiuDeHua,
    UnifiedTimbre.LinYiLian,
    UnifiedTimbre.DengLiJun,
    UnifiedTimbre.YouZhiYuanShaShou,
]



UNIFIED_VOCAB2ID_V4 = lst_to_unified_vocab2id(_UNIFIED_VOCAB2ID_V4_LST)


# =========================================================
# Mappings from strings to unified tags
# The order does not matter
# =========================================================


class ZhVocabError(Exception):
    pass


class Vocab2Id:
    def __init__(
        self,
        category_map: Dict[UnifiedCategory, Dict[str, UnifiedVocab]],
        uni_vocab2id: Dict[UnifiedVocab, int],
        dumpster_map: Optional[Dict[str, UnifiedDumpster]] = None,
    ):
        self.category_map = category_map
        self.uni_vocab2id = uni_vocab2id
        self.dumpster_map = {"None": UnifiedDumpster.NONE} if dumpster_map is None else dumpster_map
        self._validate()

    def map(self, category: UnifiedCategory, tag: str) -> int:
        return self.uni_vocab2id[self.category_map[category][tag]]

    def to_dict(self) -> Dict[str, int]:
        return {
            **{
                tag_name: self.uni_vocab2id[uni_tag]
                for tag_name, uni_tag in self.dumpster_map.items()
            },
            **{
                tag_name: self.map(cat_name, tag_name) 
                for cat_name, cat_map in self.category_map.items() 
                for tag_name in cat_map
            },
        }

    @property
    def num_categories(self) -> int:
        return len(self.category_map)

    @property
    def categories(self) -> List[UnifiedCategory]:
        return list(self.category_map.keys())

    def _validate(self):
        all_tag_names = [tag_name for cat_map in self.category_map.values() for tag_name in cat_map]
        if len(all_tag_names) != len(set(all_tag_names)):
            dups = [tag_name for tag_name in all_tag_names if all_tag_names.count(tag_name) > 1]
            raise ZhVocabError(f"Duplicated tag name(s): {set(dups)}")
        for uni_tag in self.dumpster_map.values():
            if uni_tag not in self.uni_vocab2id:
                raise ZhVocabError(f"{uni_tag} is not in uni_vocab2id")


# =========================================================

CATEGORY_MAP_SA = {
    UnifiedCategory.GENRE: {
        "Blues": UnifiedGenre.BLUES,
        "Chinese Opera": UnifiedGenre.CHINESE_OPERA,
        "Chinese Style": UnifiedGenre.CHINESE_STYLE,
        "Chinese Tradition": UnifiedGenre.CHINESE_TRADITION,
        "Classical": UnifiedGenre.CLASSICAL,
        "Country": UnifiedGenre.COUNTRY_POP,
        "DJ": UnifiedGenre.DJ,
        "Easy Listening": UnifiedGenre.EASY_LISTENING,
        "Electronic": UnifiedGenre.ELECTRONIC,
        "Folk": UnifiedGenre.FOLK,
        "Hip Hop/Rap": UnifiedGenre.HIP_HOP,
        "Jazz": UnifiedGenre.JAZZ,
        "Latin": UnifiedGenre.LATIN,
        "MC": UnifiedGenre.MC,
        "Metal": UnifiedGenre.METAL,
        "Other genre": UnifiedGenre.EMPTY,
        "Pop": UnifiedGenre.POP,
        "Punk": UnifiedGenre.PUNK,
        "R&B/Soul": UnifiedGenre.RNB_SOUL,
        "Reggae": UnifiedGenre.REGGAE,
        "Rock": UnifiedGenre.ROCK,
    },
    UnifiedCategory.MOOD: {
        "Angry": UnifiedMood.ANGRY_AGGRESSIVE,
        "Calm": UnifiedMood.CALM_RELAXING,
        "Chill": UnifiedMood.CHILL,
        "Cute_SA_MOOD": UnifiedMood.CUTE_PLAYFUL,  # dedup
        "Dynamic": UnifiedMood.DYNAMIC_ENERGETIC,
        "Excited": UnifiedMood.EXCITED,
        "Funny": UnifiedMood.FUNNY,
        "Happy": UnifiedMood.HAPPY,
        "Healing": UnifiedMood.HEALING,
        "Inspirational": UnifiedMood.INSPIRATIONAL_HOPEFUL,
        "Lonely": UnifiedMood.SENTIMENTAL_MELANCHOLIC_LONELY,
        "Miss/Memory": UnifiedMood.MISS,
        "Mysterious": UnifiedMood.MYSTERIOUS,
        # The name "Other" is also used in theme, according to the legacy lookup
        # table, they share the same index.
        # It is currently fixed. But it also depends on the parser to remap the tag name.
        "Other_SA_MOOD": UnifiedMood.EMPTY,  # dedup (versions <= v3_pretrain_alpha never use this tag)
        "Romantic": UnifiedMood.ROMANTIC,
        "Sorrow": UnifiedMood.SORROW_SAD,
        "Sweet_SA_MOOD": UnifiedMood.SWEET,  # dedup
        "Tense": UnifiedMood.THRILLING_SUSPENSEFUL_TENSE,
        "Weird": UnifiedMood.WEIRD,
    },
    UnifiedCategory.THEME: {
        "Autumn": UnifiedTheme.AUTUMN,
        "Bedtime": UnifiedTheme.BED_TIME,
        "Birthday": UnifiedTheme.BIRTHDAY,
        "Broke up": UnifiedTheme.BROKE_UP,
        "Cafe": UnifiedTheme.COFFEE_SHOP,
        "Campus": UnifiedTheme.CAMPUS,
        "Christmas": UnifiedTheme.CHRISTMAS,
        "Dance": UnifiedTheme.DANCE,
        "Danceable": UnifiedTheme.DANCEABLE,
        "Date": UnifiedTheme.DATE,
        "Dream": UnifiedTheme.DREAM,
        "Drive": UnifiedTheme.DRIVE,
        "Evening": UnifiedTheme.EVENING,
        "Family": UnifiedTheme.FAMILY_TIME,
        "Focus": UnifiedTheme.FOCUS,
        "Food": UnifiedTheme.FOOD,
        "Friendship": UnifiedTheme.FRIENDSHIP,
        "Game": UnifiedTheme.GAME,
        "Halloween": UnifiedTheme.HALLOWEEN,
        "Love": UnifiedTheme.LOVE,
        "Meditation": UnifiedTheme.MEDITATION,
        "Morning": UnifiedTheme.MORNING,
        "Other": UnifiedTheme.EMPTY,  # no renaming it to make it compatible with the old format
        "Party": UnifiedTheme.PARTY,
        "Sport": UnifiedTheme.SPORT,
        "Spring": UnifiedTheme.SPRING,
        "Spring Festival": UnifiedTheme.SPRING_FESTIVAL,
        "Summer": UnifiedTheme.SUMMER,
        "Travel": UnifiedTheme.TRAVEL,
        "Valentine's Day": UnifiedTheme.VALENTINES_DAY,
        "Wedding": UnifiedTheme.WEDDING,
        "Winter": UnifiedTheme.WINTER,
        "Yoga": UnifiedTheme.YOGA,
    },
    UnifiedCategory.LANG: {
        "Cantonese": UnifiedLang.CANTONESE,
        "Chinese": UnifiedLang.CHINESE,
        "Chinese Dialects": UnifiedLang.CHINESE_DIALECT,
        "English": UnifiedLang.ENGLISH,
    },
    UnifiedCategory.SINKING: {
        "Sinking": UnifiedSinking.SINKING,
        "non-Sinking": UnifiedSinking.NON_SINKING,
    },
}

VOCAB2ID_SA = Vocab2Id(
    CATEGORY_MAP_SA,
    UNIFIED_VOCAB2ID_LEGACY,
)

# =========================================================

CATEGORY_MAP_AUDIO_V0 = {
    UnifiedCategory.GENRE: {
        "Pop": UnifiedGenre.POP,
        "Electronic": UnifiedGenre.ELECTRONIC,
        "Chinese Style": UnifiedGenre.CHINESE_STYLE,
        "Rock": UnifiedGenre.ROCK,
        "Jazz": UnifiedGenre.JAZZ,
        "Hip Hop": UnifiedGenre.HIP_HOP,
        "Classical": UnifiedGenre.CLASSICAL,
        "R&B/Soul": UnifiedGenre.RNB_SOUL,
        "Metal": UnifiedGenre.METAL,
        "Tuhai": UnifiedGenre.TUHAI,
        "Childhood": UnifiedGenre.CHILDHOOD,
        "Devotional": UnifiedGenre.DEVOTIONAL,
        "Chinese Tradition": UnifiedGenre.CHINESE_TRADITION,
        "Easy Listening": UnifiedGenre.EASY_LISTENING,
        "New Age": UnifiedGenre.NEW_AGE,
        "8 Bit": UnifiedGenre.EIGHT_BIT,
        "Folk": UnifiedGenre.FOLK,
        "Latin": UnifiedGenre.LATIN,
        "Alternative/Indie": UnifiedGenre.ALTERNATIVE_INDIE,
        "Epic": UnifiedGenre.EPIC,
        "BGM": UnifiedGenre.BGM,
        "Chinese Pop": UnifiedGenre.CHINESE_POP,
        "Cantopop": UnifiedGenre.CANTOPOP,
        "Taiwanese Pop": UnifiedGenre.TAIWANESE_POP,
        "Pop Folk": UnifiedGenre.POP_FOLK,
        "Contemporary Pop": UnifiedGenre.CONTEMPORARY_POP,
        "Teen Pop": UnifiedGenre.TEEN_POP,
        "Indie Pop": UnifiedGenre.INDIE_POP,
        "Dream Pop": UnifiedGenre.DREAM_POP,
        "City Pop": UnifiedGenre.CITY_POP,
        "Synth Pop": UnifiedGenre.SYNTH_POP,
        "Celtic Pop": UnifiedGenre.CELTIC_POP,
        "Dance Pop": UnifiedGenre.DANCE_POP,
        "Deep Dance Pop": UnifiedGenre.DEEP_DANCE_POP,
        "Electropop": UnifiedGenre.ELECTROPOP,
        "Chamber Pop": UnifiedGenre.CHAMBER_POP,
        "A cappella": UnifiedGenre.A_CAPPELLA,
        "Country Pop": UnifiedGenre.COUNTRY_POP,
        "EDM": UnifiedGenre.EDM,
        "House": UnifiedGenre.HOUSE,
        "Dubstep": UnifiedGenre.DUBSTEP,
        "Future Bass": UnifiedGenre.FUTURE_BASS,
        "Chillout": UnifiedGenre.CHILLOUT,
        "Trance": UnifiedGenre.TRANCE,
        "Techno": UnifiedGenre.TECHNO,
        "Drum&Bass": UnifiedGenre.DRUMNBASS,
        "Tropical House": UnifiedGenre.TROPICAL_HOUSE,
        "Disco": UnifiedGenre.DISCO,
        "Vaporwave": UnifiedGenre.VAPORWAVE,
        "Trip Hop": UnifiedGenre.TRIP_HOP,
        "Ambient": UnifiedGenre.AMBIENT,
        "Deep Pop Edm": UnifiedGenre.DEEP_POP_EDM,
        "EDM Trap": UnifiedGenre.EDM_TRAP,
        "Future House": UnifiedGenre.FUTURE_HOUSE,
        "China-Wave": UnifiedGenre.CHINA_WAVE,
        "GuFeng Music": UnifiedGenre.GUFENG_MUSIC,
        "Chinoiserie Rap": UnifiedGenre.CHINOISERIE_RAP,
        "Chinoiserie Electronic": UnifiedGenre.CHINOISERIE_ELECTRONIC,
        "Hard Rock": UnifiedGenre.HARD_ROCK,
        "Psychedelic Rock": UnifiedGenre.PSYCHEDELIC_ROCK,
        "Pop Rock": UnifiedGenre.POP_ROCK,
        "Instrumental Rock": UnifiedGenre.INSTRUMENTAL_ROCK,
        "Alternative Rock": UnifiedGenre.ALTERNATIVE_ROCK,
        "Indie Rock": UnifiedGenre.INDIE_ROCK,
        "Post-Rock": UnifiedGenre.POST_ROCK,
        "Lo-Fi": UnifiedGenre.LO_FI,
        "J Rock": UnifiedGenre.J_ROCK,
        "Shoegazing": UnifiedGenre.SHOEGAZING,
        "Math Rock": UnifiedGenre.MATH_ROCK,
        "Surf Rock": UnifiedGenre.SURF_ROCK,
        "Progressive Rock": UnifiedGenre.PROGRESSIVE_ROCK,
        "Soft Rock": UnifiedGenre.SOFT_ROCK,
        "Jazz Pop": UnifiedGenre.JAZZ_POP,
        "Jazz Fusion": UnifiedGenre.JAZZ_FUSION,
        "Bossa Nova": UnifiedGenre.BOSSA_NOVA,
        "Avant-Garde Jazz": UnifiedGenre.AVANT_GARDE_JAZZ,
        "Swing": UnifiedGenre.SWING,
        "Big Band": UnifiedGenre.BIG_BAND,
        "Bop": UnifiedGenre.BOP,
        "Post-Bop": UnifiedGenre.POST_BOP,
        "Smooth Jazz": UnifiedGenre.SMOOTH_JAZZ,
        "Cool Jazz": UnifiedGenre.COOL_JAZZ,
        "Vocal Jazz": UnifiedGenre.VOCAL_JAZZ,
        "Nu Jazz": UnifiedGenre.NU_JAZZ,
        "Free Jazz": UnifiedGenre.FREE_JAZZ,
        "Trap Rap": UnifiedGenre.TRAP_RAP,
        "Old School": UnifiedGenre.OLD_SCHOOL,
        "R&B Rap": UnifiedGenre.RNB_RAP,
        "Jazz Hip Hop": UnifiedGenre.JAZZ_HIP_HOP,
        "Alternative Hip Hop": UnifiedGenre.ALTERNATIVE_HIP_HOP,
        "Instrumental Hip Hop": UnifiedGenre.INSTRUMENTAL_HIP_HOP,
        "Pop Rap": UnifiedGenre.POP_RAP,
        "Hardcore Rap": UnifiedGenre.HARDCORE_RAP,
        "Comedy Hip Hop": UnifiedGenre.COMEDY_HIP_HOP,
        "Hip House": UnifiedGenre.HIP_HOUSE,
        "Chill Beats": UnifiedGenre.CHILL_BEATS,
        "Chorus_AUDIO_GENRE": UnifiedGenre.CHORUS,  # dedup
        "Chamber Music": UnifiedGenre.CHAMBER_MUSIC,
        "Symphony": UnifiedGenre.SYMPHONY,
        "Funk": UnifiedGenre.FUNK,
        "Contemporary R&B": UnifiedGenre.CONTEMPORARY_RNB,
        "Neo Soul": UnifiedGenre.NEO_SOUL,
        "Soul": UnifiedGenre.SOUL,
        "Pop Soul": UnifiedGenre.POP_SOUL,
        "Black Metal": UnifiedGenre.BLACK_METAL,
        "Death Metal": UnifiedGenre.DEATH_METAL,
        "Glam Metal": UnifiedGenre.GLAM_METAL,
        "Grindcore": UnifiedGenre.GRINDCORE,
        "Power Metal": UnifiedGenre.POWER_METAL,
        "Progressive Metal": UnifiedGenre.PROGRESSIVE_METAL,
        "Speed Metal": UnifiedGenre.SPEED_METAL,
        "DJ": UnifiedGenre.DJ,
        "VinaHouse": UnifiedGenre.VINAHOUSE,
        "MC": UnifiedGenre.MC,
        "Gospel": UnifiedGenre.GOSPEL,
        "Holiday Music": UnifiedGenre.HOLIDAY_MUSIC,
        "Christian Music": UnifiedGenre.CHRISTIAN_MUSIC,
        "Buddhist music": UnifiedGenre.BUDDHIST_MUSIC,
        "Chinese Opera": UnifiedGenre.CHINESE_OPERA,
        "Traditional Chinese Folk": UnifiedGenre.TRADITIONAL_CHINESE_FOLK,
        "Chinese Quyi": UnifiedGenre.CHINESE_QUYI,
        "Red Song": UnifiedGenre.RED_SONG,
        "Folk Pop": UnifiedGenre.FOLK_POP,
        "Indie Folk": UnifiedGenre.INDIE_FOLK,
        "Tango": UnifiedGenre.TANGO,
        "Reggaeton": UnifiedGenre.REGGAETON,
        "Punk": UnifiedGenre.PUNK,
        "Reggae": UnifiedGenre.REGGAE,
    },
    UnifiedCategory.MOOD: {
        "Happy": UnifiedMood.HAPPY,
        "Cute/Playful": UnifiedMood.CUTE_PLAYFUL,
        "Excited": UnifiedMood.EXCITED,
        "Funny": UnifiedMood.FUNNY,
        "Inspirational/Hopeful": UnifiedMood.INSPIRATIONAL_HOPEFUL,
        "Sorrow/Sad": UnifiedMood.SORROW_SAD,
        "Sentimental/Melancholic/Lonely": UnifiedMood.SENTIMENTAL_MELANCHOLIC_LONELY,
        "Weird": UnifiedMood.WEIRD,
        "Thrilling/Suspenseful/Tense": UnifiedMood.THRILLING_SUSPENSEFUL_TENSE,
        "Angry/Aggressive": UnifiedMood.ANGRY_AGGRESSIVE,
        "Groovy/Funky": UnifiedMood.GROOVY_FUNKY,
        "Dynamic/Energetic": UnifiedMood.DYNAMIC_ENERGETIC,
        "Romantic": UnifiedMood.ROMANTIC,
        "Nostalgic/Memory": UnifiedMood.NOSTALGIC_MEMORY,
        "Dreamy/Ethereal": UnifiedMood.DREAMY_ETHEREAL,
        "Healing": UnifiedMood.HEALING,
        "Miss": UnifiedMood.MISS,
        "Chill": UnifiedMood.CHILL,
        "Calm/Relaxing": UnifiedMood.CALM_RELAXING,
        "Shocking/magnificent/epic": UnifiedMood.SHOCKING_MAGNIFICENT_EPIC,
        "Mysterious": UnifiedMood.MYSTERIOUS,
        "No Mood": UnifiedMood.EMPTY,
    },
    UnifiedCategory.THEME: {
        "Halloween": UnifiedTheme.HALLOWEEN,
        "Christmas": UnifiedTheme.CHRISTMAS,
        "New Year": UnifiedTheme.NEW_YEAR,
        "Spring Festival": UnifiedTheme.SPRING_FESTIVAL,
        "Valentine's day": UnifiedTheme.VALENTINES_DAY,
        "Qi Xi": UnifiedTheme.QI_XI,
        "Birthday": UnifiedTheme.BIRTHDAY,
        "Wedding": UnifiedTheme.WEDDING,
        "Funeral": UnifiedTheme.FUNERAL,
        "Graduation": UnifiedTheme.GRADUATION,
        "National's Day": UnifiedTheme.NATIONALS_DAY,
        "Vlog/DailyLife": UnifiedTheme.VLOG_DAILYLIFE,
        "Food": UnifiedTheme.FOOD,
        "Pet/Animals": UnifiedTheme.PET_ANIMALS,
        "Beauty/Fashion": UnifiedTheme.BEAUTY_FASHION,
        "Entertainment": UnifiedTheme.ENTERTAINMENT,
        "babies": UnifiedTheme.BABIES,
        "children": UnifiedTheme.CHILDREN,
        "Transition": UnifiedTheme.TRANSITION,
        "Anime": UnifiedTheme.ANIME,
        "Wake up": UnifiedTheme.WAKE_UP,
        "Family time": UnifiedTheme.FAMILY_TIME,
        "landscape/scenery": UnifiedTheme.LANDSCAPE_SCENERY,
        "Prank": UnifiedTheme.PRANK,
        "Timelapse": UnifiedTheme.TIMELAPSE,
        "Rainy Day": UnifiedTheme.RAINY_DAY,
        "Sunny Day": UnifiedTheme.SUNNY_DAY,
        "Spring": UnifiedTheme.SPRING,
        "Summer": UnifiedTheme.SUMMER,
        "Autumn": UnifiedTheme.AUTUMN,
        "Winter": UnifiedTheme.WINTER,
        "Evening": UnifiedTheme.EVENING,
        "Morning": UnifiedTheme.MORNING,
        "Beach": UnifiedTheme.BEACH,
        "Nightclub": UnifiedTheme.NIGHTCLUB,
        "Coffee Shop": UnifiedTheme.COFFEE_SHOP,
        "Restaurants": UnifiedTheme.RESTAURANTS,
        "Lounge": UnifiedTheme.LOUNGE,
        "Campus": UnifiedTheme.CAMPUS,
        "Park": UnifiedTheme.PARK,
        "Marketplace": UnifiedTheme.MARKETPLACE,
        "Universe": UnifiedTheme.UNIVERSE,
        "Bar": UnifiedTheme.BAR,
        "Theater/Concert hall": UnifiedTheme.THEATER_CONCERT_HALL,
        "Sport": UnifiedTheme.SPORT,
        "Dance": UnifiedTheme.DANCE,
        "Game": UnifiedTheme.GAME,
        "Travel": UnifiedTheme.TRAVEL,
        "Focus": UnifiedTheme.FOCUS,
        "Party": UnifiedTheme.PARTY,
        "Commute": UnifiedTheme.COMMUTE,
        "Roadtrip": UnifiedTheme.ROADTRIP,
        "Meditation": UnifiedTheme.MEDITATION,
        "Relaxation": UnifiedTheme.RELAXATION,
        "Cleaning and Chores": UnifiedTheme.CLEANING_AND_CHORES,
        "Running": UnifiedTheme.RUNNING,
        "Date": UnifiedTheme.DATE,
        "Danceable": UnifiedTheme.DANCEABLE,
        "Flirt": UnifiedTheme.FLIRT,
        "Other": UnifiedTheme.EMPTY,
    },
    UnifiedCategory.GENDER: {
        "Female": UnifiedGender.FEMALE,
        "Male": UnifiedGender.MALE,
        "Neutral": UnifiedGender.NEUTRAL,
        "Child": UnifiedGender.CHILD,
        "Adult": UnifiedGender.ADULT,
        "Chorus_AUDIO_GENDER": UnifiedGender.CHORUS,  # dedup
    },
    UnifiedCategory.TIMBRE: {
        "Warm": UnifiedTimbre.WARM,
        "Ethereal": UnifiedTimbre.ETHEREAL,
        "Husky": UnifiedTimbre.HUSKY,
        "Deep": UnifiedTimbre.DEEP,
        "Loud and sonorous": UnifiedTimbre.LOUD_AND_SONOROUS,
        "Extreme": UnifiedTimbre.EXTREME,
        "Sharp": UnifiedTimbre.SHARP,
        "Bright": UnifiedTimbre.BRIGHT,
        "Sweet_AUDIO_TIMBRE": UnifiedTimbre.SWEET,  # dedup
        "Powerful": UnifiedTimbre.POWERFUL,
        "Sexy/Lazy": UnifiedTimbre.SEXY_LAZY,
        "Magnetic": UnifiedTimbre.MAGNETIC,
        "Cute_AUDIO_TIMBRE": UnifiedTimbre.CUTE,  # dedup
        "Electrified voice": UnifiedTimbre.ELECTRIFIED_VOICE,
    },
}

VOCAB2ID_AUDIO_V0 = Vocab2Id(
    CATEGORY_MAP_AUDIO_V0,
    UNIFIED_VOCAB2ID_LEGACY,
)

# =========================================================

CATEGORY_MAP_AUDIO_V1 = {
    UnifiedCategory.GENRE: {
       "Pop": UnifiedGenre.POP,
       "Electronic": UnifiedGenre.ELECTRONIC,
       "Chinese Style": UnifiedGenre.CHINESE_STYLE,
       "Rock": UnifiedGenre.ROCK,
       "Jazz": UnifiedGenre.JAZZ,
       "Hip Hop/Rap": UnifiedGenre.HIP_HOP,
       "Classical": UnifiedGenre.CLASSICAL,
       "R&B/Soul": UnifiedGenre.RNB_SOUL,
       "Metal": UnifiedGenre.METAL,
       "Tuhai": UnifiedGenre.TUHAI,
       "Childhood": UnifiedGenre.CHILDHOOD,
       "Devotional": UnifiedGenre.DEVOTIONAL,
       "Chinese Tradition": UnifiedGenre.CHINESE_TRADITION,
       "Easy Listening": UnifiedGenre.EASY_LISTENING,
       "New Age": UnifiedGenre.NEW_AGE,
       "8 Bit": UnifiedGenre.EIGHT_BIT,
       "Folk": UnifiedGenre.FOLK,
       "Latin": UnifiedGenre.LATIN,
       "Alternative/Indie": UnifiedGenre.ALTERNATIVE_INDIE,
       "Epic": UnifiedGenre.EPIC,
       "BGM": UnifiedGenre.BGM,
       "Chinese Pop": UnifiedGenre.CHINESE_POP,
       "Cantopop": UnifiedGenre.CANTOPOP,
       "Taiwanese Pop": UnifiedGenre.TAIWANESE_POP,
       "Pop Folk": UnifiedGenre.POP_FOLK,
       "Contemporary Pop": UnifiedGenre.CONTEMPORARY_POP,
       "Teen Pop": UnifiedGenre.TEEN_POP,
       "Indie Pop": UnifiedGenre.INDIE_POP,
       "Dream Pop": UnifiedGenre.DREAM_POP,
       "City Pop": UnifiedGenre.CITY_POP,
       "Synth Pop": UnifiedGenre.SYNTH_POP,
       "Celtic Pop": UnifiedGenre.CELTIC_POP,
       "Dance Pop": UnifiedGenre.DANCE_POP,
       "Deep Dance Pop": UnifiedGenre.DEEP_DANCE_POP,
       "Electropop": UnifiedGenre.ELECTROPOP,
       "Chamber Pop": UnifiedGenre.CHAMBER_POP,
       "A cappella": UnifiedGenre.A_CAPPELLA,
       "Country Pop": UnifiedGenre.COUNTRY_POP,
       "EDM": UnifiedGenre.EDM,
       "House": UnifiedGenre.HOUSE,
       "Dubstep": UnifiedGenre.DUBSTEP,
       "Future Bass": UnifiedGenre.FUTURE_BASS,
       "Chillout": UnifiedGenre.CHILLOUT,
       "Trance": UnifiedGenre.TRANCE,
       "Techno": UnifiedGenre.TECHNO,
       "Drum&Bass": UnifiedGenre.DRUMNBASS,
       "Tropical House": UnifiedGenre.TROPICAL_HOUSE,
       "Disco": UnifiedGenre.DISCO,
       "Vaporwave": UnifiedGenre.VAPORWAVE,
       "Trip Hop": UnifiedGenre.TRIP_HOP,
       "Ambient": UnifiedGenre.AMBIENT,
       "Deep Pop Edm": UnifiedGenre.DEEP_POP_EDM,
       "EDM Trap": UnifiedGenre.EDM_TRAP,
       "Future House": UnifiedGenre.FUTURE_HOUSE,
       "China-Wave": UnifiedGenre.CHINA_WAVE,
       "GuFeng Music": UnifiedGenre.GUFENG_MUSIC,
       "Chinoiserie Rap": UnifiedGenre.CHINOISERIE_RAP,
       "Chinoiserie Electronic": UnifiedGenre.CHINOISERIE_ELECTRONIC,
       "Hard Rock": UnifiedGenre.HARD_ROCK,
       "Psychedelic Rock": UnifiedGenre.PSYCHEDELIC_ROCK,
       "Pop Rock": UnifiedGenre.POP_ROCK,
       "Instrumental Rock": UnifiedGenre.INSTRUMENTAL_ROCK,
       "Alternative Rock": UnifiedGenre.ALTERNATIVE_ROCK,
       "Indie Rock": UnifiedGenre.INDIE_ROCK,
       "Post-Rock": UnifiedGenre.POST_ROCK,
       "Lo-Fi": UnifiedGenre.LO_FI,
       "J Rock": UnifiedGenre.J_ROCK,
       "Shoegazing": UnifiedGenre.SHOEGAZING,
       "Math Rock": UnifiedGenre.MATH_ROCK,
       "Surf Rock": UnifiedGenre.SURF_ROCK,
       "Progressive Rock": UnifiedGenre.PROGRESSIVE_ROCK,
       "Soft Rock": UnifiedGenre.SOFT_ROCK,
       "Jazz Pop": UnifiedGenre.JAZZ_POP,
       "Jazz Fusion": UnifiedGenre.JAZZ_FUSION,
       "Bossa Nova": UnifiedGenre.BOSSA_NOVA,
       "Avant-Garde Jazz": UnifiedGenre.AVANT_GARDE_JAZZ,
       "Swing": UnifiedGenre.SWING,
       "Big Band": UnifiedGenre.BIG_BAND,
       "Bop": UnifiedGenre.BOP,
       "Post-Bop": UnifiedGenre.POST_BOP,
       "Smooth Jazz": UnifiedGenre.SMOOTH_JAZZ,
       "Cool Jazz": UnifiedGenre.COOL_JAZZ,
       "Vocal Jazz": UnifiedGenre.VOCAL_JAZZ,
       "Nu Jazz": UnifiedGenre.NU_JAZZ,
       "Free Jazz": UnifiedGenre.FREE_JAZZ,
       "Trap Rap": UnifiedGenre.TRAP_RAP,
       "Old School": UnifiedGenre.OLD_SCHOOL,
       "R&B Rap": UnifiedGenre.RNB_RAP,
       "Jazz Hip Hop": UnifiedGenre.JAZZ_HIP_HOP,
       "Alternative Hip Hop": UnifiedGenre.ALTERNATIVE_HIP_HOP,
       "Instrumental Hip Hop": UnifiedGenre.INSTRUMENTAL_HIP_HOP,
       "Pop Rap": UnifiedGenre.POP_RAP,
       "Hardcore Rap": UnifiedGenre.HARDCORE_RAP,
       "Comedy Hip Hop": UnifiedGenre.COMEDY_HIP_HOP,
       "Hip House": UnifiedGenre.HIP_HOUSE,
       "Chill Beats": UnifiedGenre.CHILL_BEATS,
       "Chorus_AUDIO_GENRE": UnifiedGenre.CHORUS,
       "Chamber Music": UnifiedGenre.CHAMBER_MUSIC,
       "Symphony": UnifiedGenre.SYMPHONY,
       "Funk": UnifiedGenre.FUNK,
       "Contemporary R&B": UnifiedGenre.CONTEMPORARY_RNB,
       "Neo Soul": UnifiedGenre.NEO_SOUL,
       "Soul": UnifiedGenre.SOUL,
       "Pop Soul": UnifiedGenre.POP_SOUL,
       "Black Metal": UnifiedGenre.BLACK_METAL,
       "Death Metal": UnifiedGenre.DEATH_METAL,
       "Glam Metal": UnifiedGenre.GLAM_METAL,
       "Grindcore": UnifiedGenre.GRINDCORE,
       "Power Metal": UnifiedGenre.POWER_METAL,
       "Progressive Metal": UnifiedGenre.PROGRESSIVE_METAL,
       "Speed Metal": UnifiedGenre.SPEED_METAL,
       "DJ": UnifiedGenre.DJ,
       "VinaHouse": UnifiedGenre.VINAHOUSE,
       "MC": UnifiedGenre.MC,
       "Gospel": UnifiedGenre.GOSPEL,
       "Holiday Music": UnifiedGenre.HOLIDAY_MUSIC,
       "Christian Music": UnifiedGenre.CHRISTIAN_MUSIC,
       "Buddhist music": UnifiedGenre.BUDDHIST_MUSIC,
       "Chinese Opera": UnifiedGenre.CHINESE_OPERA,
       "Traditional Chinese Folk": UnifiedGenre.TRADITIONAL_CHINESE_FOLK,
       "Chinese Quyi": UnifiedGenre.CHINESE_QUYI,
       "Red Song": UnifiedGenre.RED_SONG,
       "Folk Pop": UnifiedGenre.FOLK_POP,
       "Indie Folk": UnifiedGenre.INDIE_FOLK,
       "Tango": UnifiedGenre.TANGO,
       "Reggaeton": UnifiedGenre.REGGAETON,
       "Blues": UnifiedGenre.BLUES,
       "Country": UnifiedGenre.COUNTRY_POP,  # share the same unified tag with "Country Pop"
       "Punk": UnifiedGenre.PUNK,
       "Reggae": UnifiedGenre.REGGAE,
       "Other genre": UnifiedGenre.EMPTY,
    },
    UnifiedCategory.MOOD: {
       "Happy": UnifiedMood.HAPPY,
       "Cute/Playful": UnifiedMood.CUTE_PLAYFUL,
       "Excited": UnifiedMood.EXCITED,
       "Funny": UnifiedMood.FUNNY,
       "Inspirational/Hopeful": UnifiedMood.INSPIRATIONAL_HOPEFUL,
       "Sorrow/Sad": UnifiedMood.SORROW_SAD,
       "Sentimental/Melancholic/Lonely": UnifiedMood.SENTIMENTAL_MELANCHOLIC_LONELY,
       "Weird": UnifiedMood.WEIRD,
       "Thrilling/Suspenseful/Tense": UnifiedMood.THRILLING_SUSPENSEFUL_TENSE,
       "Angry/Aggressive": UnifiedMood.ANGRY_AGGRESSIVE,
       "Groovy/Funky": UnifiedMood.GROOVY_FUNKY,
       "Dynamic/Energetic": UnifiedMood.DYNAMIC_ENERGETIC,
       "Romantic": UnifiedMood.ROMANTIC,
       "Nostalgic/Memory": UnifiedMood.NOSTALGIC_MEMORY,
       "Dreamy/Ethereal": UnifiedMood.DREAMY_ETHEREAL,
       "Healing": UnifiedMood.HEALING,
       "Miss": UnifiedMood.MISS,
       "Chill": UnifiedMood.CHILL,
       "Calm/Relaxing": UnifiedMood.CALM_RELAXING,
       "Shocking/magnificent/epic": UnifiedMood.SHOCKING_MAGNIFICENT_EPIC,
       "Mysterious": UnifiedMood.MYSTERIOUS,
       "No Mood": UnifiedMood.EMPTY,
       "Sweet": UnifiedMood.SWEET,
       "Other mood": UnifiedMood.EMPTY,
    },
    UnifiedCategory.THEME: {
       "Halloween": UnifiedTheme.HALLOWEEN,
       "Christmas": UnifiedTheme.CHRISTMAS,
       "New Year": UnifiedTheme.NEW_YEAR,
       "Spring Festival": UnifiedTheme.SPRING_FESTIVAL,
       "Valentine's day": UnifiedTheme.VALENTINES_DAY,
       "Qi Xi": UnifiedTheme.QI_XI,
       "Birthday": UnifiedTheme.BIRTHDAY,
       "Wedding": UnifiedTheme.WEDDING,
       "Funeral": UnifiedTheme.FUNERAL,
       "Graduation": UnifiedTheme.GRADUATION,
       "National's Day": UnifiedTheme.NATIONALS_DAY,
       "Vlog/DailyLife": UnifiedTheme.VLOG_DAILYLIFE,
       "Food": UnifiedTheme.FOOD,
       "Pet/Animals": UnifiedTheme.PET_ANIMALS,
       "Beauty/Fashion": UnifiedTheme.BEAUTY_FASHION,
       "Entertainment": UnifiedTheme.ENTERTAINMENT,
       "babies": UnifiedTheme.BABIES,
       "children": UnifiedTheme.CHILDREN,
       "Transition": UnifiedTheme.TRANSITION,
       "Anime": UnifiedTheme.ANIME,
       "Wake up": UnifiedTheme.WAKE_UP,
       "Family time": UnifiedTheme.FAMILY_TIME,
       "landscape/scenery": UnifiedTheme.LANDSCAPE_SCENERY,
       "Prank": UnifiedTheme.PRANK,
       "Timelapse": UnifiedTheme.TIMELAPSE,
       "Rainy Day": UnifiedTheme.RAINY_DAY,
       "Sunny Day": UnifiedTheme.SUNNY_DAY,
       "Spring": UnifiedTheme.SPRING,
       "Summer": UnifiedTheme.SUMMER,
       "Autumn": UnifiedTheme.AUTUMN,
       "Winter": UnifiedTheme.WINTER,
       "Evening": UnifiedTheme.EVENING,
       "Morning": UnifiedTheme.MORNING,
       "Beach": UnifiedTheme.BEACH,
       "Nightclub": UnifiedTheme.NIGHTCLUB,
       "Coffee Shop": UnifiedTheme.COFFEE_SHOP,
       "Restaurants": UnifiedTheme.RESTAURANTS,
       "Lounge": UnifiedTheme.LOUNGE,
       "Campus": UnifiedTheme.CAMPUS,
       "Park": UnifiedTheme.PARK,
       "Marketplace": UnifiedTheme.MARKETPLACE,
       "Universe": UnifiedTheme.UNIVERSE,
       "Bar": UnifiedTheme.BAR,
       "Theater/Concert hall": UnifiedTheme.THEATER_CONCERT_HALL,
       "Sport": UnifiedTheme.SPORT,
       "Dance": UnifiedTheme.DANCE,
       "Game": UnifiedTheme.GAME,
       "Travel": UnifiedTheme.TRAVEL,
       "Focus": UnifiedTheme.FOCUS,
       "Party": UnifiedTheme.PARTY,
       "Commute": UnifiedTheme.COMMUTE,
       "Roadtrip": UnifiedTheme.ROADTRIP,
       "Meditation": UnifiedTheme.MEDITATION,
       "Relaxation": UnifiedTheme.RELAXATION,
       "Cleaning and Chores": UnifiedTheme.CLEANING_AND_CHORES,
       "Running": UnifiedTheme.RUNNING,
       "Date": UnifiedTheme.DATE,
       "Danceable": UnifiedTheme.DANCEABLE,
       "Flirt": UnifiedTheme.FLIRT,
       "Bedtime": UnifiedTheme.BED_TIME,
       "Broke up": UnifiedTheme.BROKE_UP,
       "Dream": UnifiedTheme.DREAM,
       "Drive": UnifiedTheme.DRIVE,
       "Friendship": UnifiedTheme.FRIENDSHIP,
       "Love": UnifiedTheme.LOVE,
       "Yoga": UnifiedTheme.YOGA,
       "Other scene": UnifiedTheme.EMPTY,
    },
    UnifiedCategory.GENDER: {
       "Female": UnifiedGender.FEMALE,
       "Male": UnifiedGender.MALE,
       "Neutral": UnifiedGender.NEUTRAL,
       "Child": UnifiedGender.CHILD,
       "Adult": UnifiedGender.ADULT,
       "Chorus_AUDIO_GENDER": UnifiedGender.CHORUS,  # dedup
       "Unkonwn": UnifiedGender.EMPTY,
    },
    UnifiedCategory.TIMBRE: {
       "Warm": UnifiedTimbre.WARM,
       "Ethereal": UnifiedTimbre.ETHEREAL,
       "Husky": UnifiedTimbre.HUSKY,
       "Deep": UnifiedTimbre.DEEP,
       "Loud and sonorous": UnifiedTimbre.LOUD_AND_SONOROUS,
       "Extreme": UnifiedTimbre.EXTREME,
       "Sharp": UnifiedTimbre.SHARP,
       "Bright": UnifiedTimbre.BRIGHT,
       "Sweet_AUDIO_TIMBRE": UnifiedTimbre.SWEET,
       "Powerful": UnifiedTimbre.POWERFUL,
       "Sexy/Lazy": UnifiedTimbre.SEXY_LAZY,
       "Magnetic": UnifiedTimbre.MAGNETIC,
       "Cute_AUDIO_TIMBRE": UnifiedTimbre.CUTE,
       "Electrified voice": UnifiedTimbre.ELECTRIFIED_VOICE,
    }
}


VOCAB2ID_AUDIO_V1 = Vocab2Id(
    CATEGORY_MAP_AUDIO_V1,
    UNIFIED_VOCAB2ID_LEGACY,
)


# =========================================================

CATEGORY_MAP_AUDIO_V2 = {
    UnifiedCategory.GENRE: {
       "Pop": UnifiedGenre.POP,
       "Electronic": UnifiedGenre.ELECTRONIC,
       "Chinese Style": UnifiedGenre.CHINESE_STYLE,
       "Rock": UnifiedGenre.ROCK,
       "Jazz": UnifiedGenre.JAZZ,
       "Hip Hop/Rap": UnifiedGenre.HIP_HOP,
       "Classical": UnifiedGenre.CLASSICAL,
       "R&B/Soul": UnifiedGenre.RNB_SOUL,
       "Metal": UnifiedGenre.METAL,
       "Tuhai": UnifiedGenre.TUHAI,
       "Childhood": UnifiedGenre.CHILDHOOD,
       "Devotional": UnifiedGenre.DEVOTIONAL,
       "Chinese Tradition": UnifiedGenre.CHINESE_TRADITION,
       "Easy Listening": UnifiedGenre.EASY_LISTENING,
       "New Age": UnifiedGenre.NEW_AGE,
       "8 Bit": UnifiedGenre.EIGHT_BIT,
       "Folk": UnifiedGenre.FOLK,
       "Latin": UnifiedGenre.LATIN,
       "Alternative/Indie": UnifiedGenre.ALTERNATIVE_INDIE,
       "Epic": UnifiedGenre.EPIC,
       "BGM": UnifiedGenre.BGM,
       "Chinese Pop": UnifiedGenre.CHINESE_POP,
       "Cantopop": UnifiedGenre.CANTOPOP,
       "Taiwanese Pop": UnifiedGenre.TAIWANESE_POP,
       "Pop Folk": UnifiedGenre.POP_FOLK,
       "Contemporary Pop": UnifiedGenre.CONTEMPORARY_POP,
       "Teen Pop": UnifiedGenre.TEEN_POP,
       "Indie Pop": UnifiedGenre.INDIE_POP,
       "Dream Pop": UnifiedGenre.DREAM_POP,
       "City Pop": UnifiedGenre.CITY_POP,
       "Synth Pop": UnifiedGenre.SYNTH_POP,
       "Celtic Pop": UnifiedGenre.CELTIC_POP,
       "Dance Pop": UnifiedGenre.DANCE_POP,
       "Deep Dance Pop": UnifiedGenre.DEEP_DANCE_POP,
       "Electropop": UnifiedGenre.ELECTROPOP,
       "Chamber Pop": UnifiedGenre.CHAMBER_POP,
       "A cappella": UnifiedGenre.A_CAPPELLA,
       "Country Pop": UnifiedGenre.COUNTRY_POP,
       "EDM": UnifiedGenre.EDM,
       "House": UnifiedGenre.HOUSE,
       "Dubstep": UnifiedGenre.DUBSTEP,
       "Future Bass": UnifiedGenre.FUTURE_BASS,
       "Chillout": UnifiedGenre.CHILLOUT,
       "Trance": UnifiedGenre.TRANCE,
       "Techno": UnifiedGenre.TECHNO,
       "Drum&Bass": UnifiedGenre.DRUMNBASS,
       "Tropical House": UnifiedGenre.TROPICAL_HOUSE,
       "Disco": UnifiedGenre.DISCO,
       "Vaporwave": UnifiedGenre.VAPORWAVE,
       "Trip Hop": UnifiedGenre.TRIP_HOP,
       "Ambient": UnifiedGenre.AMBIENT,
       "Deep Pop Edm": UnifiedGenre.DEEP_POP_EDM,
       "EDM Trap": UnifiedGenre.EDM_TRAP,
       "Future House": UnifiedGenre.FUTURE_HOUSE,
       "China-Wave": UnifiedGenre.CHINA_WAVE,
       "GuFeng Music": UnifiedGenre.GUFENG_MUSIC,
       "Chinoiserie Rap": UnifiedGenre.CHINOISERIE_RAP,
       "Chinoiserie Electronic": UnifiedGenre.CHINOISERIE_ELECTRONIC,
       "Hard Rock": UnifiedGenre.HARD_ROCK,
       "Psychedelic Rock": UnifiedGenre.PSYCHEDELIC_ROCK,
       "Pop Rock": UnifiedGenre.POP_ROCK,
       "Instrumental Rock": UnifiedGenre.INSTRUMENTAL_ROCK,
       "Alternative Rock": UnifiedGenre.ALTERNATIVE_ROCK,
       "Indie Rock": UnifiedGenre.INDIE_ROCK,
       "Post-Rock": UnifiedGenre.POST_ROCK,
       "Lo-Fi": UnifiedGenre.LO_FI,
       "J Rock": UnifiedGenre.J_ROCK,
       "Shoegazing": UnifiedGenre.SHOEGAZING,
       "Math Rock": UnifiedGenre.MATH_ROCK,
       "Surf Rock": UnifiedGenre.SURF_ROCK,
       "Progressive Rock": UnifiedGenre.PROGRESSIVE_ROCK,
       "Soft Rock": UnifiedGenre.SOFT_ROCK,
       "Jazz Pop": UnifiedGenre.JAZZ_POP,
       "Jazz Fusion": UnifiedGenre.JAZZ_FUSION,
       "Bossa Nova": UnifiedGenre.BOSSA_NOVA,
       "Avant-Garde Jazz": UnifiedGenre.AVANT_GARDE_JAZZ,
       "Swing": UnifiedGenre.SWING,
       "Big Band": UnifiedGenre.BIG_BAND,
       "Bop": UnifiedGenre.BOP,
       "Post-Bop": UnifiedGenre.POST_BOP,
       "Smooth Jazz": UnifiedGenre.SMOOTH_JAZZ,
       "Cool Jazz": UnifiedGenre.COOL_JAZZ,
       "Vocal Jazz": UnifiedGenre.VOCAL_JAZZ,
       "Nu Jazz": UnifiedGenre.NU_JAZZ,
       "Free Jazz": UnifiedGenre.FREE_JAZZ,
       "Trap Rap": UnifiedGenre.TRAP_RAP,
       "Old School": UnifiedGenre.OLD_SCHOOL,
       "R&B Rap": UnifiedGenre.RNB_RAP,
       "Jazz Hip Hop": UnifiedGenre.JAZZ_HIP_HOP,
       "Alternative Hip Hop": UnifiedGenre.ALTERNATIVE_HIP_HOP,
       "Instrumental Hip Hop": UnifiedGenre.INSTRUMENTAL_HIP_HOP,
       "Pop Rap": UnifiedGenre.POP_RAP,
       "Hardcore Rap": UnifiedGenre.HARDCORE_RAP,
       "Comedy Hip Hop": UnifiedGenre.COMEDY_HIP_HOP,
       "Hip House": UnifiedGenre.HIP_HOUSE,
       "Chill Beats": UnifiedGenre.CHILL_BEATS,
       "Chorus_AUDIO_GENRE": UnifiedGenre.CHORUS,
       "Chamber Music": UnifiedGenre.CHAMBER_MUSIC,
       "Symphony": UnifiedGenre.SYMPHONY,
       "Funk": UnifiedGenre.FUNK,
       "Contemporary R&B": UnifiedGenre.CONTEMPORARY_RNB,
       "Neo Soul": UnifiedGenre.NEO_SOUL,
       "Soul": UnifiedGenre.SOUL,
       "Pop Soul": UnifiedGenre.POP_SOUL,
       "Black Metal": UnifiedGenre.BLACK_METAL,
       "Death Metal": UnifiedGenre.DEATH_METAL,
       "Glam Metal": UnifiedGenre.GLAM_METAL,
       "Grindcore": UnifiedGenre.GRINDCORE,
       "Power Metal": UnifiedGenre.POWER_METAL,
       "Progressive Metal": UnifiedGenre.PROGRESSIVE_METAL,
       "Speed Metal": UnifiedGenre.SPEED_METAL,
       "DJ": UnifiedGenre.DJ,
       "VinaHouse": UnifiedGenre.VINAHOUSE,
       "MC": UnifiedGenre.MC,
       "Gospel": UnifiedGenre.GOSPEL,
       "Holiday Music": UnifiedGenre.HOLIDAY_MUSIC,
       "Christian Music": UnifiedGenre.CHRISTIAN_MUSIC,
       "Buddhist music": UnifiedGenre.BUDDHIST_MUSIC,
       "Chinese Opera": UnifiedGenre.CHINESE_OPERA,
       "Traditional Chinese Folk": UnifiedGenre.TRADITIONAL_CHINESE_FOLK,
       "Chinese Quyi": UnifiedGenre.CHINESE_QUYI,
       "Red Song": UnifiedGenre.RED_SONG,
       "Folk Pop": UnifiedGenre.FOLK_POP,
       "Indie Folk": UnifiedGenre.INDIE_FOLK,
       "Tango": UnifiedGenre.TANGO,
       "Reggaeton": UnifiedGenre.REGGAETON,
       "Blues": UnifiedGenre.BLUES,
       "Country": UnifiedGenre.COUNTRY_POP,  # share the same tag with "Country Pop"
       "Punk": UnifiedGenre.PUNK,
       "Reggae": UnifiedGenre.REGGAE,
       "Other genre": UnifiedGenre.EMPTY,
       "Boombap": UnifiedGenre.BOOMBAP,
       "Chillwave": UnifiedGenre.CHILLWAVE,
       "Chinese Folk": UnifiedGenre.CHINESE_FOLK,
       "Contemporary Blues": UnifiedGenre.CONTEMPORARY_BLUES,
       "Contemporary Folk": UnifiedGenre.CONTEMPORARY_FOLK,
       "Country Blues": UnifiedGenre.COUNTRY_BLUES,
       "Emo Rap": UnifiedGenre.EMO_RAP,
       "English Folk": UnifiedGenre.ENGLISH_FOLK,
       "Flamenco": UnifiedGenre.FLAMENCO,
       "Gangsta Rap": UnifiedGenre.GANGSTA_RAP,
       "Hardstyle": UnifiedGenre.HARDSTYLE,
       "Heavy Metal": UnifiedGenre.HEAVY_METAL,
       "HiNRG": UnifiedGenre.HINRG,
       "Indian Pop": UnifiedGenre.INDIAN_POP,
       "Italian Pop": UnifiedGenre.ITALIAN_POP,
       "Japanese Folk": UnifiedGenre.JAPANESE_FOLK,
       "J Pop": UnifiedGenre.J_POP,
       "KPop": UnifiedGenre.KPOP,
       "Latin Pop": UnifiedGenre.LATIN_POP,
       "Low pop": UnifiedGenre.LOW_POP,
       "Melbourne Bounce": UnifiedGenre.MELBOURNE_BOUNCE,
       "New Chinese Folk": UnifiedGenre.NEW_CHINESE_FOLK,
       "New Wave": UnifiedGenre.NEW_WAVE,
       "Nostalgic Pop": UnifiedGenre.NOSTALGIC_POP,
       "Pop Punk": UnifiedGenre.POP_PUNK,
       "Rock Blues": UnifiedGenre.ROCK_BLUES,
       "Russian Pop": UnifiedGenre.RUSSIAN_POP,
       "Samba": UnifiedGenre.SAMBA,
       "Shi Dai Qu": UnifiedGenre.SHI_DAI_QU,
       "Soundtrack": UnifiedGenre.SOUNDTRACK,
       "Thrash Metal": UnifiedGenre.THRASH_METAL,
       "Traditional Folk": UnifiedGenre.TRADITIONAL_FOLK,
       "Traditional Jazz": UnifiedGenre.TRADITIONAL_JAZZ,
       "Vulgar Pop": UnifiedGenre.VULGAR_POP,
       "West Coast Hip Hop": UnifiedGenre.WEST_COAST_HIP_HOP,
       "World Music": UnifiedGenre.WORLD_MUSIC,
    },
    UnifiedCategory.MOOD: {
       "Happy": UnifiedMood.HAPPY,
       "Cute/Playful": UnifiedMood.CUTE_PLAYFUL,
       "Excited": UnifiedMood.EXCITED,
       "Funny": UnifiedMood.FUNNY,
       "Inspirational/Hopeful": UnifiedMood.INSPIRATIONAL_HOPEFUL,
       "Sorrow/Sad": UnifiedMood.SORROW_SAD,
       "Sentimental/Melancholic/Lonely": UnifiedMood.SENTIMENTAL_MELANCHOLIC_LONELY,
       "Weird": UnifiedMood.WEIRD,
       "Thrilling/Suspenseful/Tense": UnifiedMood.THRILLING_SUSPENSEFUL_TENSE,
       "Angry/Aggressive": UnifiedMood.ANGRY_AGGRESSIVE,
       "Groovy/Funky": UnifiedMood.GROOVY_FUNKY,
       "Dynamic/Energetic": UnifiedMood.DYNAMIC_ENERGETIC,
       "Romantic": UnifiedMood.ROMANTIC,
       "Nostalgic/Memory": UnifiedMood.NOSTALGIC_MEMORY,
       "Dreamy/Ethereal": UnifiedMood.DREAMY_ETHEREAL,
       "Healing": UnifiedMood.HEALING,
       "Miss": UnifiedMood.MISS,
       "Chill": UnifiedMood.CHILL,
       "Calm/Relaxing": UnifiedMood.CALM_RELAXING,
       "Shocking/magnificent/epic": UnifiedMood.SHOCKING_MAGNIFICENT_EPIC,
       "Shocking/Magnificent/Epic": UnifiedMood.SHOCKING_MAGNIFICENT_EPIC,
       "Mysterious": UnifiedMood.MYSTERIOUS,
       "No Mood": UnifiedMood.EMPTY,
       "Other mood": UnifiedMood.EMPTY,
    },
    UnifiedCategory.THEME: {
       "Halloween": UnifiedTheme.HALLOWEEN,
       "Christmas": UnifiedTheme.CHRISTMAS,
       "New Year": UnifiedTheme.NEW_YEAR,
       "Spring Festival": UnifiedTheme.SPRING_FESTIVAL,
       "Valentine's day": UnifiedTheme.VALENTINES_DAY,
       "Qi Xi": UnifiedTheme.QI_XI,
       "Birthday": UnifiedTheme.BIRTHDAY,
       "Wedding": UnifiedTheme.WEDDING,
       "Funeral": UnifiedTheme.FUNERAL,
       "Graduation": UnifiedTheme.GRADUATION,
       "National's Day": UnifiedTheme.NATIONALS_DAY,
       "Vlog/DailyLife": UnifiedTheme.VLOG_DAILYLIFE,
       "Food": UnifiedTheme.FOOD,
       "Pet/Animals": UnifiedTheme.PET_ANIMALS,
       "Beauty/Fashion": UnifiedTheme.BEAUTY_FASHION,
       "Entertainment": UnifiedTheme.ENTERTAINMENT,
       "babies": UnifiedTheme.BABIES,
       "children": UnifiedTheme.CHILDREN,
       "Transition": UnifiedTheme.TRANSITION,
       "Anime": UnifiedTheme.ANIME,
       "Wake up": UnifiedTheme.WAKE_UP,
       "Family time": UnifiedTheme.FAMILY_TIME,
       "landscape/scenery": UnifiedTheme.LANDSCAPE_SCENERY,
       "Prank": UnifiedTheme.PRANK,
       "Timelapse": UnifiedTheme.TIMELAPSE,
       "Rainy Day": UnifiedTheme.RAINY_DAY,
       "Sunny Day": UnifiedTheme.SUNNY_DAY,
       "Spring": UnifiedTheme.SPRING,
       "Summer": UnifiedTheme.SUMMER,
       "Autumn": UnifiedTheme.AUTUMN,
       "Winter": UnifiedTheme.WINTER,
       "Evening": UnifiedTheme.EVENING,
       "Morning": UnifiedTheme.MORNING,
       "Beach": UnifiedTheme.BEACH,
       "Nightclub": UnifiedTheme.NIGHTCLUB,
       "Coffee Shop": UnifiedTheme.COFFEE_SHOP,
       "Restaurants": UnifiedTheme.RESTAURANTS,
       "Lounge": UnifiedTheme.LOUNGE,
       "Campus": UnifiedTheme.CAMPUS,
       "Park": UnifiedTheme.PARK,
       "Marketplace": UnifiedTheme.MARKETPLACE,
       "Universe": UnifiedTheme.UNIVERSE,
       "Bar": UnifiedTheme.BAR,
       "Theater/Concert hall": UnifiedTheme.THEATER_CONCERT_HALL,
       "Sport": UnifiedTheme.SPORT,
       "Dance": UnifiedTheme.DANCE,
       "Game": UnifiedTheme.GAME,
       "Travel": UnifiedTheme.TRAVEL,
       "Focus": UnifiedTheme.FOCUS,
       "Party": UnifiedTheme.PARTY,
       "Commute": UnifiedTheme.COMMUTE,
       "Roadtrip": UnifiedTheme.ROADTRIP,
       "Meditation": UnifiedTheme.MEDITATION,
       "Relaxation": UnifiedTheme.RELAXATION,
       "Cleaning and Chores": UnifiedTheme.CLEANING_AND_CHORES,
       "Running": UnifiedTheme.RUNNING,
       "Date": UnifiedTheme.DATE,
       "Danceable": UnifiedTheme.DANCEABLE,
       "Flirt": UnifiedTheme.FLIRT,
       "Bedtime": UnifiedTheme.BED_TIME,
       "Broke up": UnifiedTheme.BROKE_UP,
       "Dream": UnifiedTheme.DREAM,
       "Drive": UnifiedTheme.DRIVE,
       "Friendship": UnifiedTheme.FRIENDSHIP,
       "Love": UnifiedTheme.LOVE,
       "Yoga": UnifiedTheme.YOGA,
       "Other scene": UnifiedTheme.EMPTY,
       "Mid-autumn Festival": UnifiedTheme.MID_AUTUMN_FESTIVAL,
    },
    UnifiedCategory.GENDER: {
       "Female": UnifiedGender.FEMALE,
       "Male": UnifiedGender.MALE,
       "Neutral": UnifiedGender.NEUTRAL,
       "Child": UnifiedGender.CHILD,
       "Adult": UnifiedGender.ADULT,
       "Chorus_AUDIO_GENDER": UnifiedGender.CHORUS,  # dedup
       "Unkonwn": UnifiedGender.EMPTY,
    },
    UnifiedCategory.TIMBRE: {
       "Warm": UnifiedTimbre.WARM,
       "Ethereal": UnifiedTimbre.ETHEREAL,
       "Husky": UnifiedTimbre.HUSKY,
       "Deep": UnifiedTimbre.DEEP,
       "Loud and sonorous": UnifiedTimbre.LOUD_AND_SONOROUS,
       "Extreme": UnifiedTimbre.EXTREME,
       "Sharp": UnifiedTimbre.SHARP,
       "Bright": UnifiedTimbre.BRIGHT,
       "Sweet_AUDIO_TIMBRE": UnifiedTimbre.SWEET,
       "Powerful": UnifiedTimbre.POWERFUL,
       "Sexy/Lazy": UnifiedTimbre.SEXY_LAZY,
       "Magnetic": UnifiedTimbre.MAGNETIC,
       "Cute_AUDIO_TIMBRE": UnifiedTimbre.CUTE,
       "Electrified voice": UnifiedTimbre.ELECTRIFIED_VOICE,
    }
}


VOCAB2ID_AUDIO_V2 = Vocab2Id(
    CATEGORY_MAP_AUDIO_V2,
    UNIFIED_VOCAB2ID_LEGACY,
)

CATEGORY_MAP_AUDIO_V3 = {
    UnifiedCategory.GENRE: {
       "Pop": UnifiedGenre.POP,
       "Electronic": UnifiedGenre.ELECTRONIC,
       "Chinese Style": UnifiedGenre.CHINESE_STYLE,
       "Rock": UnifiedGenre.ROCK,
       "Jazz": UnifiedGenre.JAZZ,
       "Hip Hop/Rap": UnifiedGenre.HIP_HOP,
       "Classical": UnifiedGenre.CLASSICAL,
       "R&B/Soul": UnifiedGenre.RNB_SOUL,
       "Metal": UnifiedGenre.METAL,
       "Tuhai": UnifiedGenre.TUHAI,
       "Childhood": UnifiedGenre.CHILDHOOD,
       "Devotional": UnifiedGenre.DEVOTIONAL,
       "Chinese Tradition": UnifiedGenre.CHINESE_TRADITION,
       "Easy Listening": UnifiedGenre.EASY_LISTENING,
       "New Age": UnifiedGenre.NEW_AGE,
       "8 Bit": UnifiedGenre.EIGHT_BIT,
       "Folk": UnifiedGenre.FOLK,
       "Latin": UnifiedGenre.LATIN,
       "Alternative/Indie": UnifiedGenre.ALTERNATIVE_INDIE,
       "Epic": UnifiedGenre.EPIC,
       "BGM": UnifiedGenre.BGM,
       "Chinese Pop": UnifiedGenre.CHINESE_POP,
       "Cantopop": UnifiedGenre.CANTOPOP,
       "Taiwanese Pop": UnifiedGenre.TAIWANESE_POP,
       "Pop Folk": UnifiedGenre.POP_FOLK,
       "Contemporary Pop": UnifiedGenre.CONTEMPORARY_POP,
       "Teen Pop": UnifiedGenre.TEEN_POP,
       "Indie Pop": UnifiedGenre.INDIE_POP,
       "Dream Pop": UnifiedGenre.DREAM_POP,
       "City Pop": UnifiedGenre.CITY_POP,
       "Synth Pop": UnifiedGenre.SYNTH_POP,
       "Celtic Pop": UnifiedGenre.CELTIC_POP,
       "Dance Pop": UnifiedGenre.DANCE_POP,
       "Deep Dance Pop": UnifiedGenre.DEEP_DANCE_POP,
       "Electropop": UnifiedGenre.ELECTROPOP,
       "Chamber Pop": UnifiedGenre.CHAMBER_POP,
       "A cappella": UnifiedGenre.A_CAPPELLA,
       "Country Pop": UnifiedGenre.COUNTRY_POP,
       "EDM": UnifiedGenre.EDM,
       "House": UnifiedGenre.HOUSE,
       "Dubstep": UnifiedGenre.DUBSTEP,
       "Future Bass": UnifiedGenre.FUTURE_BASS,
       "Chillout": UnifiedGenre.CHILLOUT,
       "Trance": UnifiedGenre.TRANCE,
       "Techno": UnifiedGenre.TECHNO,
       "Drum&Bass": UnifiedGenre.DRUMNBASS,
       "Tropical House": UnifiedGenre.TROPICAL_HOUSE,
       "Disco": UnifiedGenre.DISCO,
       "Vaporwave": UnifiedGenre.VAPORWAVE,
       "Trip Hop": UnifiedGenre.TRIP_HOP,
       "Ambient": UnifiedGenre.AMBIENT,
       "Deep Pop Edm": UnifiedGenre.DEEP_POP_EDM,
       "EDM Trap": UnifiedGenre.EDM_TRAP,
       "Future House": UnifiedGenre.FUTURE_HOUSE,
       "China-Wave": UnifiedGenre.CHINA_WAVE,
       "GuFeng Music": UnifiedGenre.GUFENG_MUSIC,
       "Chinoiserie Rap": UnifiedGenre.CHINOISERIE_RAP,
       "Chinoiserie Electronic": UnifiedGenre.CHINOISERIE_ELECTRONIC,
       "Hard Rock": UnifiedGenre.HARD_ROCK,
       "Psychedelic Rock": UnifiedGenre.PSYCHEDELIC_ROCK,
       "Pop Rock": UnifiedGenre.POP_ROCK,
       "Instrumental Rock": UnifiedGenre.INSTRUMENTAL_ROCK,
       "Alternative Rock": UnifiedGenre.ALTERNATIVE_ROCK,
       "Indie Rock": UnifiedGenre.INDIE_ROCK,
       "Post-Rock": UnifiedGenre.POST_ROCK,
       "Lo-Fi": UnifiedGenre.LO_FI,
       "J Rock": UnifiedGenre.J_ROCK,
       "Shoegazing": UnifiedGenre.SHOEGAZING,
       "Math Rock": UnifiedGenre.MATH_ROCK,
       "Surf Rock": UnifiedGenre.SURF_ROCK,
       "Progressive Rock": UnifiedGenre.PROGRESSIVE_ROCK,
       "Soft Rock": UnifiedGenre.SOFT_ROCK,
       "Jazz Pop": UnifiedGenre.JAZZ_POP,
       "Jazz Fusion": UnifiedGenre.JAZZ_FUSION,
       "Bossa Nova": UnifiedGenre.BOSSA_NOVA,
       "Avant-Garde Jazz": UnifiedGenre.AVANT_GARDE_JAZZ,
       "Swing": UnifiedGenre.SWING,
       "Big Band": UnifiedGenre.BIG_BAND,
       "Bop": UnifiedGenre.BOP,
       "Post-Bop": UnifiedGenre.POST_BOP,
       "Smooth Jazz": UnifiedGenre.SMOOTH_JAZZ,
       "Cool Jazz": UnifiedGenre.COOL_JAZZ,
       "Vocal Jazz": UnifiedGenre.VOCAL_JAZZ,
       "Nu Jazz": UnifiedGenre.NU_JAZZ,
       "Free Jazz": UnifiedGenre.FREE_JAZZ,
       "Trap Rap": UnifiedGenre.TRAP_RAP,
       "Old School": UnifiedGenre.OLD_SCHOOL,
       "R&B Rap": UnifiedGenre.RNB_RAP,
       "Jazz Hip Hop": UnifiedGenre.JAZZ_HIP_HOP,
       "Alternative Hip Hop": UnifiedGenre.ALTERNATIVE_HIP_HOP,
       "Instrumental Hip Hop": UnifiedGenre.INSTRUMENTAL_HIP_HOP,
       "Pop Rap": UnifiedGenre.POP_RAP,
       "Hardcore Rap": UnifiedGenre.HARDCORE_RAP,
       "Comedy Hip Hop": UnifiedGenre.COMEDY_HIP_HOP,
       "Hip House": UnifiedGenre.HIP_HOUSE,
       "Chill Beats": UnifiedGenre.CHILL_BEATS,
       "Chorus_AUDIO_GENRE": UnifiedGenre.CHORUS,
       "Chamber Music": UnifiedGenre.CHAMBER_MUSIC,
       "Symphony": UnifiedGenre.SYMPHONY,
       "Funk": UnifiedGenre.FUNK,
       "Contemporary R&B": UnifiedGenre.CONTEMPORARY_RNB,
       "Neo Soul": UnifiedGenre.NEO_SOUL,
       "Soul": UnifiedGenre.SOUL,
       "Pop Soul": UnifiedGenre.POP_SOUL,
       "Black Metal": UnifiedGenre.BLACK_METAL,
       "Death Metal": UnifiedGenre.DEATH_METAL,
       "Glam Metal": UnifiedGenre.GLAM_METAL,
       "Grindcore": UnifiedGenre.GRINDCORE,
       "Power Metal": UnifiedGenre.POWER_METAL,
       "Progressive Metal": UnifiedGenre.PROGRESSIVE_METAL,
       "Speed Metal": UnifiedGenre.SPEED_METAL,
       "DJ": UnifiedGenre.DJ,
       "VinaHouse": UnifiedGenre.VINAHOUSE,
       "MC": UnifiedGenre.MC,
       "Gospel": UnifiedGenre.GOSPEL,
       "Holiday Music": UnifiedGenre.HOLIDAY_MUSIC,
       "Christian Music": UnifiedGenre.CHRISTIAN_MUSIC,
       "Buddhist music": UnifiedGenre.BUDDHIST_MUSIC,
       "Chinese Opera": UnifiedGenre.CHINESE_OPERA,
       "Traditional Chinese Folk": UnifiedGenre.TRADITIONAL_CHINESE_FOLK,
       "Chinese Quyi": UnifiedGenre.CHINESE_QUYI,
       "Red Song": UnifiedGenre.RED_SONG,
       "Red Song/": UnifiedGenre.RED_SONG,
       "Folk Pop": UnifiedGenre.FOLK_POP,
       "Indie Folk": UnifiedGenre.INDIE_FOLK,
       "Tango": UnifiedGenre.TANGO,
       "Reggaeton": UnifiedGenre.REGGAETON,
       "Blues": UnifiedGenre.BLUES,
       "Country": UnifiedGenre.COUNTRY_POP,  # share the same tag with "Country Pop"
       "Punk": UnifiedGenre.PUNK,
       "Reggae": UnifiedGenre.REGGAE,
       "Other genre": UnifiedGenre.EMPTY,
       "Boombap": UnifiedGenre.BOOMBAP,
       "Chillwave": UnifiedGenre.CHILLWAVE,
       "Chinese Folk": UnifiedGenre.CHINESE_FOLK,
       "Contemporary Blues": UnifiedGenre.CONTEMPORARY_BLUES,
       "Contemporary Folk": UnifiedGenre.CONTEMPORARY_FOLK,
       "Country Blues": UnifiedGenre.COUNTRY_BLUES,
       "Emo Rap": UnifiedGenre.EMO_RAP,
       "English Folk": UnifiedGenre.ENGLISH_FOLK,
       "Flamenco": UnifiedGenre.FLAMENCO,
       "Gangsta Rap": UnifiedGenre.GANGSTA_RAP,
       "Hardstyle": UnifiedGenre.HARDSTYLE,
       "Heavy Metal": UnifiedGenre.HEAVY_METAL,
       "HiNRG": UnifiedGenre.HINRG,
       "Indian Pop": UnifiedGenre.INDIAN_POP,
       "Italian Pop": UnifiedGenre.ITALIAN_POP,
       "Japanese Folk": UnifiedGenre.JAPANESE_FOLK,
       "J Pop": UnifiedGenre.J_POP,
       "KPop": UnifiedGenre.KPOP,
       "K-Pop": UnifiedGenre.KPOP,
       "Latin Pop": UnifiedGenre.LATIN_POP,
       "Low pop": UnifiedGenre.LOW_POP,
       "Melbourne Bounce": UnifiedGenre.MELBOURNE_BOUNCE,
       "New Chinese Folk": UnifiedGenre.NEW_CHINESE_FOLK,
       "New Wave": UnifiedGenre.NEW_WAVE,
       "Nostalgic Pop": UnifiedGenre.NOSTALGIC_POP,
       "Pop Punk": UnifiedGenre.POP_PUNK,
       "Rock Blues": UnifiedGenre.ROCK_BLUES,
       "Rock BluesNew": UnifiedGenre.ROCK_BLUES,
       "Russian Pop": UnifiedGenre.RUSSIAN_POP,
       "Samba": UnifiedGenre.SAMBA,
       "Shi Dai Qu": UnifiedGenre.SHI_DAI_QU,
       "Soundtrack": UnifiedGenre.SOUNDTRACK,
       "Thrash Metal": UnifiedGenre.THRASH_METAL,
       "Traditional Folk": UnifiedGenre.TRADITIONAL_FOLK,
       "Traditional Jazz": UnifiedGenre.TRADITIONAL_JAZZ,
       "Vulgar Pop": UnifiedGenre.VULGAR_POP,
       "West Coast Hip Hop": UnifiedGenre.WEST_COAST_HIP_HOP,
       "World Music": UnifiedGenre.WORLD_MUSIC,
       "Breakbeat": UnifiedGenre.BREAKBEAT,
       "Country Folk": UnifiedGenre.COUNTRY_FOLK,
       "Country Rock": UnifiedGenre.COUNTRY_ROCK,
       "Jazz Blues": UnifiedGenre.JAZZ_BLUES,
    },
    UnifiedCategory.MOOD: {
       "Happy": UnifiedMood.HAPPY,
       "Cute/Playful": UnifiedMood.CUTE_PLAYFUL,
       "Excited": UnifiedMood.EXCITED,
       "Funny": UnifiedMood.FUNNY,
       "Inspirational/Hopeful": UnifiedMood.INSPIRATIONAL_HOPEFUL,
       "Sorrow/Sad": UnifiedMood.SORROW_SAD,
       "Sentimental/Melancholic/Lonely": UnifiedMood.SENTIMENTAL_MELANCHOLIC_LONELY,
       "Weird": UnifiedMood.WEIRD,
       "Thrilling/Suspenseful/Tense": UnifiedMood.THRILLING_SUSPENSEFUL_TENSE,
       "Angry/Aggressive": UnifiedMood.ANGRY_AGGRESSIVE,
       "Groovy/Funky": UnifiedMood.GROOVY_FUNKY,
       "Dynamic/Energetic": UnifiedMood.DYNAMIC_ENERGETIC,
       "Romantic": UnifiedMood.ROMANTIC,
       "Nostalgic/Memory": UnifiedMood.NOSTALGIC_MEMORY,
       "Dreamy/Ethereal": UnifiedMood.DREAMY_ETHEREAL,
       "Healing": UnifiedMood.HEALING,
       "Miss": UnifiedMood.MISS,
       "Chill": UnifiedMood.CHILL,
       "Calm/Relaxing": UnifiedMood.CALM_RELAXING,
       "Shocking/magnificent/epic": UnifiedMood.SHOCKING_MAGNIFICENT_EPIC,
       "Mysterious": UnifiedMood.MYSTERIOUS,
       "No Mood": UnifiedMood.EMPTY,
       "Other mood": UnifiedMood.EMPTY,
    },
    UnifiedCategory.THEME: {
       "Halloween": UnifiedTheme.HALLOWEEN,
       "Christmas": UnifiedTheme.CHRISTMAS,
       "New Year": UnifiedTheme.NEW_YEAR,
       "Spring Festival": UnifiedTheme.SPRING_FESTIVAL,
       "Valentine's day": UnifiedTheme.VALENTINES_DAY,
       "Qi Xi": UnifiedTheme.QI_XI,
       "Birthday": UnifiedTheme.BIRTHDAY,
       "Wedding": UnifiedTheme.WEDDING,
       "Funeral": UnifiedTheme.FUNERAL,
       "Graduation": UnifiedTheme.GRADUATION,
       "National's Day": UnifiedTheme.NATIONALS_DAY,
       "Vlog/DailyLife": UnifiedTheme.VLOG_DAILYLIFE,
       "Food": UnifiedTheme.FOOD,
       "Pet/Animals": UnifiedTheme.PET_ANIMALS,
       "Beauty/Fashion": UnifiedTheme.BEAUTY_FASHION,
       "Entertainment": UnifiedTheme.ENTERTAINMENT,
       "babies": UnifiedTheme.BABIES,
       "children": UnifiedTheme.CHILDREN,
       "Transition": UnifiedTheme.TRANSITION,
       "Anime": UnifiedTheme.ANIME,
       "Wake up": UnifiedTheme.WAKE_UP,
       "Family time": UnifiedTheme.FAMILY_TIME,
       "landscape/scenery": UnifiedTheme.LANDSCAPE_SCENERY,
       "Prank": UnifiedTheme.PRANK,
       "Timelapse": UnifiedTheme.TIMELAPSE,
       "Rainy Day": UnifiedTheme.RAINY_DAY,
       "Sunny Day": UnifiedTheme.SUNNY_DAY,
       "Spring": UnifiedTheme.SPRING,
       "Summer": UnifiedTheme.SUMMER,
       "Autumn": UnifiedTheme.AUTUMN,
       "Winter": UnifiedTheme.WINTER,
       "Evening": UnifiedTheme.EVENING,
       "Morning": UnifiedTheme.MORNING,
       "Beach": UnifiedTheme.BEACH,
       "Nightclub": UnifiedTheme.NIGHTCLUB,
       "Coffee Shop": UnifiedTheme.COFFEE_SHOP,
       "Restaurants": UnifiedTheme.RESTAURANTS,
       "Lounge": UnifiedTheme.LOUNGE,
       "Campus": UnifiedTheme.CAMPUS,
       "Park": UnifiedTheme.PARK,
       "Marketplace": UnifiedTheme.MARKETPLACE,
       "Universe": UnifiedTheme.UNIVERSE,
       "Bar": UnifiedTheme.BAR,
       "Theater/Concert hall": UnifiedTheme.THEATER_CONCERT_HALL,
       "Sport": UnifiedTheme.SPORT,
       "Dance": UnifiedTheme.DANCE,
       "Game": UnifiedTheme.GAME,
       "Travel": UnifiedTheme.TRAVEL,
       "Focus": UnifiedTheme.FOCUS,
       "Party": UnifiedTheme.PARTY,
       "Commute": UnifiedTheme.COMMUTE,
       "Roadtrip": UnifiedTheme.ROADTRIP,
       "Meditation": UnifiedTheme.MEDITATION,
       "Relaxation": UnifiedTheme.RELAXATION,
       "Cleaning and Chores": UnifiedTheme.CLEANING_AND_CHORES,
       "Running": UnifiedTheme.RUNNING,
       "Date": UnifiedTheme.DATE,
       "Danceable": UnifiedTheme.DANCEABLE,
       "Flirt": UnifiedTheme.FLIRT,
       "Bedtime": UnifiedTheme.BED_TIME,
       "Broke up": UnifiedTheme.BROKE_UP,
       "Dream": UnifiedTheme.DREAM,
       "Drive": UnifiedTheme.DRIVE,
       "Friendship": UnifiedTheme.FRIENDSHIP,
       "Love": UnifiedTheme.LOVE,
       "Yoga": UnifiedTheme.YOGA,
       "Other scene": UnifiedTheme.EMPTY,
       "Mid-autumn Festival": UnifiedTheme.MID_AUTUMN_FESTIVAL,
    },
    UnifiedCategory.GENDER: {
       "Female": UnifiedGender.FEMALE,
       "Male": UnifiedGender.MALE,
       "Neutral": UnifiedGender.NEUTRAL,
       "Child": UnifiedGender.CHILD,
       "Adult": UnifiedGender.ADULT,
       "Chorus_AUDIO_GENDER": UnifiedGender.CHORUS,  # dedup
       "Unkonwn": UnifiedGender.EMPTY,
    },
    UnifiedCategory.TIMBRE: {
       "Warm": UnifiedTimbre.WARM,
       "Ethereal": UnifiedTimbre.ETHEREAL,
       "Husky": UnifiedTimbre.HUSKY,
       "Deep": UnifiedTimbre.DEEP,
       "Loud and sonorous": UnifiedTimbre.LOUD_AND_SONOROUS,
       "Extreme": UnifiedTimbre.EXTREME,
       "Sharp": UnifiedTimbre.SHARP,
       "Bright": UnifiedTimbre.BRIGHT,
       "Sweet_AUDIO_TIMBRE": UnifiedTimbre.SWEET,
       "Powerful": UnifiedTimbre.POWERFUL,
       "Sexy/Lazy": UnifiedTimbre.SEXY_LAZY,
       "Magnetic": UnifiedTimbre.MAGNETIC,
       "Cute_AUDIO_TIMBRE": UnifiedTimbre.CUTE,
       "Electrified voice": UnifiedTimbre.ELECTRIFIED_VOICE,
    }
}


VOCAB2ID_AUDIO_V3 = Vocab2Id(
    CATEGORY_MAP_AUDIO_V3,
    UNIFIED_VOCAB2ID_LEGACY,
)


CATEGORY_MAP_AUDIO_V4 = {
    UnifiedCategory.GENRE: {
        "Pop": UnifiedGenre.POP,
        "Electronic": UnifiedGenre.ELECTRONIC,
        "Chinese Style": UnifiedGenre.CHINESE_STYLE,
        "Rock": UnifiedGenre.ROCK,
        "Jazz": UnifiedGenre.JAZZ,
        "Hip Hop/Rap": UnifiedGenre.HIP_HOP,
        "Classical": UnifiedGenre.CLASSICAL,
        "R&B/Soul": UnifiedGenre.RNB_SOUL,
        "Metal": UnifiedGenre.METAL,
        "Tuhai": UnifiedGenre.TUHAI,
        "Childhood": UnifiedGenre.CHILDHOOD,
        "Devotional": UnifiedGenre.DEVOTIONAL,
        "Chinese Tradition": UnifiedGenre.CHINESE_TRADITION,
        "Easy Listening": UnifiedGenre.EASY_LISTENING,
        "New Age": UnifiedGenre.NEW_AGE,
        "8 Bit": UnifiedGenre.EIGHT_BIT,
        "Folk": UnifiedGenre.FOLK,
        "Latin": UnifiedGenre.LATIN,
        "Alternative/Indie": UnifiedGenre.ALTERNATIVE_INDIE,
        "Epic": UnifiedGenre.EPIC,
        "BGM": UnifiedGenre.BGM,
        "Chinese Pop": UnifiedGenre.CHINESE_POP,
        "Cantopop": UnifiedGenre.CANTOPOP,
        "Taiwanese Pop": UnifiedGenre.TAIWANESE_POP,
        "Pop Folk": UnifiedGenre.POP_FOLK,
        "Contemporary Pop": UnifiedGenre.CONTEMPORARY_POP,
        "Teen Pop": UnifiedGenre.TEEN_POP,
        "Indie Pop": UnifiedGenre.INDIE_POP,
        "Dream Pop": UnifiedGenre.DREAM_POP,
        "City Pop": UnifiedGenre.CITY_POP,
        "Synth Pop": UnifiedGenre.SYNTH_POP,
        "Celtic Pop": UnifiedGenre.CELTIC_POP,
        "Dance Pop": UnifiedGenre.DANCE_POP,
        "Deep Dance Pop": UnifiedGenre.DEEP_DANCE_POP,
        "Electropop": UnifiedGenre.ELECTROPOP,
        "Chamber Pop": UnifiedGenre.CHAMBER_POP,
        "A cappella": UnifiedGenre.A_CAPPELLA,
        "Country Pop": UnifiedGenre.COUNTRY_POP,
        "EDM": UnifiedGenre.EDM,
        "House": UnifiedGenre.HOUSE,
        "Dubstep": UnifiedGenre.DUBSTEP,
        "Future Bass": UnifiedGenre.FUTURE_BASS,
        "Chillout": UnifiedGenre.CHILLOUT,
        "Trance": UnifiedGenre.TRANCE,
        "Techno": UnifiedGenre.TECHNO,
        "Drum&Bass": UnifiedGenre.DRUMNBASS,
        "Tropical House": UnifiedGenre.TROPICAL_HOUSE,
        "Disco": UnifiedGenre.DISCO,
        "Vaporwave": UnifiedGenre.VAPORWAVE,
        "Trip Hop": UnifiedGenre.TRIP_HOP,
        "Ambient": UnifiedGenre.AMBIENT,
        "Deep Pop Edm": UnifiedGenre.DEEP_POP_EDM,
        "EDM Trap": UnifiedGenre.EDM_TRAP,
        "Future House": UnifiedGenre.FUTURE_HOUSE,
        "China-Wave": UnifiedGenre.CHINA_WAVE,
        "ChinaWave": UnifiedGenre.CHINA_WAVE,
        "GuFeng Music": UnifiedGenre.GUFENG_MUSIC,
        "Chinoiserie Rap": UnifiedGenre.CHINOISERIE_RAP,
        "Chinoiserie Electronic": UnifiedGenre.CHINOISERIE_ELECTRONIC,
        "Hard Rock": UnifiedGenre.HARD_ROCK,
        "Psychedelic Rock": UnifiedGenre.PSYCHEDELIC_ROCK,
        "Pop Rock": UnifiedGenre.POP_ROCK,
        "Instrumental Rock": UnifiedGenre.INSTRUMENTAL_ROCK,
        "Alternative Rock": UnifiedGenre.ALTERNATIVE_ROCK,
        "Indie Rock": UnifiedGenre.INDIE_ROCK,
        "Post-Rock": UnifiedGenre.POST_ROCK,
        "Lo-Fi": UnifiedGenre.LO_FI,
        "J Rock": UnifiedGenre.J_ROCK,
        "Shoegazing": UnifiedGenre.SHOEGAZING,
        "Math Rock": UnifiedGenre.MATH_ROCK,
        "Surf Rock": UnifiedGenre.SURF_ROCK,
        "Progressive Rock": UnifiedGenre.PROGRESSIVE_ROCK,
        "Soft Rock": UnifiedGenre.SOFT_ROCK,
        "Jazz Pop": UnifiedGenre.JAZZ_POP,
        "Jazz Fusion": UnifiedGenre.JAZZ_FUSION,
        "Bossa Nova": UnifiedGenre.BOSSA_NOVA,
        "Avant-Garde Jazz": UnifiedGenre.AVANT_GARDE_JAZZ,
        "Swing": UnifiedGenre.SWING,
        "Big Band": UnifiedGenre.BIG_BAND,
        "Bop": UnifiedGenre.BOP,
        "Post-Bop": UnifiedGenre.POST_BOP,
        "Smooth Jazz": UnifiedGenre.SMOOTH_JAZZ,
        "Cool Jazz": UnifiedGenre.COOL_JAZZ,
        "Vocal Jazz": UnifiedGenre.VOCAL_JAZZ,
        "Nu Jazz": UnifiedGenre.NU_JAZZ,
        "Free Jazz": UnifiedGenre.FREE_JAZZ,
        "Trap Rap": UnifiedGenre.TRAP_RAP,
        "Old School": UnifiedGenre.OLD_SCHOOL,
        "R&B Rap": UnifiedGenre.RNB_RAP,
        "Jazz Hip Hop": UnifiedGenre.JAZZ_HIP_HOP,
        "Alternative Hip Hop": UnifiedGenre.ALTERNATIVE_HIP_HOP,
        "Instrumental Hip Hop": UnifiedGenre.INSTRUMENTAL_HIP_HOP,
        "Pop Rap": UnifiedGenre.POP_RAP,
        "Hardcore Rap": UnifiedGenre.HARDCORE_RAP,
        "Comedy Hip Hop": UnifiedGenre.COMEDY_HIP_HOP,
        "Hip House": UnifiedGenre.HIP_HOUSE,
        "Chill Beats": UnifiedGenre.CHILL_BEATS,
        "Chorus_AUDIO_GENRE": UnifiedGenre.CHORUS,
        "Chamber Music": UnifiedGenre.CHAMBER_MUSIC,
        "Symphony": UnifiedGenre.SYMPHONY,
        "Funk": UnifiedGenre.FUNK,
        "Contemporary R&B": UnifiedGenre.CONTEMPORARY_RNB,
        "Neo Soul": UnifiedGenre.NEO_SOUL,
        "Soul": UnifiedGenre.SOUL,
        "Pop Soul": UnifiedGenre.POP_SOUL,
        "Black Metal": UnifiedGenre.BLACK_METAL,
        "Death Metal": UnifiedGenre.DEATH_METAL,
        "Glam Metal": UnifiedGenre.GLAM_METAL,
        "Grindcore": UnifiedGenre.GRINDCORE,
        "Power Metal": UnifiedGenre.POWER_METAL,
        "Progressive Metal": UnifiedGenre.PROGRESSIVE_METAL,
        "Progressive MetalProg Metal": UnifiedGenre.PROGRESSIVE_METAL,
        "Speed Metal": UnifiedGenre.SPEED_METAL,
        "DJ": UnifiedGenre.DJ,
        "VinaHouse": UnifiedGenre.VINAHOUSE,
        "MC": UnifiedGenre.MC,
        "Gospel": UnifiedGenre.GOSPEL,
        "Holiday Music": UnifiedGenre.HOLIDAY_MUSIC,
        "Christian Music": UnifiedGenre.CHRISTIAN_MUSIC,
        "Buddhist music": UnifiedGenre.BUDDHIST_MUSIC,
        "Chinese Opera": UnifiedGenre.CHINESE_OPERA,
        "Traditional Chinese Folk": UnifiedGenre.TRADITIONAL_CHINESE_FOLK,
        "Chinese Quyi": UnifiedGenre.CHINESE_QUYI,
        "Red Song": UnifiedGenre.RED_SONG,
        "Folk Pop": UnifiedGenre.FOLK_POP,
        "Indie Folk": UnifiedGenre.INDIE_FOLK,
        "Tango": UnifiedGenre.TANGO,
        "Reggaeton": UnifiedGenre.REGGAETON,
        "Blues": UnifiedGenre.BLUES,
        "Country": UnifiedGenre.COUNTRY_POP,  # share the same tag with "Country Pop"
        "Punk": UnifiedGenre.PUNK,
        "Reggae": UnifiedGenre.REGGAE,
        "Other genre": UnifiedGenre.EMPTY,
        "Boombap": UnifiedGenre.BOOMBAP,
        "Chillwave": UnifiedGenre.CHILLWAVE,
        "Chinese Folk": UnifiedGenre.CHINESE_FOLK,
        "Contemporary Blues": UnifiedGenre.CONTEMPORARY_BLUES,
        "Contemporary Folk": UnifiedGenre.CONTEMPORARY_FOLK,
        "Country Blues": UnifiedGenre.COUNTRY_BLUES,
        "Emo Rap": UnifiedGenre.EMO_RAP,
        "English Folk": UnifiedGenre.ENGLISH_FOLK,
        "Flamenco": UnifiedGenre.FLAMENCO,
        "Gangsta Rap": UnifiedGenre.GANGSTA_RAP,
        "Hardstyle": UnifiedGenre.HARDSTYLE,
        "Heavy Metal": UnifiedGenre.HEAVY_METAL,
        "HiNRG": UnifiedGenre.HINRG,
        "Indian Pop": UnifiedGenre.INDIAN_POP,
        "Italian Pop": UnifiedGenre.ITALIAN_POP,
        "Japanese Folk": UnifiedGenre.JAPANESE_FOLK,
        "J Pop": UnifiedGenre.J_POP,
        "KPop": UnifiedGenre.KPOP,
        "Latin Pop": UnifiedGenre.LATIN_POP,
        "Low pop": UnifiedGenre.LOW_POP,
        "Melbourne Bounce": UnifiedGenre.MELBOURNE_BOUNCE,
        "New Chinese Folk": UnifiedGenre.NEW_CHINESE_FOLK,
        "New Wave": UnifiedGenre.NEW_WAVE,
        "Nostalgic Pop": UnifiedGenre.NOSTALGIC_POP,
        "Pop Punk": UnifiedGenre.POP_PUNK,
        "Rock Blues": UnifiedGenre.ROCK_BLUES,
        "Rock BluesNew": UnifiedGenre.ROCK_BLUES,
        "Russian Pop": UnifiedGenre.RUSSIAN_POP,
        "Samba": UnifiedGenre.SAMBA,
        "Shi Dai Qu": UnifiedGenre.SHI_DAI_QU,
        "Soundtrack": UnifiedGenre.SOUNDTRACK,
        "Thrash Metal": UnifiedGenre.THRASH_METAL,
        "Traditional Folk": UnifiedGenre.TRADITIONAL_FOLK,
        "Traditional Jazz": UnifiedGenre.TRADITIONAL_JAZZ,
        "Vulgar Pop": UnifiedGenre.VULGAR_POP,
        "West Coast Hip Hop": UnifiedGenre.WEST_COAST_HIP_HOP,
        "World Music": UnifiedGenre.WORLD_MUSIC,
        "Breakbeat": UnifiedGenre.BREAKBEAT,
        "Country Folk": UnifiedGenre.COUNTRY_FOLK,
        "Country Rock": UnifiedGenre.COUNTRY_ROCK,
        "Jazz Blues": UnifiedGenre.JAZZ_BLUES,
        # audio tags ver2
        "Acapella": UnifiedGenre.ACAPELLA,
        "Acid Jazz": UnifiedGenre.ACID_JAZZ,
        "African Folk": UnifiedGenre.AFRICAN_FOLK,
        "African Pop": UnifiedGenre.AFRICAN_POP,
        "Arabic Folk": UnifiedGenre.ARABIC_FOLK,
        "Arabic Pop": UnifiedGenre.ARABIC_POP,
        "Baroque": UnifiedGenre.BAROQUE,
        "Bass House": UnifiedGenre.BASS_HOUSE,
        "Bebop": UnifiedGenre.BEHOP,
        "Bhangra": UnifiedGenre.BHANGRA,
        "Blue Grass": UnifiedGenre.BLUE_GRASS,
        "Blues Rock": UnifiedGenre.BLUES_ROCK,
        "Boogie Woogie": UnifiedGenre.BOOGIE_WOOGIE,
        "Brazilian Funk Style": UnifiedGenre.BRAZILIAN_FUNK_STYLE,
        "Brit Pop": UnifiedGenre.BRIT_POP,
        "Bubblegum Bass": UnifiedGenre.BUBBLEGUM_BASS,
        "Chinese Ballad Pop": UnifiedGenre.CHINESE_BALLAD_POP,
        "Chinese Traditional Instrumental Music": UnifiedGenre.CHINESE_TRADITIONAL_INSTRUMENTAL_MUSIC,
        "Classical period": UnifiedGenre.CLASSICAL_PERIOD,
        "Classic R&B / Soul": UnifiedGenre.CLASSIC_RNB_SOUL,
        "Contemporary classical music": UnifiedGenre.CONTEMPORARY_CLASSICAL_MUSIC,
        "Dancehall": UnifiedGenre.DANCEHALL,
        "Dance Punk": UnifiedGenre.DANCE_PUNK,
        "Dark Ambient": UnifiedGenre.DARK_AMBIENT,
        "Deathcore": UnifiedGenre.DEATHCORE,
        "Doom Metal": UnifiedGenre.DOOM_METAL,
        "Doo-Wop": UnifiedGenre.DOO_WOP,
        "Downtempo": UnifiedGenre.DOWNTEMPO,
        "Drill Rap": UnifiedGenre.DRILL_RAP,
        "East Coast Hip Hop": UnifiedGenre.EAST_COAST_HIP_HOP,
        "Emo Punk": UnifiedGenre.EMO_PUNK,
        "Enka": UnifiedGenre.ENKA,
        "Fado": UnifiedGenre.FADO,
        "Folk Metal": UnifiedGenre.FOLK_METAL,
        "Folktronica": UnifiedGenre.FOLKTRONICA,
        "French Folk": UnifiedGenre.FRENCH_FOLK,
        "French Pop": UnifiedGenre.FRENCH_POP,
        "German Pop": UnifiedGenre.GERMAN_POP,
        "Glam Metal（Hair Metal，Pop Metal）": UnifiedGenre.GLAM_METAL,
        "Glam Metal\n（Hair Metal，Pop Metal）": UnifiedGenre.GLAM_METAL,
        "Glitch": UnifiedGenre.GLITCH,
        "Gothic Metal": UnifiedGenre.GOTHIC_METAL,
        "Grindcore ": UnifiedGenre.GRINDCORE,
        "Grunge Rock": UnifiedGenre.GRUNGE_ROCK,
        "Gypsy Jazz": UnifiedGenre.GYPSY_JAZZ,
        "Hard Bop": UnifiedGenre.HARD_BOP,
        "Hardcore": UnifiedGenre.HARDCORE,
        "Hardcore Punk ": UnifiedGenre.HARDCORE_PUNK,
        "Hi-NRG": UnifiedGenre.HINRG,
        "Hip Hop": UnifiedGenre.HIP_HOP,
        "IDM": UnifiedGenre.IDM,
        "Impressionism": UnifiedGenre.IMPRESSIONISM,
        "Industrial Metal": UnifiedGenre.INDUSTRIAL_METAL,
        "Irish Folk": UnifiedGenre.IRISH_FOLK,
        "Japanese Traditional Music": UnifiedGenre.JANPANESE_TRADITIONAL_MUSIC,
        "Korean Folk": UnifiedGenre.KOREAN_FOLK,
        "Korea Trot": UnifiedGenre.KOREA_TROT,
        "K-Pop": UnifiedGenre.KPOP,
        "Latin Rock": UnifiedGenre.LATIN_ROCK,
        "Lo-Fi House": UnifiedGenre.LO_FI_HOUSE,
        "Lo-Fi Rock": UnifiedGenre.LO_FI_ROCK,
        "Metalcore": UnifiedGenre.METALCORE,
        "Midwest Hip Hop": UnifiedGenre.MIDWEST_HIP_HOP,
        "Modernism": UnifiedGenre.MODERNISM,
        "Mongolian Folk Songs": UnifiedGenre.MONGOLIAN_FOLK_SONGS,
        "Neo Funk": UnifiedGenre.NEO_FUNK,
        "New Wave ": UnifiedGenre.NEW_WAVE,
        "No Wave": UnifiedGenre.NO_WAVE,
        "Nu Metal": UnifiedGenre.NU_METAL,
        "Post-hardcore": UnifiedGenre.POST_HARDCORE,
        "Post-Punk": UnifiedGenre.POST_PUNK,
        "Progressive Metal（Prog Metal)": UnifiedGenre.PROGRESSIVE_METAL,
        "Progressive R&B": UnifiedGenre.PROGRESSIVE_RNB,
        "Ragtime": UnifiedGenre.RAGTIME,
        "Rap Metal": UnifiedGenre.RAP_METAL,
        "Romantic music": UnifiedGenre.ROMANTIC_MUSIC,
        "Rumba": UnifiedGenre.RUMBA,
        "Score": UnifiedGenre.SCORE,
        "Shima Uta": UnifiedGenre.SHIMA_UTA,
        "Shoegaze-Rock": UnifiedGenre.SHOEGAZE_ROCK,
        "Ska Punk": UnifiedGenre.SKA_PUNK,
        "Son cubano": UnifiedGenre.SON_CUBANO,
        "Southern Hip Hop": UnifiedGenre.SOUTHERN_HIP_HOP,
        "Synthwave": UnifiedGenre.SYNTHWAVE,
        "Traditional Blues": UnifiedGenre.TRADITIONAL_BLUES,
        "Turkish Pop": UnifiedGenre.TURKISH_POP,
        "Uygur Folk Songs": UnifiedGenre.UYGUR_FOLK_SONGS,
        "Worldbeat": UnifiedGenre.WORLDBEAT,
        "Yodeling": UnifiedGenre.YODELING,
        "Red Song/": UnifiedGenre.RED_SONG,
        "Red Song": UnifiedGenre.RED_SONG,
    },
    UnifiedCategory.MOOD: {
        "Happy": UnifiedMood.HAPPY,
        "Cute/Playful": UnifiedMood.CUTE_PLAYFUL,
        "Excited": UnifiedMood.EXCITED,
        "Funny": UnifiedMood.FUNNY,
        "Inspirational/Hopeful": UnifiedMood.INSPIRATIONAL_HOPEFUL,
        "Sorrow/Sad": UnifiedMood.SORROW_SAD,
        "Sentimental/Melancholic/Lonely": UnifiedMood.SENTIMENTAL_MELANCHOLIC_LONELY,
        "Weird": UnifiedMood.WEIRD,
        "Thrilling/Suspenseful/Tense": UnifiedMood.THRILLING_SUSPENSEFUL_TENSE,
        "Angry/Aggressive": UnifiedMood.ANGRY_AGGRESSIVE,
        "Groovy/Funky": UnifiedMood.GROOVY_FUNKY,
        "Dynamic/Energetic": UnifiedMood.DYNAMIC_ENERGETIC,
        "Romantic": UnifiedMood.ROMANTIC,
        "Nostalgic/Memory": UnifiedMood.NOSTALGIC_MEMORY,
        "Dreamy/Ethereal": UnifiedMood.DREAMY_ETHEREAL,
        "Healing": UnifiedMood.HEALING,
        "Miss": UnifiedMood.MISS,
        "Chill": UnifiedMood.CHILL,
        "Calm/Relaxing": UnifiedMood.CALM_RELAXING,
        "Shocking/magnificent/epic": UnifiedMood.SHOCKING_MAGNIFICENT_EPIC,
        "Mysterious": UnifiedMood.MYSTERIOUS,
        "No Mood": UnifiedMood.EMPTY,
        "Other mood": UnifiedMood.EMPTY
    },
    UnifiedCategory.THEME: {
        "Halloween": UnifiedTheme.HALLOWEEN,
        "Christmas": UnifiedTheme.CHRISTMAS,
        "New Year": UnifiedTheme.NEW_YEAR,
        "Spring Festival": UnifiedTheme.SPRING_FESTIVAL,
        "Valentine's day": UnifiedTheme.VALENTINES_DAY,
        "Qi Xi": UnifiedTheme.QI_XI,
        "Birthday": UnifiedTheme.BIRTHDAY,
        "Wedding": UnifiedTheme.WEDDING,
        "Funeral": UnifiedTheme.FUNERAL,
        "Graduation": UnifiedTheme.GRADUATION,
        "National's Day": UnifiedTheme.NATIONALS_DAY,
        "National's Day ": UnifiedTheme.NATIONALS_DAY,
        "Vlog/DailyLife": UnifiedTheme.VLOG_DAILYLIFE,
        "Food": UnifiedTheme.FOOD,
        "Pet/Animals": UnifiedTheme.PET_ANIMALS,
        "Beauty/Fashion": UnifiedTheme.BEAUTY_FASHION,
        "Entertainment": UnifiedTheme.ENTERTAINMENT,
        "babies": UnifiedTheme.BABIES,
        "Babies": UnifiedTheme.BABIES,
        "children": UnifiedTheme.CHILDREN,
        "Children": UnifiedTheme.CHILDREN,
        "Transition": UnifiedTheme.TRANSITION,
        "Transition (卡点)": UnifiedTheme.TRANSITION,
        "Anime": UnifiedTheme.ANIME,
        "Wake up": UnifiedTheme.WAKE_UP,
        "Family time": UnifiedTheme.FAMILY_TIME,
        "landscape/scenery": UnifiedTheme.LANDSCAPE_SCENERY,
        "Landscape/Scenery": UnifiedTheme.LANDSCAPE_SCENERY,
        "Prank": UnifiedTheme.PRANK,
        "Timelapse": UnifiedTheme.TIMELAPSE,
        "Rainy Day": UnifiedTheme.RAINY_DAY,
        "Sunny Day": UnifiedTheme.SUNNY_DAY,
        "Spring": UnifiedTheme.SPRING,
        "Summer": UnifiedTheme.SUMMER,
        "Autumn": UnifiedTheme.AUTUMN,
        "Winter": UnifiedTheme.WINTER,
        "Evening": UnifiedTheme.EVENING,
        "Morning": UnifiedTheme.MORNING,
        "Beach": UnifiedTheme.BEACH,
        "Nightclub": UnifiedTheme.NIGHTCLUB,
        "Coffee Shop": UnifiedTheme.COFFEE_SHOP,
        "Restaurants": UnifiedTheme.RESTAURANTS,
        "Lounge": UnifiedTheme.LOUNGE,
        "Campus": UnifiedTheme.CAMPUS,
        "Park": UnifiedTheme.PARK,
        "Marketplace": UnifiedTheme.MARKETPLACE,
        "Universe": UnifiedTheme.UNIVERSE,
        "Bar": UnifiedTheme.BAR,
        "Theater/Concert hall": UnifiedTheme.THEATER_CONCERT_HALL,
        "Theater / Concert hall": UnifiedTheme.THEATER_CONCERT_HALL,
        "Sport": UnifiedTheme.SPORT,
        "Dance": UnifiedTheme.DANCE,
        "Game": UnifiedTheme.GAME,
        "Travel": UnifiedTheme.TRAVEL,
        "Focus": UnifiedTheme.FOCUS,
        "Party": UnifiedTheme.PARTY,
        "Commute": UnifiedTheme.COMMUTE,
        "Roadtrip": UnifiedTheme.ROADTRIP,
        "Meditation": UnifiedTheme.MEDITATION,
        "Relaxation": UnifiedTheme.RELAXATION,
        "Cleaning and Chores": UnifiedTheme.CLEANING_AND_CHORES,
        "Running": UnifiedTheme.RUNNING,
        "Date": UnifiedTheme.DATE,
        "Dating": UnifiedTheme.DATE,
        "Danceable": UnifiedTheme.DANCEABLE,
        "Flirt": UnifiedTheme.FLIRT,
        "Bedtime": UnifiedTheme.BED_TIME,
        "Broke up": UnifiedTheme.BROKE_UP,
        "Dream": UnifiedTheme.DREAM,
        "Drive": UnifiedTheme.DRIVE,
        "Friendship": UnifiedTheme.FRIENDSHIP,
        "Love": UnifiedTheme.LOVE,
        "Yoga": UnifiedTheme.YOGA,
        "Other scene": UnifiedTheme.EMPTY,
        "Mid-autumn Festival": UnifiedTheme.MID_AUTUMN_FESTIVAL,
    },
    UnifiedCategory.GENDER: {
        "Female": UnifiedGender.FEMALE,
        "Male": UnifiedGender.MALE,
        "Neutral": UnifiedGender.NEUTRAL,
        "Child": UnifiedGender.CHILD,
        "Adult": UnifiedGender.ADULT,
        "Chorus_AUDIO_GENDER": UnifiedGender.CHORUS,  # dedup
        "Chrous": UnifiedGender.CHORUS,
        "Unkonwn": UnifiedGender.EMPTY,
        # audio tags ver2
        "Multiple": UnifiedGender.MULTIPLE,
    },
    UnifiedCategory.TIMBRE: {
        "Warm": UnifiedTimbre.WARM,
        "Ethereal": UnifiedTimbre.ETHEREAL,
        "Husky": UnifiedTimbre.HUSKY,
        "Deep": UnifiedTimbre.DEEP,
        "Loud and sonorous": UnifiedTimbre.LOUD_AND_SONOROUS,
        "Extreme": UnifiedTimbre.EXTREME,
        "Sharp": UnifiedTimbre.SHARP,
        "Bright": UnifiedTimbre.BRIGHT,
        "Sweet_AUDIO_TIMBRE": UnifiedTimbre.SWEET,
        "Sweet": UnifiedTimbre.SWEET,
        "Powerful": UnifiedTimbre.POWERFUL,
        "Sexy/Lazy": UnifiedTimbre.SEXY_LAZY,
        "Magnetic": UnifiedTimbre.MAGNETIC,
        "Cute_AUDIO_TIMBRE": UnifiedTimbre.CUTE,
        "Cute": UnifiedTimbre.CUTE,
        "Electrified voice": UnifiedTimbre.ELECTRIFIED_VOICE,
    },
    UnifiedCategory.LANG: {
        "Cantonese": UnifiedLang.CANTONESE,
        "Chinese": UnifiedLang.CHINESE,
        "Chinese Dialects": UnifiedLang.CHINESE_DIALECT,
        "English": UnifiedLang.ENGLISH,
        # audio tags ver2
        "African": UnifiedLang.AFRICAN,
        "Arabic": UnifiedLang.ARABIC,
        "French": UnifiedLang.FRENCH,
        "German": UnifiedLang.GERMAN,
        "Hindi": UnifiedLang.HINDI,
        "Instrumental/Non-vocal": UnifiedLang.INSTRUMENTAL,
        "Italian": UnifiedLang.ITALIAN,
        "Japanese": UnifiedLang.JAPANESE,
        "Korea": UnifiedLang.KOREA,
        "Other lang": UnifiedLang.EMPTY, # dedup
        "Russian": UnifiedLang.RUSSIAN,
        "Sichuanese": UnifiedLang.SICHUANESE,
        "Taiwanese": UnifiedLang.TAIWANESE,
        "Turkish": UnifiedLang.TURKISH,
    },
    UnifiedCategory.SINKING: {
        "Sinking": UnifiedSinking.SINKING,
        "non-Sinking": UnifiedSinking.NON_SINKING,
    },
    UnifiedCategory.EXTRA: {
        # audio tags ver2
        "Tuhai_AUDIO_EXTRA": UnifiedExtra.TUHAI,
        "Soundtrack_AUDIO_EXTRA": UnifiedExtra.SOUNDTRACK,
        "Nostalgic": UnifiedExtra.NOSTALGIC,
        "Lo-fi": UnifiedExtra.LO_FI,
    }
}

CATEGORY_MAP_AUDIO_V5 = {
    UnifiedCategory.GENRE: {
        "Pop": UnifiedGenre.POP,
        "Electronic": UnifiedGenre.ELECTRONIC,
        "Chinese Style": UnifiedGenre.CHINESE_STYLE,
        "Rock": UnifiedGenre.ROCK,
        "Jazz": UnifiedGenre.JAZZ,
        "Hip Hop/Rap": UnifiedGenre.HIP_HOP,
        "Classical": UnifiedGenre.CLASSICAL,
        "R&B/Soul": UnifiedGenre.RNB_SOUL,
        "Metal": UnifiedGenre.METAL,
        "Tuhai": UnifiedGenre.TUHAI,
        "Childhood": UnifiedGenre.CHILDHOOD,
        "Devotional": UnifiedGenre.DEVOTIONAL,
        "Chinese Tradition": UnifiedGenre.CHINESE_TRADITION,
        "Easy Listening": UnifiedGenre.EASY_LISTENING,
        "New Age": UnifiedGenre.NEW_AGE,
        "8 Bit": UnifiedGenre.EIGHT_BIT,
        "Folk": UnifiedGenre.FOLK,
        "Latin": UnifiedGenre.LATIN,
        "Alternative/Indie": UnifiedGenre.ALTERNATIVE_INDIE,
        "Epic": UnifiedGenre.EPIC,
        "BGM": UnifiedGenre.BGM,
        "Chinese Pop": UnifiedGenre.CHINESE_POP,
        "Cantopop": UnifiedGenre.CANTOPOP,
        "Taiwanese Pop": UnifiedGenre.TAIWANESE_POP,
        "Pop Folk": UnifiedGenre.POP_FOLK,
        "Contemporary Pop": UnifiedGenre.CONTEMPORARY_POP,
        "Teen Pop": UnifiedGenre.TEEN_POP,
        "Indie Pop": UnifiedGenre.INDIE_POP,
        "Dream Pop": UnifiedGenre.DREAM_POP,
        "City Pop": UnifiedGenre.CITY_POP,
        "Synth Pop": UnifiedGenre.SYNTH_POP,
        "Celtic Pop": UnifiedGenre.CELTIC_POP,
        "Dance Pop": UnifiedGenre.DANCE_POP,
        "Deep Dance Pop": UnifiedGenre.DEEP_DANCE_POP,
        "Electropop": UnifiedGenre.ELECTROPOP,
        "Chamber Pop": UnifiedGenre.CHAMBER_POP,
        "A cappella": UnifiedGenre.A_CAPPELLA,
        "Country Pop": UnifiedGenre.COUNTRY_POP,
        "EDM": UnifiedGenre.EDM,
        "House": UnifiedGenre.HOUSE,
        "Dubstep": UnifiedGenre.DUBSTEP,
        "Future Bass": UnifiedGenre.FUTURE_BASS,
        "Chillout": UnifiedGenre.CHILLOUT,
        "Trance": UnifiedGenre.TRANCE,
        "Techno": UnifiedGenre.TECHNO,
        "Drum&Bass": UnifiedGenre.DRUMNBASS,
        "Tropical House": UnifiedGenre.TROPICAL_HOUSE,
        "Disco": UnifiedGenre.DISCO,
        "Vaporwave": UnifiedGenre.VAPORWAVE,
        "Trip Hop": UnifiedGenre.TRIP_HOP,
        "Ambient": UnifiedGenre.AMBIENT,
        "Deep Pop Edm": UnifiedGenre.DEEP_POP_EDM,
        "EDM Trap": UnifiedGenre.EDM_TRAP,
        "Future House": UnifiedGenre.FUTURE_HOUSE,
        "China-Wave": UnifiedGenre.CHINA_WAVE,
        "ChinaWave": UnifiedGenre.CHINA_WAVE,
        "GuFeng Music": UnifiedGenre.GUFENG_MUSIC,
        "Chinoiserie Rap": UnifiedGenre.CHINOISERIE_RAP,
        "Chinoiserie Electronic": UnifiedGenre.CHINOISERIE_ELECTRONIC,
        "Hard Rock": UnifiedGenre.HARD_ROCK,
        "Psychedelic Rock": UnifiedGenre.PSYCHEDELIC_ROCK,
        "Pop Rock": UnifiedGenre.POP_ROCK,
        "Instrumental Rock": UnifiedGenre.INSTRUMENTAL_ROCK,
        "Alternative Rock": UnifiedGenre.ALTERNATIVE_ROCK,
        "Indie Rock": UnifiedGenre.INDIE_ROCK,
        "Post-Rock": UnifiedGenre.POST_ROCK,
        "Lo-Fi": UnifiedGenre.LO_FI,
        "J Rock": UnifiedGenre.J_ROCK,
        "Shoegazing": UnifiedGenre.SHOEGAZING,
        "Math Rock": UnifiedGenre.MATH_ROCK,
        "Surf Rock": UnifiedGenre.SURF_ROCK,
        "Progressive Rock": UnifiedGenre.PROGRESSIVE_ROCK,
        "Soft Rock": UnifiedGenre.SOFT_ROCK,
        "Jazz Pop": UnifiedGenre.JAZZ_POP,
        "Jazz Fusion": UnifiedGenre.JAZZ_FUSION,
        "Bossa Nova": UnifiedGenre.BOSSA_NOVA,
        "Avant-Garde Jazz": UnifiedGenre.AVANT_GARDE_JAZZ,
        "Swing": UnifiedGenre.SWING,
        "Big Band": UnifiedGenre.BIG_BAND,
        "Bop": UnifiedGenre.BOP,
        "Post-Bop": UnifiedGenre.POST_BOP,
        "Smooth Jazz": UnifiedGenre.SMOOTH_JAZZ,
        "Cool Jazz": UnifiedGenre.COOL_JAZZ,
        "Vocal Jazz": UnifiedGenre.VOCAL_JAZZ,
        "Nu Jazz": UnifiedGenre.NU_JAZZ,
        "Free Jazz": UnifiedGenre.FREE_JAZZ,
        "Trap Rap": UnifiedGenre.TRAP_RAP,
        "Old School": UnifiedGenre.OLD_SCHOOL,
        "R&B Rap": UnifiedGenre.RNB_RAP,
        "Jazz Hip Hop": UnifiedGenre.JAZZ_HIP_HOP,
        "Alternative Hip Hop": UnifiedGenre.ALTERNATIVE_HIP_HOP,
        "Instrumental Hip Hop": UnifiedGenre.INSTRUMENTAL_HIP_HOP,
        "Pop Rap": UnifiedGenre.POP_RAP,
        "Hardcore Rap": UnifiedGenre.HARDCORE_RAP,
        "Comedy Hip Hop": UnifiedGenre.COMEDY_HIP_HOP,
        "Hip House": UnifiedGenre.HIP_HOUSE,
        "Chill Beats": UnifiedGenre.CHILL_BEATS,
        "Chorus_AUDIO_GENRE": UnifiedGenre.CHORUS,
        "Chamber Music": UnifiedGenre.CHAMBER_MUSIC,
        "Symphony": UnifiedGenre.SYMPHONY,
        "Funk": UnifiedGenre.FUNK,
        "Contemporary R&B": UnifiedGenre.CONTEMPORARY_RNB,
        "Neo Soul": UnifiedGenre.NEO_SOUL,
        "Soul": UnifiedGenre.SOUL,
        "Pop Soul": UnifiedGenre.POP_SOUL,
        "Black Metal": UnifiedGenre.BLACK_METAL,
        "Death Metal": UnifiedGenre.DEATH_METAL,
        "Glam Metal": UnifiedGenre.GLAM_METAL,
        "Grindcore": UnifiedGenre.GRINDCORE,
        "Power Metal": UnifiedGenre.POWER_METAL,
        "Progressive Metal": UnifiedGenre.PROGRESSIVE_METAL,
        "Progressive MetalProg Metal": UnifiedGenre.PROGRESSIVE_METAL,
        "Speed Metal": UnifiedGenre.SPEED_METAL,
        "DJ": UnifiedGenre.DJ,
        "VinaHouse": UnifiedGenre.VINAHOUSE,
        "MC": UnifiedGenre.MC,
        "Gospel": UnifiedGenre.GOSPEL,
        "Holiday Music": UnifiedGenre.HOLIDAY_MUSIC,
        "Christian Music": UnifiedGenre.CHRISTIAN_MUSIC,
        "Buddhist music": UnifiedGenre.BUDDHIST_MUSIC,
        "Chinese Opera": UnifiedGenre.CHINESE_OPERA,
        "Traditional Chinese Folk": UnifiedGenre.TRADITIONAL_CHINESE_FOLK,
        "Chinese Quyi": UnifiedGenre.CHINESE_QUYI,
        "Red Song": UnifiedGenre.RED_SONG,
        "Folk Pop": UnifiedGenre.FOLK_POP,
        "Indie Folk": UnifiedGenre.INDIE_FOLK,
        "Tango": UnifiedGenre.TANGO,
        "Reggaeton": UnifiedGenre.REGGAETON,
        "Blues": UnifiedGenre.BLUES,
        "Country": UnifiedGenre.COUNTRY_POP,  # share the same tag with "Country Pop"
        "Punk": UnifiedGenre.PUNK,
        "Reggae": UnifiedGenre.REGGAE,
        "Other genre": UnifiedGenre.EMPTY,
        "Boombap": UnifiedGenre.BOOMBAP,
        "Chillwave": UnifiedGenre.CHILLWAVE,
        "Chinese Folk": UnifiedGenre.CHINESE_FOLK,
        "Contemporary Blues": UnifiedGenre.CONTEMPORARY_BLUES,
        "Contemporary Folk": UnifiedGenre.CONTEMPORARY_FOLK,
        "Country Blues": UnifiedGenre.COUNTRY_BLUES,
        "Emo Rap": UnifiedGenre.EMO_RAP,
        "English Folk": UnifiedGenre.ENGLISH_FOLK,
        "Flamenco": UnifiedGenre.FLAMENCO,
        "Gangsta Rap": UnifiedGenre.GANGSTA_RAP,
        "Hardstyle": UnifiedGenre.HARDSTYLE,
        "Heavy Metal": UnifiedGenre.HEAVY_METAL,
        "HiNRG": UnifiedGenre.HINRG,
        "Indian Pop": UnifiedGenre.INDIAN_POP,
        "Italian Pop": UnifiedGenre.ITALIAN_POP,
        "Japanese Folk": UnifiedGenre.JAPANESE_FOLK,
        "J Pop": UnifiedGenre.J_POP,
        "KPop": UnifiedGenre.KPOP,
        "Latin Pop": UnifiedGenre.LATIN_POP,
        "Low pop": UnifiedGenre.LOW_POP,
        "Melbourne Bounce": UnifiedGenre.MELBOURNE_BOUNCE,
        "New Chinese Folk": UnifiedGenre.NEW_CHINESE_FOLK,
        "New Wave": UnifiedGenre.NEW_WAVE,
        "Nostalgic Pop": UnifiedGenre.NOSTALGIC_POP,
        "Pop Punk": UnifiedGenre.POP_PUNK,
        "Rock Blues": UnifiedGenre.ROCK_BLUES,
        "Rock BluesNew": UnifiedGenre.ROCK_BLUES,
        "Russian Pop": UnifiedGenre.RUSSIAN_POP,
        "Samba": UnifiedGenre.SAMBA,
        "Shi Dai Qu": UnifiedGenre.SHI_DAI_QU,
        "Soundtrack": UnifiedGenre.SOUNDTRACK,
        "Thrash Metal": UnifiedGenre.THRASH_METAL,
        "Traditional Folk": UnifiedGenre.TRADITIONAL_FOLK,
        "Traditional Jazz": UnifiedGenre.TRADITIONAL_JAZZ,
        "Vulgar Pop": UnifiedGenre.VULGAR_POP,
        "West Coast Hip Hop": UnifiedGenre.WEST_COAST_HIP_HOP,
        "World Music": UnifiedGenre.WORLD_MUSIC,
        "Breakbeat": UnifiedGenre.BREAKBEAT,
        "Country Folk": UnifiedGenre.COUNTRY_FOLK,
        "Country Rock": UnifiedGenre.COUNTRY_ROCK,
        "Jazz Blues": UnifiedGenre.JAZZ_BLUES,
        # audio tags ver2
        "Acapella": UnifiedGenre.ACAPELLA,
        "Acid Jazz": UnifiedGenre.ACID_JAZZ,
        "African Folk": UnifiedGenre.AFRICAN_FOLK,
        "African Pop": UnifiedGenre.AFRICAN_POP,
        "Arabic Folk": UnifiedGenre.ARABIC_FOLK,
        "Arabic Pop": UnifiedGenre.ARABIC_POP,
        "Baroque": UnifiedGenre.BAROQUE,
        "Bass House": UnifiedGenre.BASS_HOUSE,
        "Bebop": UnifiedGenre.BEHOP,
        "Bhangra": UnifiedGenre.BHANGRA,
        "Blue Grass": UnifiedGenre.BLUE_GRASS,
        "Blues Rock": UnifiedGenre.BLUES_ROCK,
        "Boogie Woogie": UnifiedGenre.BOOGIE_WOOGIE,
        "Brazilian Funk Style": UnifiedGenre.BRAZILIAN_FUNK_STYLE,
        "Brit Pop": UnifiedGenre.BRIT_POP,
        "Bubblegum Bass": UnifiedGenre.BUBBLEGUM_BASS,
        "Chinese Ballad Pop": UnifiedGenre.CHINESE_BALLAD_POP,
        "Chinese Traditional Instrumental Music": UnifiedGenre.CHINESE_TRADITIONAL_INSTRUMENTAL_MUSIC,
        "Classical period": UnifiedGenre.CLASSICAL_PERIOD,
        "Classic R&B / Soul": UnifiedGenre.CLASSIC_RNB_SOUL,
        "Contemporary classical music": UnifiedGenre.CONTEMPORARY_CLASSICAL_MUSIC,
        "Dancehall": UnifiedGenre.DANCEHALL,
        "Dance Punk": UnifiedGenre.DANCE_PUNK,
        "Dark Ambient": UnifiedGenre.DARK_AMBIENT,
        "Deathcore": UnifiedGenre.DEATHCORE,
        "Doom Metal": UnifiedGenre.DOOM_METAL,
        "Doo-Wop": UnifiedGenre.DOO_WOP,
        "Downtempo": UnifiedGenre.DOWNTEMPO,
        "Drill Rap": UnifiedGenre.DRILL_RAP,
        "East Coast Hip Hop": UnifiedGenre.EAST_COAST_HIP_HOP,
        "Emo Punk": UnifiedGenre.EMO_PUNK,
        "Enka": UnifiedGenre.ENKA,
        "Fado": UnifiedGenre.FADO,
        "Folk Metal": UnifiedGenre.FOLK_METAL,
        "Folktronica": UnifiedGenre.FOLKTRONICA,
        "French Folk": UnifiedGenre.FRENCH_FOLK,
        "French Pop": UnifiedGenre.FRENCH_POP,
        "German Pop": UnifiedGenre.GERMAN_POP,
        "Glam Metal（Hair Metal，Pop Metal）": UnifiedGenre.GLAM_METAL,
        "Glam Metal\n（Hair Metal，Pop Metal）": UnifiedGenre.GLAM_METAL,
        "Glitch": UnifiedGenre.GLITCH,
        "Gothic Metal": UnifiedGenre.GOTHIC_METAL,
        "Grindcore ": UnifiedGenre.GRINDCORE,
        "Grunge Rock": UnifiedGenre.GRUNGE_ROCK,
        "Gypsy Jazz": UnifiedGenre.GYPSY_JAZZ,
        "Hard Bop": UnifiedGenre.HARD_BOP,
        "Hardcore": UnifiedGenre.HARDCORE,
        "Hardcore Punk ": UnifiedGenre.HARDCORE_PUNK,
        "Hardcore Punk": UnifiedGenre.HARDCORE_PUNK,
        "Hi-NRG": UnifiedGenre.HINRG,
        "Hip Hop": UnifiedGenre.HIP_HOP,
        "IDM": UnifiedGenre.IDM,
        "Impressionism": UnifiedGenre.IMPRESSIONISM,
        "Industrial Metal": UnifiedGenre.INDUSTRIAL_METAL,
        "Irish Folk": UnifiedGenre.IRISH_FOLK,
        "Japanese Traditional Music": UnifiedGenre.JANPANESE_TRADITIONAL_MUSIC,
        "Korean Folk": UnifiedGenre.KOREAN_FOLK,
        "Korea Trot": UnifiedGenre.KOREA_TROT,
        "K-Pop": UnifiedGenre.KPOP,
        "Latin Rock": UnifiedGenre.LATIN_ROCK,
        "Lo-Fi House": UnifiedGenre.LO_FI_HOUSE,
        "Lo-Fi Rock": UnifiedGenre.LO_FI_ROCK,
        "Metalcore": UnifiedGenre.METALCORE,
        "Midwest Hip Hop": UnifiedGenre.MIDWEST_HIP_HOP,
        "Modernism": UnifiedGenre.MODERNISM,
        "Mongolian Folk Songs": UnifiedGenre.MONGOLIAN_FOLK_SONGS,
        "Neo Funk": UnifiedGenre.NEO_FUNK,
        "New Wave ": UnifiedGenre.NEW_WAVE,
        "No Wave": UnifiedGenre.NO_WAVE,
        "Nu Metal": UnifiedGenre.NU_METAL,
        "Post-hardcore": UnifiedGenre.POST_HARDCORE,
        "Post-Punk": UnifiedGenre.POST_PUNK,
        "Progressive Metal（Prog Metal)": UnifiedGenre.PROGRESSIVE_METAL,
        "Progressive R&B": UnifiedGenre.PROGRESSIVE_RNB,
        "Ragtime": UnifiedGenre.RAGTIME,
        "Rap Metal": UnifiedGenre.RAP_METAL,
        "Romantic music": UnifiedGenre.ROMANTIC_MUSIC,
        "Rumba": UnifiedGenre.RUMBA,
        "Score": UnifiedGenre.SCORE,
        "Shima Uta": UnifiedGenre.SHIMA_UTA,
        "Shoegaze-Rock": UnifiedGenre.SHOEGAZE_ROCK,
        "Ska Punk": UnifiedGenre.SKA_PUNK,
        "Son cubano": UnifiedGenre.SON_CUBANO,
        "Southern Hip Hop": UnifiedGenre.SOUTHERN_HIP_HOP,
        "Synthwave": UnifiedGenre.SYNTHWAVE,
        "Traditional Blues": UnifiedGenre.TRADITIONAL_BLUES,
        "Turkish Pop": UnifiedGenre.TURKISH_POP,
        "Uygur Folk Songs": UnifiedGenre.UYGUR_FOLK_SONGS,
        "Worldbeat": UnifiedGenre.WORLDBEAT,
        "Yodeling": UnifiedGenre.YODELING,
        "Red Song/": UnifiedGenre.RED_SONG,
        "Red Song": UnifiedGenre.RED_SONG,
    },
    UnifiedCategory.MOOD: {
        "Happy": UnifiedMood.HAPPY,
        "Cute/Playful": UnifiedMood.CUTE_PLAYFUL,
        "Excited": UnifiedMood.EXCITED,
        "Funny": UnifiedMood.FUNNY,
        "Inspirational/Hopeful": UnifiedMood.INSPIRATIONAL_HOPEFUL,
        "Sorrow/Sad": UnifiedMood.SORROW_SAD,
        "Sentimental/Melancholic/Lonely": UnifiedMood.SENTIMENTAL_MELANCHOLIC_LONELY,
        "Weird": UnifiedMood.WEIRD,
        "Thrilling/Suspenseful/Tense": UnifiedMood.THRILLING_SUSPENSEFUL_TENSE,
        "Angry/Aggressive": UnifiedMood.ANGRY_AGGRESSIVE,
        "Groovy/Funky": UnifiedMood.GROOVY_FUNKY,
        "Dynamic/Energetic": UnifiedMood.DYNAMIC_ENERGETIC,
        "Romantic": UnifiedMood.ROMANTIC,
        "Nostalgic/Memory": UnifiedMood.NOSTALGIC_MEMORY,
        "Dreamy/Ethereal": UnifiedMood.DREAMY_ETHEREAL,
        "Healing": UnifiedMood.HEALING,
        "Miss": UnifiedMood.MISS,
        "Chill": UnifiedMood.CHILL,
        "Calm/Relaxing": UnifiedMood.CALM_RELAXING,
        "Shocking/magnificent/epic": UnifiedMood.SHOCKING_MAGNIFICENT_EPIC,
        "Mysterious": UnifiedMood.MYSTERIOUS,
        "No Mood": UnifiedMood.EMPTY,
        "Other mood": UnifiedMood.EMPTY
    },
    UnifiedCategory.THEME: {
        "Halloween": UnifiedTheme.HALLOWEEN,
        "Christmas": UnifiedTheme.CHRISTMAS,
        "New Year": UnifiedTheme.NEW_YEAR,
        "Spring Festival": UnifiedTheme.SPRING_FESTIVAL,
        "Valentine's day": UnifiedTheme.VALENTINES_DAY,
        "Qi Xi": UnifiedTheme.QI_XI,
        "Birthday": UnifiedTheme.BIRTHDAY,
        "Wedding": UnifiedTheme.WEDDING,
        "Funeral": UnifiedTheme.FUNERAL,
        "Graduation": UnifiedTheme.GRADUATION,
        "National's Day": UnifiedTheme.NATIONALS_DAY,
        "National's Day ": UnifiedTheme.NATIONALS_DAY,
        "Vlog/DailyLife": UnifiedTheme.VLOG_DAILYLIFE,
        "Food": UnifiedTheme.FOOD,
        "Pet/Animals": UnifiedTheme.PET_ANIMALS,
        "Beauty/Fashion": UnifiedTheme.BEAUTY_FASHION,
        "Entertainment": UnifiedTheme.ENTERTAINMENT,
        "babies": UnifiedTheme.BABIES,
        "Babies": UnifiedTheme.BABIES,
        "children": UnifiedTheme.CHILDREN,
        "Children": UnifiedTheme.CHILDREN,
        "Transition": UnifiedTheme.TRANSITION,
        "Transition (卡点)": UnifiedTheme.TRANSITION,
        "Anime": UnifiedTheme.ANIME,
        "Wake up": UnifiedTheme.WAKE_UP,
        "Family time": UnifiedTheme.FAMILY_TIME,
        "landscape/scenery": UnifiedTheme.LANDSCAPE_SCENERY,
        "Landscape/Scenery": UnifiedTheme.LANDSCAPE_SCENERY,
        "Prank": UnifiedTheme.PRANK,
        "Timelapse": UnifiedTheme.TIMELAPSE,
        "Rainy Day": UnifiedTheme.RAINY_DAY,
        "Sunny Day": UnifiedTheme.SUNNY_DAY,
        "Spring": UnifiedTheme.SPRING,
        "Summer": UnifiedTheme.SUMMER,
        "Autumn": UnifiedTheme.AUTUMN,
        "Winter": UnifiedTheme.WINTER,
        "Evening": UnifiedTheme.EVENING,
        "Morning": UnifiedTheme.MORNING,
        "Beach": UnifiedTheme.BEACH,
        "Nightclub": UnifiedTheme.NIGHTCLUB,
        "Coffee Shop": UnifiedTheme.COFFEE_SHOP,
        "Restaurants": UnifiedTheme.RESTAURANTS,
        "Lounge": UnifiedTheme.LOUNGE,
        "Campus": UnifiedTheme.CAMPUS,
        "Park": UnifiedTheme.PARK,
        "Marketplace": UnifiedTheme.MARKETPLACE,
        "Universe": UnifiedTheme.UNIVERSE,
        "Bar": UnifiedTheme.BAR,
        "Theater/Concert hall": UnifiedTheme.THEATER_CONCERT_HALL,
        "Theater / Concert hall": UnifiedTheme.THEATER_CONCERT_HALL,
        "Sport": UnifiedTheme.SPORT,
        "Dance": UnifiedTheme.DANCE,
        "Game": UnifiedTheme.GAME,
        "Travel": UnifiedTheme.TRAVEL,
        "Focus": UnifiedTheme.FOCUS,
        "Party": UnifiedTheme.PARTY,
        "Commute": UnifiedTheme.COMMUTE,
        "Roadtrip": UnifiedTheme.ROADTRIP,
        "Meditation": UnifiedTheme.MEDITATION,
        "Relaxation": UnifiedTheme.RELAXATION,
        "Cleaning and Chores": UnifiedTheme.CLEANING_AND_CHORES,
        "Running": UnifiedTheme.RUNNING,
        "Date": UnifiedTheme.DATE,
        "Dating": UnifiedTheme.DATE,
        "Danceable": UnifiedTheme.DANCEABLE,
        "Flirt": UnifiedTheme.FLIRT,
        "Bedtime": UnifiedTheme.BED_TIME,
        "Broke up": UnifiedTheme.BROKE_UP,
        "Dream": UnifiedTheme.DREAM,
        "Drive": UnifiedTheme.DRIVE,
        "Friendship": UnifiedTheme.FRIENDSHIP,
        "Love": UnifiedTheme.LOVE,
        "Yoga": UnifiedTheme.YOGA,
        "Other scene": UnifiedTheme.EMPTY,
        "Mid-autumn Festival": UnifiedTheme.MID_AUTUMN_FESTIVAL,
    },
    UnifiedCategory.GENDER: {
        "Female": UnifiedGender.FEMALE,
        "Male": UnifiedGender.MALE,
        "Neutral": UnifiedGender.NEUTRAL,
        "Child": UnifiedGender.CHILD,
        "Adult": UnifiedGender.ADULT,
        "Chorus_AUDIO_GENDER": UnifiedGender.CHORUS,  # dedup
        "Chrous": UnifiedGender.CHORUS,
        "Unkonwn": UnifiedGender.EMPTY,
        # audio tags ver2
        "Multiple": UnifiedGender.MULTIPLE,
    },

    UnifiedCategory.TIMBRE: {
        "Warm": UnifiedTimbre.WARM,
        "Ethereal": UnifiedTimbre.ETHEREAL,
        "Husky": UnifiedTimbre.HUSKY,
        "Deep": UnifiedTimbre.DEEP,
        "Loud and sonorous": UnifiedTimbre.LOUD_AND_SONOROUS,
        "Extreme": UnifiedTimbre.EXTREME,
        "Sharp": UnifiedTimbre.SHARP,
        "Bright": UnifiedTimbre.BRIGHT,
        "Sweet_AUDIO_TIMBRE": UnifiedTimbre.SWEET,
        "Powerful": UnifiedTimbre.POWERFUL,
        "Sexy/Lazy": UnifiedTimbre.SEXY_LAZY,
        "Magnetic": UnifiedTimbre.MAGNETIC,
        "Cute_AUDIO_TIMBRE": UnifiedTimbre.CUTE,
        "Electrified voice": UnifiedTimbre.ELECTRIFIED_VOICE,
        "9m88": UnifiedTimbre.NineM88,
        "Akini Jing": UnifiedTimbre.ZhuJingXi,
        "A-Mei Chang": UnifiedTimbre.ZhangHuiMei,
        "Andy Lau": UnifiedTimbre.LiuDeHua,
        "Angela Chang": UnifiedTimbre.ZhangShaoHan,
        "Anti-general": UnifiedTimbre.ANTI_GENERAL,
        "Anti-General": UnifiedTimbre.ANTI_GENERAL,
        "AR刘夫阳": UnifiedTimbre.AR_LiuFuYang,
        "A Si": UnifiedTimbre.ASi,
        "Aska Yang": UnifiedTimbre.YangZongWei,
        "Capper": UnifiedTimbre.CAPPER,
        "C-BLOCK": UnifiedTimbre.C_BLOCK,
        "Cheer Chen": UnifiedTimbre.ChenQiZhen,
        "CORSAK": UnifiedTimbre.HuMengZhou,
        "Corsak胡梦周": UnifiedTimbre.HuMengZhou,
        "Danny Chan": UnifiedTimbre.ChenBaiQiang,
        "David Tao": UnifiedTimbre.TaoZhe,
        "Dean Ting": UnifiedTimbre.DingShiGuang,
        "Dou Wei": UnifiedTimbre.DouWei,
        "Eason Chan": UnifiedTimbre.ChenYiXun,
        "Eric Chou": UnifiedTimbre.ERIC_ZhouXingZhe,
        "Eric周兴哲": UnifiedTimbre.ERIC_ZhouXingZhe,
        "Eve Ai": UnifiedTimbre.AiYiLiang,
        "Faye Wong": UnifiedTimbre.WangFei,
        "Fine乐团": UnifiedTimbre.FineYueTuan,
        "F.I.R.": UnifiedTimbre.FeiErYueDui,
        "Fish Leong": UnifiedTimbre.LiangJingRu,
        "GAI周延": UnifiedTimbre.GAI_ZhouYan,
        "GALI": UnifiedTimbre.GALI,
        "Gareth.T": UnifiedTimbre.GARETH_T,
        "Gary Chaw": UnifiedTimbre.CaoGe,
        "G.E.M.": UnifiedTimbre.GEM_DengZiQi,
        "G.E.M.邓紫棋": UnifiedTimbre.GEM_DengZiQi,
        "Hacken Lee": UnifiedTimbre.LiKeQin,
        "Higher Brothers": UnifiedTimbre.HIGHER_BROTHERS,
        "Hua Chen Yu": UnifiedTimbre.HuaChenYu,
        "Ice Paper": UnifiedTimbre.ICE_PAPER,
        "ICE杨长青": UnifiedTimbre.ICE_YangChangQing,
        "Jacky Cheung": UnifiedTimbre.ZhangXueYou,
        "Jam Hsiao": UnifiedTimbre.XiaoJingTeng,
        "Jane Zhang": UnifiedTimbre.ZhangLiangYing,
        "Jason Zhang": UnifiedTimbre.ZhangJie,
        "Jay Chou": UnifiedTimbre.ZhouJieLun,
        "Jeff Chang": UnifiedTimbre.ZhangXinZhe,
        "JJ Lin": UnifiedTimbre.LinJunJie,
        "Joker Xue": UnifiedTimbre.XueZhiQian,
        "Jolin Tsai": UnifiedTimbre.CaiYiLin,
        "Jony J": UnifiedTimbre.JONY_J,
        "Jordan Chan": UnifiedTimbre.ChenXiaoChun,
        "J.Sheon": UnifiedTimbre.J_SHEON,
        "Jude Chiu": UnifiedTimbre.QiuDe,
        "Kafe.Hu": UnifiedTimbre.KAFE_HU,
        "KEY.L刘聪": UnifiedTimbre.KEY_L_LiuCong,
        "Khalil Fong": UnifiedTimbre.FangDaTong,
        "kkluv": UnifiedTimbre.KKLUV,
        "KnowKnow": UnifiedTimbre.KNOWKNOW,
        "LaLa Hsu": UnifiedTimbre.XuJiaYing,
        "Lao Lang": UnifiedTimbre.LaoLang,
        "Leah Dou": UnifiedTimbre.DouJingTong,
        "Leehom Wang": UnifiedTimbre.WangLiHong,
        "Lexie Liu": UnifiedTimbre.LiuBoXin,
        "Li Jian": UnifiedTimbre.LiJian,
        "Li Yugang": UnifiedTimbre.LiYuGang,
        "Lu1": UnifiedTimbre.LU1,
        "Mao Buyi": UnifiedTimbre.MaoBuYi,
        "Masiwei": UnifiedTimbre.MaSiWei,
        "Matt Lv": UnifiedTimbre.LvYanLiang,
        "Mayday": UnifiedTimbre.WuYueTian,
        "MC Hotdog": UnifiedTimbre.MC_HOTDOG,
        "MC HotDog": UnifiedTimbre.MC_HOTDOG,
        "Nicky Lee": UnifiedTimbre.LiJiuZhe,
        "NINEONE赵馨玥": UnifiedTimbre.NINEONE_ZhaoXingYue,
        "No Party For Cao Dong": UnifiedTimbre.CaoDongMeiYouPaiDui,
        "Pu Shu": UnifiedTimbre.PuShu,
        "Re-TROS": UnifiedTimbre.ChongSuDiaoXiangDeQuanLi,
        "Ronghao Li": UnifiedTimbre.LiRongHao,
        "Sandy Lam": UnifiedTimbre.LinYiLian,
        "S.H.E": UnifiedTimbre.SHE,
        "Shi Shi": UnifiedTimbre.SunShengXi,
        "Shuang Sheng": UnifiedTimbre.ShuangSheng,
        "Silence Wang": UnifiedTimbre.WangSuLong,
        "sodagreen": UnifiedTimbre.SuDaLv,
        "Soft Lipa": UnifiedTimbre.DanBao,
        "Stefanie Sun": UnifiedTimbre.SunYanZi,
        "step.jad": UnifiedTimbre.STEP_JAD,
        "Step.Jad": UnifiedTimbre.STEP_JAD,
        "Sue": UnifiedTimbre.SuYunYing,
        "Tanya Chua": UnifiedTimbre.CaiJianYa,
        "Teresa Teng": UnifiedTimbre.DengLiJun,
        "TFBOYS": UnifiedTimbre.TFBOYS,
        "The Crane": UnifiedTimbre.THE_CRANE,
        "TIA RAY": UnifiedTimbre.YuanYaWei,
        "Tiger Hu": UnifiedTimbre.HuYanBin,
        "Tizzy T": UnifiedTimbre.TIZZY_T,
        "VaVa": UnifiedTimbre.VAVA_MaoYanQi,
        "VaVa毛衍七": UnifiedTimbre.VAVA_MaoYanQi,
        "Vinida Weng": UnifiedTimbre.WanNiDa,
        "Waa Wei": UnifiedTimbre.WeiRuXuan,
        "Walter Kwan": UnifiedTimbre.GuanHaoDe,
        "Wang Feng": UnifiedTimbre.WangFeng,
        "WeiBird": UnifiedTimbre.WeiLiAn,
        "Will Pan": UnifiedTimbre.PanWeiBo,
        "Xu Wei": UnifiedTimbre.XuWei,
        "YELLOW黃宣": UnifiedTimbre.HuangXuan,
        "YELLOW黄宣": UnifiedTimbre.HuangXuan,
        "Yitai Wang": UnifiedTimbre.WangYiTai,
        "Yoga Lin": UnifiedTimbre.LinYouJia,
        "Zhang Zhen Yue": UnifiedTimbre.ZhangZhenYue,
        "Zhou Shen": UnifiedTimbre.ZhouShen,
        "Z.TAO": UnifiedTimbre.HuangZiTao,
        "丁世光": UnifiedTimbre.DingShiGuang,
        "万妮达": UnifiedTimbre.WanNiDa,
        "万能青年旅店": UnifiedTimbre.WanNengQingNianLvDian,
        "二手玫瑰": UnifiedTimbre.ErShouMeiGui,
        "五月天": UnifiedTimbre.WuYueTian,
        "五条人": UnifiedTimbre.WuTiaoRen,
        "五條人": UnifiedTimbre.WuTiaoRen,
        "余佳运": UnifiedTimbre.YuJiaYun,
        "余佳運": UnifiedTimbre.YuJiaYun,
        "信乐团": UnifiedTimbre.XinYueTuan,
        "信樂團": UnifiedTimbre.XinYueTuan,
        "关浩德": UnifiedTimbre.GuanHaoDe,
        "刘德华": UnifiedTimbre.LiuDeHua,
        "刘柏辛Lexie": UnifiedTimbre.LiuBoXin,
        "刺猬Hedgehog": UnifiedTimbre.CiWei_HEDGEHOG,
        "华晨宇": UnifiedTimbre.HuaChenYu,
        "单依纯": UnifiedTimbre.ShanYiChun,
        "双笙": UnifiedTimbre.ShuangSheng,
        "吕彦良": UnifiedTimbre.LvYanLiang,
        "告五人": UnifiedTimbre.GaoWuRen,
        "周杰伦": UnifiedTimbre.ZhouJieLun,
        "周深": UnifiedTimbre.ZhouShen,
        "唐朝": UnifiedTimbre.TangChao,
        "單依純": UnifiedTimbre.ShanYiChun,
        "回春丹": UnifiedTimbre.HuiChunDan,
        "夏之禹": UnifiedTimbre.XiaZhiYu,
        "大波浪": UnifiedTimbre.DaBoLang,
        "好妹妹": UnifiedTimbre.HaoMeiMei,
        "孙楠": UnifiedTimbre.SunNan,
        "孙燕姿": UnifiedTimbre.SunYanZi,
        "孙盛希": UnifiedTimbre.SunShengXi,
        "孫楠": UnifiedTimbre.SunNan,
        "宋冬野": UnifiedTimbre.SongDongYe,
        "小霞": UnifiedTimbre.XiaoXia,
        "尧十三": UnifiedTimbre.YaoShiSan,
        "幼稚园杀手": UnifiedTimbre.YouZhiYuanShaShou,
        "张信哲": UnifiedTimbre.ZhangXinZhe,
        "张学友": UnifiedTimbre.ZhangXueYou,
        "张惠妹": UnifiedTimbre.ZhangHuiMei,
        "张杰": UnifiedTimbre.ZhangJie,
        "张楚": UnifiedTimbre.ZhangChu,
        "张震岳": UnifiedTimbre.ZhangZhenYue,
        "张靓颖": UnifiedTimbre.ZhangLiangYing,
        "张韶涵": UnifiedTimbre.ZhangShaoHan,
        "張楚": UnifiedTimbre.ZhangChu,
        "徐佳莹": UnifiedTimbre.XuJiaYing,
        "房东的猫": UnifiedTimbre.FangDongDeMao,
        "房東的貓": UnifiedTimbre.FangDongDeMao,
        "接个吻,开一枪": UnifiedTimbre.JieGeWenKaiYiQiang,
        "新裤子": UnifiedTimbre.XinKuZi,
        "新裤子乐队": UnifiedTimbre.XinKuZi,
        "方大同": UnifiedTimbre.FangDaTong,
        "曹格": UnifiedTimbre.CaoGe,
        "朱婧汐": UnifiedTimbre.ZhuJingXi,
        "朴树": UnifiedTimbre.PuShu,
        "李佳隆": UnifiedTimbre.LiJiaLong,
        "李健": UnifiedTimbre.LiJian,
        "李克勤": UnifiedTimbre.LiKeQin,
        "李玉刚": UnifiedTimbre.LiYuGang,
        "李玖哲": UnifiedTimbre.LiJiuZhe,
        "李荣浩": UnifiedTimbre.LiRongHao,
        "杨和苏KeyNG": UnifiedTimbre.YangHeSu,
        "杨宗纬": UnifiedTimbre.YangZongWei,
        "林俊杰": UnifiedTimbre.LinJunJie,
        "林宥嘉": UnifiedTimbre.LinYouJia,
        "林忆莲": UnifiedTimbre.LinYiLian,
        "梁静茹": UnifiedTimbre.LiangJingRu,
        "棱镜": UnifiedTimbre.LengJing,
        "楊和蘇KeyNG": UnifiedTimbre.YangHeSu,
        "毛不易": UnifiedTimbre.MaoBuYi,
        "汪峰": UnifiedTimbre.WangFeng,
        "汪苏泷": UnifiedTimbre.WangSuLong,
        "河图": UnifiedTimbre.HeTu,
        "法老": UnifiedTimbre.FaLao,
        "派克特": UnifiedTimbre.PaiKeTe,
        "潘玮柏": UnifiedTimbre.PanWeiBo,
        "王以太": UnifiedTimbre.WangYiTai,
        "王力宏": UnifiedTimbre.WangLiHong,
        "王菲": UnifiedTimbre.WangFei,
        "痛仰乐队": UnifiedTimbre.TongYangYueDui,
        "痛仰樂隊": UnifiedTimbre.TongYangYueDui,
        "窦唯": UnifiedTimbre.DouWei,
        "窦靖童": UnifiedTimbre.DouJingTong,
        "老狼": UnifiedTimbre.LaoLang,
        "胡彦斌": UnifiedTimbre.HuYanBin,
        "艾怡良": UnifiedTimbre.AiYiLiang,
        "艾热AIR": UnifiedTimbre.AiRe,
        "艾熱AIR": UnifiedTimbre.AiRe,
        "艾辰": UnifiedTimbre.AiChen,
        "花粥": UnifiedTimbre.HuaZhou,
        "苏打绿": UnifiedTimbre.SuDaLv,
        "苏运莹": UnifiedTimbre.SuYunYing,
        "草东没有派对": UnifiedTimbre.CaoDongMeiYouPaiDui,
        "萧敬腾": UnifiedTimbre.XiaoJingTeng,
        "落日飛車 Sunset Rollercoaster": UnifiedTimbre.LuoRiFeiChe,
        "落日飞车": UnifiedTimbre.LuoRiFeiChe,
        "蔡依林": UnifiedTimbre.CaiYiLin,
        "蔡健雅": UnifiedTimbre.CaiJianYa,
        "薛之谦": UnifiedTimbre.XueZhiQian,
        "蛋堡": UnifiedTimbre.DanBao,
        "袁娅维": UnifiedTimbre.YuanYaWei,
        "裘德": UnifiedTimbre.QiuDe,
        "許嵩": UnifiedTimbre.XuSong,
        "謝春花": UnifiedTimbre.XieChunHua,
        "许嵩": UnifiedTimbre.XuSong,
        "许巍": UnifiedTimbre.XuWei,
        "谢春花": UnifiedTimbre.XieChunHua,
        "赵雷": UnifiedTimbre.ZhaoLei,
        "趙雷": UnifiedTimbre.ZhaoLei,
        "邓丽君": UnifiedTimbre.DengLiJun,
        "那奇沃夫": UnifiedTimbre.NaQiWoFu,
        "郭顶": UnifiedTimbre.GuoDing,
        "重塑雕像的权利": UnifiedTimbre.ChongSuDiaoXiangDeQuanLi,
        "銀臨": UnifiedTimbre.YinLin,
        "银临": UnifiedTimbre.YinLin,
        "队长": UnifiedTimbre.DuiZhang,
        "阿肆": UnifiedTimbre.ASi,
        "陈奕迅": UnifiedTimbre.ChenYiXun,
        "陈小春": UnifiedTimbre.ChenXiaoChun,
        "陈百强": UnifiedTimbre.ChenBaiQiang,
        "陈粒": UnifiedTimbre.ChenLi,
        "陈绮贞": UnifiedTimbre.ChenQiZhen,
        "陈鸿宇": UnifiedTimbre.ChenHongYu,
        "陳粒": UnifiedTimbre.ChenLi,
        "陶喆": UnifiedTimbre.TaoZhe,
        "隊長": UnifiedTimbre.DuiZhang,
        "韦礼安": UnifiedTimbre.WeiLiAn,
        "音阙诗听": UnifiedTimbre.YinQueShiTing,
        "飞儿乐团": UnifiedTimbre.FeiErYueDui,
        "马思唯": UnifiedTimbre.MaSiWei,
        "马赛克乐队": UnifiedTimbre.MaSaiKeYueDui,
        "魏如萱": UnifiedTimbre.WeiRuXuan,
        "黃齡": UnifiedTimbre.HuangLing,
        "黄子韬": UnifiedTimbre.HuangZiTao,
        "黄龄": UnifiedTimbre.HuangLing,
    },

    UnifiedCategory.LANG: {
        "Cantonese": UnifiedLang.CANTONESE,
        "Chinese": UnifiedLang.CHINESE,
        "Chinese Dialects": UnifiedLang.CHINESE_DIALECT,
        "English": UnifiedLang.ENGLISH,
        # audio tags ver2
        "African": UnifiedLang.AFRICAN,
        "Arabic": UnifiedLang.ARABIC,
        "French": UnifiedLang.FRENCH,
        "German": UnifiedLang.GERMAN,
        "Hindi": UnifiedLang.HINDI,
        "Instrumental/Non-vocal": UnifiedLang.INSTRUMENTAL,
        "Italian": UnifiedLang.ITALIAN,
        "Japanese": UnifiedLang.JAPANESE,
        "Korea": UnifiedLang.KOREA,
        "Other lang": UnifiedLang.EMPTY, # dedup
        "Russian": UnifiedLang.RUSSIAN,
        "Sichuanese": UnifiedLang.SICHUANESE,
        "Taiwanese": UnifiedLang.TAIWANESE,
        "Turkish": UnifiedLang.TURKISH,
    },
    UnifiedCategory.SINKING: {
        "Sinking": UnifiedSinking.SINKING,
        "non-Sinking": UnifiedSinking.NON_SINKING,
    },
    UnifiedCategory.EXTRA: {
        # audio tags ver2
        "Tuhai_AUDIO_EXTRA": UnifiedExtra.TUHAI,
        "Soundtrack_AUDIO_EXTRA": UnifiedExtra.SOUNDTRACK,
        "Nostalgic": UnifiedExtra.NOSTALGIC,
        "Lo-fi": UnifiedExtra.LO_FI,
    },

    UnifiedCategory.INSTRUMENT:{
        "Wurlitzer":UnifiedInstrument.WURLITZER,
        "French Horn":UnifiedInstrument.FRENCH_HORN,
        "Bongos":UnifiedInstrument.BONGOS,
        "Double Bass":UnifiedInstrument.DOUBLE_BASS,
        "Pan Flute":UnifiedInstrument.PAN_FLUTE,
        "Lead Acoustic Guitar":UnifiedInstrument.LEAD_ACOUSTIC_GUITAR,
        "Mandolin":UnifiedInstrument.MANDOLIN,
        "talk box":UnifiedInstrument.TALK_BOX,
        "Tabla":UnifiedInstrument.TABLA,
        "Traditional Wind Instruments":UnifiedInstrument.TRADITIONAL_WIND_INSTRUMENTS,
        "String Section":UnifiedInstrument.STRING_SECTION,
        "Vibes":UnifiedInstrument.VIBES,
        "Rhythm Acoustic Guitar (Arpeggio)":UnifiedInstrument.RHYTHM_ACOUSTIC_GUITAR_ARPEGGIO,
        "Electronic Percussion":UnifiedInstrument.ELECTRONIC_PERCUSSION,
        "Synth Bass":UnifiedInstrument.SYNTH_BASS,
        "Synth Brass":UnifiedInstrument.SYNTH_BRASS,
        "Backing Vocals":UnifiedInstrument.BACKING_VOCALS,
        "Sea":UnifiedInstrument.SEA,
        "Celesta":UnifiedInstrument.CELESTA,
        "Female Lead Vocal":UnifiedInstrument.FEMALE_LEAD_VOCAL,
        "Music Box":UnifiedInstrument.MUSIC_BOX,
        "Moog + Synthesizer":UnifiedInstrument.MOOG_SYNTHESIZER,
        "Ukulele":UnifiedInstrument.UKULELE,
        "Tenor Saxophone":UnifiedInstrument.TENOR_SAXOPHONE,
        "Pedal Steel Guitar":UnifiedInstrument.PEDAL_STEEL_GUITAR,
        "Rainstick":UnifiedInstrument.RAINSTICK,
        "Synth Flute":UnifiedInstrument.SYNTH_FLUTE,
        "English Horn":UnifiedInstrument.ENGLISH_HORN,
        "Acoustic Bass Guitar":UnifiedInstrument.ACOUSTIC_BASS_GUITAR,
        "Electronic Hi-Hat":UnifiedInstrument.ELECTRONIC_HI_HAT ,
        "Finger Cymbals":UnifiedInstrument.FINGER_CYMBALS,
        "Steel Drums":UnifiedInstrument.STEEL_DRUMS,
        "Didgeridoo":UnifiedInstrument.DIDGERIDOO,
        "Drum":UnifiedInstrument.DRUM,
        "Bassoon":UnifiedInstrument.BASSOON,
        "Koto":UnifiedInstrument.KOTO,
        "Castanets":UnifiedInstrument.CASTANETS,
        "Distorted Electric Guitar":UnifiedInstrument.DISTORTED_ELECTRIC_GUITAR,
        "Shamisen":UnifiedInstrument.SHAMISEN,
        "Voice":UnifiedInstrument.VOICE,
        "Bass":UnifiedInstrument.BASS,
        "Kora":UnifiedInstrument.KORA,
        "Muted Trombone":UnifiedInstrument.MUTED_TROMBONE,
        "Piano":UnifiedInstrument.PIANO,
        "Rhythm Electric Guitar":UnifiedInstrument.RHYTHM_ELECTRIC_GUITAR,
        "Lead Electric Guitar":UnifiedInstrument.LEAD_ELECTRIC_GUITAR,
        "Pitched Percussion":UnifiedInstrument.PITCHED_PERCUSSION,
        "Tuba":UnifiedInstrument.TUBA,
        "Arr. Acoustic Guitar":UnifiedInstrument.ARR_ACOUSTIC_GUITAR,
        "CP":UnifiedInstrument.CP,
        "Surdo":UnifiedInstrument.SURDO,
        "Brass and Harmonica":UnifiedInstrument.BRASS_AND_HARMONICA,
        "Sandpaper":UnifiedInstrument.SANDPAPER,
        "Charango":UnifiedInstrument.CHARANGO,
        "Keyboard":UnifiedInstrument.KEYBOARD,
        "Fiddle":UnifiedInstrument.FIDDLE,
        "Soprano Saxophone":UnifiedInstrument.SOPRANO_SAXOPHONE,
        "Polysynth":UnifiedInstrument.POLYSYNTH,
        "vinyl":UnifiedInstrument.VINYL,
        "Sound effects":UnifiedInstrument.SOUND_EFFECTS,
        "Human Beatbox":UnifiedInstrument.HUMAN_BEATBOX,
        "Breath":UnifiedInstrument.BREATH,
        "Rhythm Acoustic Guitar":UnifiedInstrument.RHYTHM_ACOUSTIC_GUITAR,
        "Harmonica":UnifiedInstrument.HARMONICA,
        "Balalaïka":UnifiedInstrument.BALALAIKA,
        "Clarinet":UnifiedInstrument.CLARINET,
        "Timpani":UnifiedInstrument.TIMPANI,
        "Xylophone":UnifiedInstrument.XYLOPHONE,
        "Clap":UnifiedInstrument.CLAP,
        "Toms":UnifiedInstrument.TOMS,
        "Sample":UnifiedInstrument.SAMPLE,
        "Guthbuinne":UnifiedInstrument.GUTHBUINNE,
        "Metal bars":UnifiedInstrument.METAL_BARS,
        "Other Guitar":UnifiedInstrument.OTHER_GUITAR,
        "Snap (Fingers)":UnifiedInstrument.SNAP_FINGERS,
        "Cavaquinho":UnifiedInstrument.CAVAQUINHO,
        "Guitar":UnifiedInstrument.GUITAR,
        "Maracas":UnifiedInstrument.MARACAS,
        "Tambourine":UnifiedInstrument.TAMBOURINE,
        "Toy Piano":UnifiedInstrument.TOY_PIANO,
        "Lap Steel Guitar":UnifiedInstrument.LAP_STEEL_GUITAR,
        "Male Lead Vocal":UnifiedInstrument.MALE_LEAD_VOCAL,
        "Hammond":UnifiedInstrument.HAMMOND,
        "Whistl":UnifiedInstrument.WHISTL,
        "Dulcimer":UnifiedInstrument.DULCIMER,
        "Orchestral Percussion":UnifiedInstrument.ORCHESTRAL_PERCUSSION,
        "Spoons":UnifiedInstrument.SPOONS,
        "Gong":UnifiedInstrument.GONG,
        "Synth Pad":UnifiedInstrument.SYNTH_PAD,
        "Electronic Cymbals":UnifiedInstrument.ELECTRONIC_CYMBALS,
        "Baritone Saxophone":UnifiedInstrument.BARITONE_SAXOPHONE,
        "Male Backing Vocals":UnifiedInstrument.MALE_BACKING_VOCALS,
        "Marimba":UnifiedInstrument.MARIMBA,
        "Rhythm Electric Guitar (Arpeggio)":UnifiedInstrument.RHYTHM_ELECTRIC_GUITAR_ARPEGGIO,
        "Acoustic Drum Kit":UnifiedInstrument.ACOUSTIC_DRUM_KIT,
        "Saxophone":UnifiedInstrument.SAXOPHONE,
        "Djembe":UnifiedInstrument.DJEMBE,
        "Lead Vocal":UnifiedInstrument.LEAD_VOCAL,
        "Snare Drum":UnifiedInstrument.SNARE_DRUM,
        "Claves_Woodblock":UnifiedInstrument.CLAVES_WOODBLOCK,
        "Wind Chimes":UnifiedInstrument.WIND_CHIMES,
        "Musical Saw":UnifiedInstrument.MUSICAL_SAW,
        "Latin Percussion":UnifiedInstrument.LATIN_PERCUSSION,
        "loop":UnifiedInstrument.LOOP,
        "Wooden Bars":UnifiedInstrument.WOODEN_BARS,
        "Cowbell":UnifiedInstrument.COWBELL,
        "Siren":UnifiedInstrument.SIREN,
        "Organ":UnifiedInstrument.ORGAN,
        "Brass Instruments":UnifiedInstrument.BRASS_INSTRUMENTS,
        "Shaker":UnifiedInstrument.SHAKER,
        "Congas":UnifiedInstrument.CONGAS,
        "Orchestra":UnifiedInstrument.ORCHESTRA,
        "Electronic Drum Kit":UnifiedInstrument.ELECTRONIC_DRUM_KIT,
        "Electric Bass":UnifiedInstrument.ELECTRIC_BASS,
        "Shehnai":UnifiedInstrument.SHEHNAI,
        "Trumpet":UnifiedInstrument.TRUMPET,
        "Cymbals":UnifiedInstrument.CYMBALS,
        "Synth Strings":UnifiedInstrument.SYNTH_STRINGS,
        "Flute":UnifiedInstrument.FLUTE,
        "Scratch":UnifiedInstrument.SCRATCH,
        "Digital Piano":UnifiedInstrument.DIGITAL_PIANO,
        "Clavinet":UnifiedInstrument.CLAVINET,
        "Agogô":UnifiedInstrument.AGOGO,
        "Theremin":UnifiedInstrument.THEREMIN,
        "Tambora":UnifiedInstrument.TAMBORA,
        "Tamboura":UnifiedInstrument.TAMBOURA,
        "Glockenspiel":UnifiedInstrument.GLOCKENSPIEL,
        "Noise effects":UnifiedInstrument.NOISE_EFFECTS,
        "Organ and Accordion":UnifiedInstrument.ORGAN_AND_ACCORDION,
        "Harmonium":UnifiedInstrument.HARMONIUM,
        "Electric Piano":UnifiedInstrument.ELECTRIC_PIANO,
        "Acoustic Guitar":UnifiedInstrument.ACOUSTIC_GUITAR,
        "Violin":UnifiedInstrument.VIOLIN,
        "Orchestra Hit":UnifiedInstrument.ORCHESTRA_HIT,
        "Rhodes":UnifiedInstrument.RHODES,
        "Wind":UnifiedInstrument.WIND,
        "Alto Saxophone":UnifiedInstrument.ALTO_SAXOPHONE,
        "Kazoo":UnifiedInstrument.KAZOO,
        "Hand Clap":UnifiedInstrument.HAND_CLAP,
        "Cajón":UnifiedInstrument.CAJON,
        "None":UnifiedInstrument.NONE,
        "Lute":UnifiedInstrument.LUTE,
        "Sitar":UnifiedInstrument.SITAR,
        "Explosion":UnifiedInstrument.EXPLOSION,
        "Guitar Synth":UnifiedInstrument.GUITAR_SYNTH,
        "Ethnic Percussion":UnifiedInstrument.ETHNIC_PERCUSSION,
        "Muted Trumpet":UnifiedInstrument.MUTED_TRUMPET,
        "Arpegiator":UnifiedInstrument.ARPEGIATOR,
        "Rototom":UnifiedInstrument.ROTOTOM,
        "FX":UnifiedInstrument.FX,
        "Triangle":UnifiedInstrument.TRIANGLE,
        "Fretless Bass":UnifiedInstrument.FRETLESS_BASS,
        "Thumb Piano":UnifiedInstrument.THUMB_PIANO,
        "Arr. Electric Guitar":UnifiedInstrument.ARR_ELECTRIC_GUITAR,
        "Hi-Hat":UnifiedInstrument.HI_HAT,
        "Bouzouki":UnifiedInstrument.BOUZOUKI,
        "Appalachian Dulcimer":UnifiedInstrument.APPALACHIAN_DULCIMER,
        "Piccolo":UnifiedInstrument.PICCOLO,
        "Sarod":UnifiedInstrument.SAROD,
        "Bass Drum":UnifiedInstrument.BASS_DRUM,
        "Drum Kit":UnifiedInstrument.DRUM_KIT,
        "Mixed Percussion":UnifiedInstrument.MIXED_PERCUSSION,
        "Washboard_Güiro":UnifiedInstrument.WASHBOARD_GUIRO,
        "Cabasa":UnifiedInstrument.CABASA,
        "Percussion":UnifiedInstrument.PERCUSSION,
        "Electronic Bass Drum":UnifiedInstrument.ELECTRONIC_BASS_DRUM,
        "Female Backing Vocals":UnifiedInstrument.FEMALE_BACKING_VOCALS,
        "Synth Ambiant":UnifiedInstrument.SYNTH_AMBIANT,
        "Mellotron":UnifiedInstrument.MELLOTRON,
        "Jews Harp":UnifiedInstrument.JEWS_HARP,
        "Sound Effects":UnifiedInstrument.SOUND_EFFECTS,
        "Crowd":UnifiedInstrument.CROWD,
        "Jingle Bells":UnifiedInstrument.JINGLE_BELLS,
        "Traditional Stringed Instruments":UnifiedInstrument.TRADITIONAL_STRINGED_INSTRUMENTS,
        "Trombone":UnifiedInstrument.TROMBONE,
        "Electronic Toms":UnifiedInstrument.ELECTRONIC_TOMS,
        "Irish Flute":UnifiedInstrument.IRISH_FLUTE,
        "Accordion":UnifiedInstrument.ACCORDION,
        "Pandeiro":UnifiedInstrument.PANDEIRO,
        "Bells":UnifiedInstrument.BELLS,
        "Sonar":UnifiedInstrument.SONAR,
        "Synth Voice":UnifiedInstrument.SYNTH_VOICE,
        "Drums and Percussion":UnifiedInstrument.DRUMS_AND_PERCUSSION,
        "Moog":UnifiedInstrument.MOOG,
        "Vibraslap":UnifiedInstrument.VIBRASLAP,
        "Tubular Bell":UnifiedInstrument.TUBULAR_BELL,
        "Calliope":UnifiedInstrument.CALLIOPE,
        "Vocal Bassist":UnifiedInstrument.VOCAL_BASSIST,
        "clock":UnifiedInstrument.CLOCK,
        "Cello":UnifiedInstrument.CELLO,
        "Synthesizer":UnifiedInstrument.SYNTHESIZER,
        "Birds":UnifiedInstrument.BIRDS,
        "Baritone Guitar":UnifiedInstrument.BARITONE_GUITAR,
        "Synth Keys":UnifiedInstrument.SYNTH_KEYS,
        "Oboe":UnifiedInstrument.OBOE,
        "Upright Bass":UnifiedInstrument.UPRIGHT_BASS,
        "Synth Lead":UnifiedInstrument.SYNTH_LEAD,
        "Shakuhachi":UnifiedInstrument.SHAKUHACHI,
        "filter":UnifiedInstrument.FILTER,
        "Vocoder":UnifiedInstrument.VOCODER,
        "Taiko":UnifiedInstrument.TAIKO,
        "Tap dance":UnifiedInstrument.TAP_DANCE,
        "Banjo":UnifiedInstrument.BANJO,
        "Brass section":UnifiedInstrument.BRASS_SECTION,
        "Viola":UnifiedInstrument.VIOLA,
        "Sub":UnifiedInstrument.SUB,
        "Dobro":UnifiedInstrument.DOBRO,
        "Harp":UnifiedInstrument.HARP,
        "Melodica":UnifiedInstrument.MELODICA,
        "Bagpipes":UnifiedInstrument.BAGPIPES,
        "Ocarina":UnifiedInstrument.OCARINA,
        "Bowl":UnifiedInstrument.BOWL,
        "Woodwinds":UnifiedInstrument.WOODWINDS,
        "Harpsichord":UnifiedInstrument.HARPSICHORD,
        "Electronic Snare Drum":UnifiedInstrument.ELECTRONIC_SNARE_DRUM,
        "Body Percussion":UnifiedInstrument.BODY_PERCUSSION,
        "Electric Guitar":UnifiedInstrument.ELECTRIC_GUITAR,
        "Flugelhorn":UnifiedInstrument.FLUGELHORN,
        "Zither":UnifiedInstrument.ZITHER,
        "Traditional Flute":UnifiedInstrument.TRADITIONAL_FLUTE,
    },


    UnifiedCategory.AUDIO_QUALITY:{
        "other audio_quality": UnifiedAudioQuality.EMPTY,
        "yes audio_quality": UnifiedAudioQuality.HIGH_QUALITY,
        "no audio_quality": UnifiedAudioQuality.LOW_QUALITY,
    },

    UnifiedCategory.POPULARITY:{
        "other popularity": UnifiedPopularity.EMPTY,
        "yes popularity": UnifiedPopularity.HIGH_POPULARITY,
        "no popularity": UnifiedPopularity.LOW_POPULARITY
    }
}

VOCAB2ID_AUDIO_V4 = Vocab2Id(
    CATEGORY_MAP_AUDIO_V4,
    UNIFIED_VOCAB2ID_LEGACY,
)

# =========================================================

# NOTE: The assumption is that in the same category, tags in different
# vocab type (SA or Audio) that have the same name are mapped to the same
# unified value. Otherwise it will be overwritten by the Audio tag mapping.
# Different categories should NOT have tags with the same name.
def _mix_map(maps: List[Dict[str, UnifiedVocab]]):
    """Merging multiple mappings. Raise an error if there is any string being mapped to different unified vocab values."""
    unified_map = {}
    for _map in maps:
        for k, v in _map.items():
            if k in unified_map:
                # Same tag name must be mapped to the same unified vocab value
                if unified_map[k] != v:
                    raise ZhVocabError(f"Key {k} has two unmatched values {unified_map[k]} and {v}")
            else:
                unified_map[k] = v
    return unified_map

# =========================================================

# SA + Audio V0
CATEGORY_MAP_MIX_V0 = {
    category: _mix_map([
        CATEGORY_MAP_SA[category] if category in CATEGORY_MAP_SA else {},
        CATEGORY_MAP_AUDIO_V0[category] if category in CATEGORY_MAP_AUDIO_V0 else {},
    ])
    for category in set(list(CATEGORY_MAP_SA.keys()) + list(CATEGORY_MAP_AUDIO_V0.keys()))
}

VOCAB2ID_MIX_V0 = Vocab2Id(
    CATEGORY_MAP_MIX_V0,
    UNIFIED_VOCAB2ID_LEGACY,
)

# =========================================================

# SA + Audio V0, V1, V2 (each version is an expansion of the previous one)
CATEGORY_MAP_MIX_V1 = {
    category: _mix_map([
        CATEGORY_MAP_SA[category] if category in CATEGORY_MAP_SA else {},
        CATEGORY_MAP_AUDIO_V2[category] if category in CATEGORY_MAP_AUDIO_V2 else {},
    ])
    for category in set(list(CATEGORY_MAP_SA.keys()) + list(CATEGORY_MAP_AUDIO_V2.keys()))
}

VOCAB2ID_MIX_V1 = Vocab2Id(
    CATEGORY_MAP_MIX_V1,
    UNIFIED_VOCAB2ID_V1,
)

# SA + Audio V0, V1, V2, V3 (each version is an expansion of the previous one)
CATEGORY_MAP_MIX_V2 = {
    category: _mix_map([
        CATEGORY_MAP_SA[category] if category in CATEGORY_MAP_SA else {},
        CATEGORY_MAP_AUDIO_V3[category] if category in CATEGORY_MAP_AUDIO_V3 else {},
    ])
    for category in set(list(CATEGORY_MAP_SA.keys()) + list(CATEGORY_MAP_AUDIO_V3.keys()))
}

VOCAB2ID_MIX_V2 = Vocab2Id(
    CATEGORY_MAP_MIX_V2,
    UNIFIED_VOCAB2ID_V2,
)

# SA + Audio V0, V1, V2, V3, V4 (each version is an expansion of the previous one)
CATEGORY_MAP_MIX_V3 = {
    category: _mix_map([
        CATEGORY_MAP_SA[category] if category in CATEGORY_MAP_SA else {},
        CATEGORY_MAP_AUDIO_V4[category] if category in CATEGORY_MAP_AUDIO_V4 else {},
    ])
    for category in set(list(CATEGORY_MAP_SA.keys()) + list(CATEGORY_MAP_AUDIO_V4.keys()))
}

CATEGORY_MAP_MIX_V4 = {
    category: _mix_map([
        CATEGORY_MAP_SA[category] if category in CATEGORY_MAP_SA else {},
        CATEGORY_MAP_AUDIO_V5[category] if category in CATEGORY_MAP_AUDIO_V5 else {},
    ])
    for category in set(list(CATEGORY_MAP_SA.keys()) + list(CATEGORY_MAP_AUDIO_V5.keys()))
}


VOCAB2ID_MIX_V3 = Vocab2Id(
    CATEGORY_MAP_MIX_V3,
    UNIFIED_VOCAB2ID_V3,
)


VOCAB2ID_MIX_V4 = Vocab2Id(
    CATEGORY_MAP_MIX_V4,
    UNIFIED_VOCAB2ID_V4,
)


# =========================================================
# Compatibility Check
# Make sure the following vocab2id is compatible with SA vocab2id
# =========================================================

def _sa_vocab_compatibility_check():
    _VOCAB2IDS_COMPAT_CHECK = {
        "mix_v0": VOCAB2ID_MIX_V0,
        "mix_v1": VOCAB2ID_MIX_V1,
    }
    sa_vocab2id = VOCAB2ID_SA.to_dict()
    for name, vocab2id in _VOCAB2IDS_COMPAT_CHECK.items():
        vocab2id_dict = vocab2id.to_dict()
        for k, v in sa_vocab2id.items():
            if k not in vocab2id_dict:
                raise ZhVocabError(f"Key {k} is not in {name}")
            if v != vocab2id_dict[k]:
                raise ZhVocabError(f"For key {k}, value {vocab2id_dict[k]} does not match {v} in VOCAB2ID_SA")

_sa_vocab_compatibility_check()


# ==========================================================
# Mapping from fine grained labels to SA tag vocab.
# This is a temporary solution before we have hierachical category map.
# ==========================================================

AUDIO_V3_TO_SA_TAG_MAP = {
    # Genre
    "Pop": 'Pop',
    "Electronic": 'Electronic',
    "Chinese Style": 'Chinese Style',
    "Rock": 'Rock',
    "Jazz": 'Jazz',
    "Hip Hop/Rap": 'Hip Hop/Rap',
    "Classical": 'Classical',
    "R&B/Soul": 'R&B/Soul',
    "Metal": 'Metal',
    "Tuhai": 'DJ',
    "Childhood": 'Pop',
    "Devotional": 'Pop',
    "Chinese Tradition": "Chinese Tradition", 
    "Easy Listening": 'Easy Listening',
    "New Age": 'Easy Listening',
    "8 Bit": 'Electronic',
    "Folk": 'Folk',
    "Latin": 'Latin',
    "Alternative/Indie": 'Pop',
    "Epic": 'Other genre',
    "BGM": 'Other genre',
    "Chinese Pop": 'Pop',
    "Cantopop": 'Pop',
    "Taiwanese Pop": 'Pop',
    "Pop Folk": 'Folk',
    "Contemporary Pop": 'Pop',
    "Teen Pop": 'Pop',
    "Indie Pop": 'Pop',
    "Dream Pop": 'Pop',
    "City Pop": 'Pop',
    "Synth Pop": 'Pop',
    "Celtic Pop": 'Pop',
    "Dance Pop": 'Pop',
    "Deep Dance Pop": 'Electronic',
    "Electropop": 'Electronic',
    "Chamber Pop": 'Classical',
    "A cappella": 'Pop',
    "Country Pop": 'Country',
    "EDM": 'Electronic',
    "House": 'Electronic',
    "Dubstep": 'Electronic',
    "Future Bass": 'Electronic',
    "Chillout": 'Electronic',
    "Trance": 'Electronic',
    "Techno": 'Electronic',
    "Drum&Bass": 'Electronic',
    "Tropical House": 'Electronic',
    "Disco": 'Electronic',
    "Vaporwave": 'Electronic',
    "Trip Hop": 'Electronic',
    "Ambient": 'Electronic',
    "Deep Pop Edm": 'Electronic',
    "EDM Trap": 'Electronic',
    "Future House": 'Electronic',
    "China-Wave": 'Chinese Style',
    "GuFeng Music": 'Chinese Style',
    "Chinoiserie Rap": 'Hip Hop/Rap',
    "Chinoiserie Electronic": 'Electronic',
    "Hard Rock": 'Rock',
    "Psychedelic Rock": 'Rock',
    "Pop Rock": 'Rock',
    "Instrumental Rock": 'Rock',
    "Alternative Rock": 'Rock',
    "Indie Rock": 'Rock',
    "Post-Rock": 'Rock',
    "Lo-Fi": 'Easy Listening',
    "J Rock": 'Rock',
    "Shoegazing": 'Rock',
    "Math Rock": 'Rock',
    "Surf Rock": 'Rock',
    "Progressive Rock": 'Rock',
    "Soft Rock": 'Rock',
    "Jazz Pop": 'Jazz',
    "Jazz Fusion": 'Jazz',
    "Bossa Nova": 'Latin',
    "Avant-Garde Jazz": 'Jazz',
    "Swing": 'Jazz',
    "Big Band": 'Jazz',
    "Bop": 'Jazz',
    "Post-Bop": 'Jazz',
    "Smooth Jazz": 'Jazz',
    "Cool Jazz": 'Jazz',
    "Vocal Jazz": 'Jazz',
    "Nu Jazz": 'Jazz',
    "Free Jazz": 'Jazz',
    "Trap Rap": 'Hip Hop/Rap',
    "Old School": 'Hip Hop/Rap',
    "R&B Rap": 'Hip Hop/Rap',
    "Jazz Hip Hop": 'Hip Hop/Rap',
    "Alternative Hip Hop": 'Hip Hop/Rap',
    "Instrumental Hip Hop": 'Hip Hop/Rap',
    "Pop Rap": 'Hip Hop/Rap',
    "Hardcore Rap": 'Hip Hop/Rap',
    "Comedy Hip Hop": 'Hip Hop/Rap',
    "Hip House": 'Hip Hop/Rap',
    "Chill Beats": 'Hip Hop/Rap',
    "Chorus_AUDIO_GENRE": 'Other genre',
    "Chamber Music": 'Classical',
    "Symphony": 'Classical',
    "Funk": 'R&B/Soul',
    "Contemporary R&B": 'R&B/Soul',
    "Neo Soul": 'R&B/Soul',
    "Soul": 'R&B/Soul',
    "Pop Soul": 'R&B/Soul',
    "Black Metal": 'Metal',
    "Death Metal": 'Metal',
    "Glam Metal": 'Metal',
    "Grindcore": 'Metal',
    "Power Metal": 'Metal',
    "Progressive Metal": 'Metal',
    "Speed Metal": 'Metal',
    "DJ": 'DJ',
    "VinaHouse": 'Electronic',
    "MC": "MC",
    "Gospel": 'R&B/Soul',
    "Holiday Music": 'Pop',
    "Christian Music": 'Pop',
    "Buddhist music": 'Other genre',
    "Chinese Opera": 'Chinese Opera',
    "Traditional Chinese Folk": 'Chinese Tradition',
    "Chinese Quyi": 'Chinese Tradition',
    "Red Song": 'Chinese Tradition',
    "Folk Pop": 'Folk',
    "Indie Folk": 'Folk',
    "Tango": 'Latin',
    "Reggaeton": 'Latin',
    "Blues": 'Blues',
    "Country": 'Country',
    "Punk": 'Punk',
    "Reggae": 'Reggae',
    "Other genre": 'Other genre',
    "Boombap": 'Hip Hop/Rap',
    "Chillwave": 'Easy Listening',
    "Chinese Folk": 'Folk',
    "Contemporary Blues": 'Blues',
    "Contemporary Folk": 'Folk',
    "Country Blues": 'Blues',
    "Emo Rap": 'Hip Hop/Rap',
    "English Folk": 'Folk',
    "Flamenco": 'Latin',
    "Gangsta Rap": 'Hip Hop/Rap',
    "Hardstyle": 'Electronic',
    "Heavy Metal": 'Metal',
    "HiNRG": 'Electronic',
    "Indian Pop": 'Pop',
    "Italian Pop": 'Pop',
    "Japanese Folk": 'Folk',
    "J Pop": 'Pop',
    "KPop": 'Pop',
    "Latin Pop": 'Latin',
    "Low pop": 'Pop',
    "Melbourne Bounce": 'Electronic',
    "New Chinese Folk": 'Folk',
    "New Wave": 'Punk',
    "Nostalgic Pop": 'Pop',
    "Pop Punk": 'Punk',
    "Rock Blues": 'Blues',
    "Russian Pop": 'Pop',
    "Samba": 'Latin',
    "Shi Dai Qu": 'Pop',
    "Soundtrack": 'Other genre',
    "Thrash Metal": 'Metal',
    "Traditional Folk": 'Folk',
    "Traditional Jazz": 'Jazz',
    "Vulgar Pop": 'Pop',
    "West Coast Hip Hop": 'Hip Hop/Rap',
    "World Music": 'Other genre',
    "Breakbeat": 'Electronic',
    "Country Folk": 'Country',
    "Country Rock": 'Country',
    "Jazz Blues": 'Blues',
    "": 'Other genre',
    # Mood
    "Happy": "Happy",
    "Cute/Playful": "Cute",
    "Excited": "Excited",
    "Funny": "Funny",
    "Inspirational/Hopeful": "Inspirational",
    "Sorrow/Sad": "Sorrow",
    "Sentimental/Melancholic/Lonely": "Lonely",
    "Weird": "Weird",
    "Thrilling/Suspenseful/Tense": "Tense",
    "Angry/Aggressive": "Angry",
    "Groovy/Funky": "Dynamic",
    "Dynamic/Energetic": "Dynamic",
    "Romantic": "Romantic",
    "Nostalgic/Memory": "Miss/Memory",
    "Dreamy/Ethereal": "Calm",
    "Healing": "Healing",
    "Miss": "Miss/Memory",
    "Chill": "Chill",
    "Calm/Relaxing": "Calm",
    "Shocking/magnificent/epic": "Inspirational",
    "Mysterious": "Mysterious",
    "No Mood": "Other",
    "Other mood": "Other",
}


from recipes.mir2.parquet_dataset.vocab_index import KARAOKE_TAGS
from recipes.mir2.parquet_dataset.vocab_index import ARTIST_INFO
from recipes.mir2.parquet_dataset.utils import combine_tag_types
import torch
import numpy as np


def bigmusic_id2tag(tag_type, tag_id):
    #from IPython import embed; embed(using=False);
    meta_tag_type = tag_type_dict[tag_type]
    vocab2id = VOCAB2ID_MIX_V4
    for unified_tag, item in vocab2id.category_map[meta_tag_type].items():
        if item.value - 1 == tag_id:
            return unified_tag
    return 'none'

def get_tag_map():
    tag_map = combine_tag_types(datasets=[KARAOKE_TAGS, ARTIST_INFO])
    tag_map = { tag_type: { idx: category for category, idx in val.items()} for tag_type, val in tag_map.items() }
    return tag_map

def prob_to_tag(tag_pred, tag_map):
    res_tag = {}
    for tag_type in tag_map:
        res_tag[tag_type] = []
        batch_prob = torch.sigmoid(tag_pred[tag_type]).detach().cpu().numpy()
        # TODO: normalize the string to AR V5 vocab.
        for prob in batch_prob:            
            idx = np.argsort(-prob)
            if tag_type in ['ARTIST']:
                artist_tag = [(bigmusic_id2tag(tag_type, i), round(prob[i], 4)) for i in idx[:1] if prob[i]>0.3]
                if len(artist_tag) > 0:
                    res_tag[tag_type].append(artist_tag[0][0])
                else:
                    res_tag[tag_type].append("")
            elif tag_type == 'GENRE':
                genre_tags = [(bigmusic_id2tag(tag_type, i), round(prob[i], 4)) for i in idx[:3]]
                res_tag[tag_type].append(genre_tags[0][0])
            elif tag_type == 'instruments':
                instrument_tags = [(tag_map[tag_type][i], round(prob[i], 4)) for i in idx if prob[i]>0.22]
                res_tag[tag_type].append(",".join([x[0] for x in instrument_tags]))
            elif tag_type == 'genres':
                genre_tags = [(tag_map[tag_type][i], round(prob[i], 4)) for i in idx[:3]]
                res_tag[tag_type].append(genre_tags[0][0])
            elif tag_type in ['MOOD', 'THEME', 'GENDER', 'TIMBRE']:
                res_tag[tag_type].append([(bigmusic_id2tag(tag_type, i), round(prob[i], 4)) for i in idx[:1]])
            else:
                res_tag[tag_type].append({tag_map[tag_type][idx[0]]: round(prob[idx[0]], 4)})
    return res_tag

