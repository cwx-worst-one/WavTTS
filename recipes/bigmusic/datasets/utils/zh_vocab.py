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
# Genre, Mood, Theme, Sinking, Lang, Gender, Timber

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


class UnifiedVocab(Enum):
    pass


class UnifiedDumpster(UnifiedVocab):
    NONE = auto()


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


# =========================================================
# Unified Vocab2id Lookup Table
# NOTE: Order matters! Do not insert, only append.
# =========================================================

def lst_to_unified_vocab2id(lst: List[UnifiedVocab]) -> Dict[UnifiedVocab, int]:
    return {v: i for i, v in enumerate(lst)}

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
    UnifiedLang.CANTONESE,

    # SA lang
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
