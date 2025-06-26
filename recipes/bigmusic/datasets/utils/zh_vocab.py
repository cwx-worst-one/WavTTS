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
    EXTRA = auto()
    ARTIST = auto()
    INSTRUMENT = auto()
    TEMPO = auto()
    MODE = auto()
    KEY = auto()


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
    # audio tags ver2
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
    # Audio genre V5
    JERSEY_CLUB = auto()
    UK_GARAGE = auto()
    CHINOISERIE_ROCK = auto()
    MODERN_POP_BALLAD = auto()
    CELTIC_FOLK = auto()
    FUTURE_BOUNCE = auto()
    PHONK = auto()
    EARLY_COUNTRY = auto()
    AFROBEATS = auto()
    OTHER = auto()
    SCI_FI = auto()
    MIDDLE_AGES = auto()
    RENAISSANCE = auto()
    BIG_ROOM_HOUSE = auto()
    MIDTEMPO = auto()
    MELODIC_BASS = auto()
    COLOR_BASS = auto()
    MOOMBAHTON = auto()
    MELODIC_HARDCORE = auto()
    DUB = auto()
    CHA_CHA_CHA = auto()
    KAWAII_BASS = auto()
    JAPANESE_JAZZ_FUSION = auto()
    KAWAII_METAL = auto()
    AINU_FOLK = auto()
    OKINAWAN_POP = auto()


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
    # mood v5
    OTHER = auto()
    NO_MOOD = auto()
    BRISK_CAREFREE = auto()
    WARM_KIND = auto()
    CONFIDENT_DETERMINED = auto()
    COOL_SWAG = auto()
    ELEGANT_SOPHISTICATED = auto()
    SERIOUS_REFLECTIVE = auto()
    LYRICAL_BALLAD = auto()
    MOVING = auto()
    MAGICAL_FAIRYTALE = auto()
    DARK = auto()


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

    # audio theme V5
    LIVEHOUSE = auto()
    MUSIC_FESTIVAL = auto()
    OTHER = auto()
    NATURE = auto()
    NATIONAL_PRIDE = auto()
    SELF_AND_GROWTH = auto()
    CELEBRATION_AND_JOY = auto()
    LIFE_AND_MINDSET = auto()
    DREAM_AND_FUTURE = auto()
    MATSURI = auto()
    SUNRISE_DAWN = auto()
    SUNSET_AFTERNOON = auto()
    MIDNIGHT = auto()
    CLOUDY_DAY = auto()
    FOG = auto()
    STORM = auto()
    SKY_HEAVEN_PARADISE = auto()
    OCEAN_SEA = auto()
    FOREST = auto()
    OUTDOOR = auto()
    TEA_ROOM = auto()
    UNDERGOUND = auto()
    CIRCUS_CARNIVAL = auto()
    TEMPLE = auto()
    CHURCH = auto()
    COUNTRYSIDE = auto()
    EMOTIONAL = auto()
    TECHNOLOGY_SCIENCE = auto()
    HIGH_ENERGY_VIDEO = auto()
    DOCUMENTARY = auto()
    DRAMA = auto()
    SUPERNATURAL = auto()
    COMMERCIAL = auto()
    ADVERTISEMENT = auto()
    SENIORS_VIDEO = auto()
    VIRAL_SHORT_VIDEO = auto()
    TRAILER = auto()
    PATRIOTIC_VIDEO = auto()
    CULTURAL_EVENTS = auto()
    WAR_FIGHT = auto()
    ADVENTURE_DISCOVERY = auto()


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
    # lang v5
    OTHER = auto()
    PORTUGUESE = auto()
    SPANISH = auto()
    INDONESIAN = auto()
    THAI = auto()
    VIETNAMESE = auto()


class UnifiedSinking(UnifiedVocab):
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
    # gender v5
    NONVOCAL_NONGENDER = auto()


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
    # audio timbre V5
    BREATHY_VOICE = auto()
    ACG_VOICE = auto()
    BEL_CANTO = auto()
    CHINESE_FOLK = auto() # 民歌唱法，英文写法和genre重名，待修改
    GENTLE = auto()
    DELICATE = auto()
    ENERGETIC = auto()
    SASSY = auto()
    REFRESHING = auto()
    SOOTHING = auto()
    COMICAL = auto()
    UNCLE_LIKE = auto()
    GRITTY = auto()
    ROMANTIC = auto() # 英文写法和mood重名，待修改
    ENTHUSIASTIC = auto()
    SMOKY = auto()
    CHILDLIKE = auto()
    OPERA_TUNE = auto()
    NONVOCAL_NONTIMBRE = auto()
    ROUND = auto()
    FLAT = auto()
    GROWLING = auto()
    NASAL_VOICE = auto()

class UnifiedExtra(UnifiedVocab):
    # audio tags ver2
    EMPTY = auto()
    LO_FI = auto()
    NOSTALGIC = auto()
    SOUNDTRACK = auto()
    TUHAI = auto()
    # audio extra V5
    EXPERIMENTAL = auto()
    FASHIONABLE = auto()
    CANNED_MUSIC = auto()
    NON_NOSTALGIC = auto()
    NON_CANNED_MUSIC = auto()
    GRASSROOTS = auto()
    SQUARE_DANCE = auto()
    SENIORS_GRASSROOTS = auto()
    HITS_GRASSROOTS = auto()
    YOUNG_GRASSROOTS = auto()
    NIGHT_LOVING_SCENE = auto()
    ACG = auto()
    ANIME_OPENING_ENDING = auto()
    MUSICAL_THEATER = auto()
    OPERA = auto()
    SYMPHONY = auto()
    SONATA = auto()
    CONCERTO = auto()
    FUGUE = auto()
    WALTZ = auto()
    CHAMBER_MUSIC = auto()
    SOLO = auto()
    JAPANESE_STYLE = auto()

class UnifiedInstrument(UnifiedVocab):
    # from audio tags (https://bytedance.us.larkoffice.com/sheets/Nxjxs1D30hM6GItuKl0uaggXs5c?sheet=rSZMZo)
    EMPTY = auto()
    WOODWINDS = auto()
    PICCOLO = auto()
    FLUTE = auto()
    CLARINET = auto()
    OBOE = auto()
    ENGLISH_HORN = auto()
    PIPE = auto()
    SAXOPHONE = auto()
    SOPRANO_ALTO_SAX = auto()
    TENOR_SAX = auto()
    BARITONE_SAX = auto()
    BASSOON = auto()
    BRASS = auto()
    TRUMPET = auto()
    FRENCH_HORN = auto()
    TROMBONE = auto()
    TUBA = auto()
    BRASS_SECTION = auto()
    SYNTH_BRASS = auto()
    PERCUSSION = auto()
    DRUMS = auto()
    BASSDRUM = auto()
    SNARE = auto()
    HI_HAT = auto()
    TOM = auto()
    CYMBALS = auto()
    CHROMATIC_PERCUSSION = auto()
    MARIMBA = auto()
    BELLS = auto()
    BELLTREE = auto()
    XYLOPHONE = auto()
    GLOCKENSPIEL = auto()
    VIBRAPHONE = auto()
    CHIMES = auto()
    PERCUSSIVE = auto()
    SANDHAMMER = auto()
    TAMBOURINE = auto()
    TIMPANI = auto()
    SYNTH_DRUMS = auto()
    SYNTH_KICK = auto()
    SYNTH_SNARE = auto()
    SYNTH_HI_HAT = auto()
    SYNTH_TOM = auto()
    SYNTH_CYMBALS = auto()
    SYNTH_CLAVE = auto()
    FX = auto()
    CLAP = auto()
    KEYS = auto()
    ACOUSTIC_PIANO = auto()
    ELECTRIC_PIANO = auto()
    ORGAN = auto()
    ACCORDION = auto()
    STRINGS = auto()
    VIOLIN = auto()
    VIOLA = auto()
    CELLO = auto()
    CONTRABASS = auto()
    STRING_ENSEMBLE = auto()
    SYNTH_STRINGS = auto()
    BASS = auto()
    ELECTRIC_BASS = auto()
    SYNTH_BASS = auto()
    GUITAR = auto()
    ACOUSTIC_GUITAR = auto()
    ELECTRIC_GUITAR = auto()
    CLEAN_ELECTRIC_GUITAR = auto()
    DISTORTED_ELECTRIC_GUITAR = auto()
    CHINESE_TRADITIONAL_INSTRUMENTS = auto()
    DI = auto()
    XIAO = auto()
    SUONA = auto()
    ERHU = auto()
    GUZHENG = auto()
    PIPA = auto()
    YANGQIN = auto()
    SHENG = auto()
    HULUSI = auto()
    PANFLUTE = auto()
    XUN = auto()
    MATOUQIN = auto()
    RUAN = auto()
    SANXIAN = auto()
    GUQIN = auto()
    BIANZHONG = auto()
    CHINESE_DRUMS = auto()
    SYNTHESIZERS = auto()
    SYNTH_PLUCK = auto()
    SYNTH_LEAD = auto()
    SYNTH_PAD = auto()
    SYNTH_EFFECTS = auto()
    PLUCKED_STRINGS = auto()
    UKELELE = auto()
    ORCHESTRAL_HARP = auto()
    BANJO = auto()
    MANDOLIN = auto()
    ETHNICS = auto()
    BAGPIPE = auto()
    DULCIMER = auto()
    HANGDRUM = auto()
    HARMONICA = auto()
    IRISHWHISTLE = auto()
    OCARINA = auto()
    SITAR = auto()
    WHISTLE = auto()
    MUSICBOX = auto()
    SOUND_EFFECTS = auto()
    VOCALS = auto()
    BACKINGVOCAL = auto()
    CHORUS = auto()
    CHOIR_AND_VOICE = auto()
    VOCAL_CHOPS = auto()
    RIDE_CYMBAL = auto()
    HARPSICHORD = auto()
    PIPE_ORGAN = auto()
    SYNTH_BELL = auto()
    DOUBLE_BASS_PIZZICATO = auto()
    CHURCH_BELLS = auto()
    SINGING_BOWL = auto()
    CASTANETS = auto()
    TRIANGLE = auto()
    CLAVES = auto()
    COWBELL = auto()
    MARK_TREE = auto()
    CONGAS_BONGOS = auto()
    CAJON_BOX_DRUM = auto()
    WIND_CHIMES = auto()
    ORCHESTRAL_DRUMS = auto()
    ORCHESTRAL_BASSDRUM = auto()
    ORCHESTRAL_SNARE = auto()
    ORCHESTRAL_CYMBALS = auto()
    TAIKO_DRUMS = auto()
    TAM_TAM = auto()
    BODY_PERCUSSION = auto()
    FINGER_SNAPS = auto()
    BEAT_BOX = auto()
    CHINESE_PERCUSSION = auto()
    GONGS = auto()
    WOOD_BLOCK = auto()
    TANG_DRUMS = auto()
    BAN_DRUMS = auto()
    CHINESE_CLAPPERS = auto()
    CHINESE_CYMBALS = auto()
    YUNLUO = auto()
    BANGZI = auto()
    SYNTH_CHORD = auto()
    OTHER = auto()
    NONVOCAL = auto()
    CLAVINET= auto()
    CELESTA = auto()

class UnifiedTempo(UnifiedVocab):
    EMPTY = auto()
    GRAVE = auto()
    LARGO = auto()
    ADAGIO = auto()
    ANDANTE = auto()
    MODERATO = auto()
    ALLEGRO = auto()
    VIVACE = auto()
    PRESTO = auto()

class UnifiedMode(UnifiedVocab):
    EMPTY = auto()
    MAJOR = auto()
    MINOR = auto()
    GREGORIAN = auto()
    PENTATONIC = auto()
    BLUES = auto()
    NATURAL_MAJOR = auto()
    HARMONIC_MAJOR = auto()
    MELODIC_MAJOR = auto()
    NATURAK_MINOR = auto()
    HARMONIC_MINOR = auto()
    MELODIC_MINOR = auto()
    LYDIAN = auto()
    MIXOLYDIAN = auto()
    DORIAN = auto()
    PHRYGIAN = auto()
    BLUES_MAJOR = auto()
    BLUES_MINOR = auto()
    BLUES_COMBINED = auto()
    GONG = auto()
    SHANG = auto()
    YU = auto()
    ZHI = auto()
    JUE = auto()
    DUJIE = auto()
    OTHER = auto()
    LOCRIAN = auto()

class UnifiedKey(UnifiedVocab):
    EMPTY = auto()
    A = auto()
    A_SHARP = auto()
    #Ab = auto()
    B = auto()
    #Bb = auto()
    C = auto()
    C_SHARP = auto()
    Cb = auto()
    D = auto()
    D_SHARP = auto()
    #Db = auto()
    E = auto()
    #Eb = auto()
    F = auto()
    F_SHARP = auto()
    G = auto()
    G_SHARP = auto()
    #Gb = auto()


CATEGORY_MAP_INSTRUMENT_V1 = {
    UnifiedCategory.INSTRUMENT: {
        "Other Inst": UnifiedInstrument.EMPTY,
        "Woodwinds": UnifiedInstrument.WOODWINDS,
        "Piccolo": UnifiedInstrument.PICCOLO,
        "Flute": UnifiedInstrument.FLUTE,
        "Clarinet": UnifiedInstrument.CLARINET,
        "Oboe": UnifiedInstrument.OBOE,
        "English_Horn": UnifiedInstrument.ENGLISH_HORN,
        "Pipe": UnifiedInstrument.PIPE,
        "Saxophone": UnifiedInstrument.SAXOPHONE,
        "Soprano/Alto_Sax": UnifiedInstrument.SOPRANO_ALTO_SAX,
        "Tenor_Sax": UnifiedInstrument.TENOR_SAX,
        "Baritone_Sax": UnifiedInstrument.BARITONE_SAX,
        "Bassoon": UnifiedInstrument.BASSOON,
        "Brass": UnifiedInstrument.BRASS,
        "Trumpet": UnifiedInstrument.TRUMPET,
        "French_Horn": UnifiedInstrument.FRENCH_HORN,
        "Trombone": UnifiedInstrument.TROMBONE,
        "Tuba": UnifiedInstrument.TUBA,
        "Brass_Section": UnifiedInstrument.BRASS_SECTION,
        "Synth_Brass": UnifiedInstrument.SYNTH_BRASS,
        "Percussion": UnifiedInstrument.PERCUSSION,
        "Percussions": UnifiedInstrument.PERCUSSION,
        "Drums": UnifiedInstrument.DRUMS,
        "Drum_Set": UnifiedInstrument.DRUMS,    # merge Drums, Drum_Set
        "Bassdrum": UnifiedInstrument.BASSDRUM,
        "Bass_Drum": UnifiedInstrument.BASSDRUM,
        "Snare": UnifiedInstrument.SNARE,
        "Hi_hat": UnifiedInstrument.HI_HAT,
        "Tom": UnifiedInstrument.TOM,
        "Tom_Tom": UnifiedInstrument.TOM,
        "Cymbals": UnifiedInstrument.CYMBALS,
        "Crash_Cymbal": UnifiedInstrument.CYMBALS,
        "Chromatic_Percussion": UnifiedInstrument.CHROMATIC_PERCUSSION,
        "Marimba": UnifiedInstrument.MARIMBA,
        "Bells": UnifiedInstrument.BELLS,
        "Belltree": UnifiedInstrument.BELLTREE,
        "Xylophone": UnifiedInstrument.XYLOPHONE,
        "Glockenspiel": UnifiedInstrument.GLOCKENSPIEL,
        "Vibraphone": UnifiedInstrument.VIBRAPHONE,
        "Chimes": UnifiedInstrument.CHIMES,
        "Tubular_Bells ": UnifiedInstrument.CHIMES,
        "Tubular_Bells": UnifiedInstrument.CHIMES,
        "Percussive": UnifiedInstrument.PERCUSSIVE,
        "Sandhammer": UnifiedInstrument.SANDHAMMER,
        "Tambourine": UnifiedInstrument.TAMBOURINE,
        "Timpani": UnifiedInstrument.TIMPANI,
        "Synth_Drums": UnifiedInstrument.SYNTH_DRUMS,
        "Synth_Kick": UnifiedInstrument.SYNTH_KICK,
        "Synth_Snare": UnifiedInstrument.SYNTH_SNARE,
        "Synth_Hi_hat": UnifiedInstrument.SYNTH_HI_HAT,
        "Synth_Tom": UnifiedInstrument.SYNTH_TOM,
        "Synth_Cymbals": UnifiedInstrument.SYNTH_CYMBALS,
        "Synth_Clave": UnifiedInstrument.SYNTH_CLAVE,
        "Fx": UnifiedInstrument.FX,
        "Clap": UnifiedInstrument.CLAP,
        "Keys": UnifiedInstrument.KEYS,
        "Acoustic_Piano": UnifiedInstrument.ACOUSTIC_PIANO,
        "Acousticpiano": UnifiedInstrument.ACOUSTIC_PIANO,
        "Electric_Piano": UnifiedInstrument.ELECTRIC_PIANO,
        "Electric_piano": UnifiedInstrument.ELECTRIC_PIANO,
        "Organ": UnifiedInstrument.ORGAN,
        "Accordion": UnifiedInstrument.ACCORDION,
        "Strings": UnifiedInstrument.STRINGS,
        "Violin": UnifiedInstrument.VIOLIN,
        "Viola": UnifiedInstrument.VIOLA,
        "Cello": UnifiedInstrument.CELLO,
        "Contrabass": UnifiedInstrument.CONTRABASS,
        "Double_Bass": UnifiedInstrument.CONTRABASS,
        "String_Ensemble": UnifiedInstrument.STRING_ENSEMBLE,
        "Synth_Strings": UnifiedInstrument.SYNTH_STRINGS,
        "Bass": UnifiedInstrument.BASS,
        "Electric_Bass": UnifiedInstrument.ELECTRIC_BASS,
        "Synth_Bass": UnifiedInstrument.SYNTH_BASS,
        "Guitar": UnifiedInstrument.GUITAR,
        "Acoustic_Guitar": UnifiedInstrument.ACOUSTIC_GUITAR,
        "Electric_Guitar": UnifiedInstrument.ELECTRIC_GUITAR,
        "Clean_Electric_Guitar": UnifiedInstrument.CLEAN_ELECTRIC_GUITAR,
        "Distorted_Electric_Guitar": UnifiedInstrument.DISTORTED_ELECTRIC_GUITAR,
        "Chinese_Traditional_Instruments": UnifiedInstrument.CHINESE_TRADITIONAL_INSTRUMENTS,
        "Chinese Traditional Instruments": UnifiedInstrument.CHINESE_TRADITIONAL_INSTRUMENTS,
        "Di": UnifiedInstrument.DI,
        "Xiao": UnifiedInstrument.XIAO,
        "Suona": UnifiedInstrument.SUONA,
        "Erhu": UnifiedInstrument.ERHU,
        "Guzheng": UnifiedInstrument.GUZHENG,
        "Pipa": UnifiedInstrument.PIPA,
        "Yangqin": UnifiedInstrument.YANGQIN,
        "Sheng": UnifiedInstrument.SHENG,
        "Hulusi": UnifiedInstrument.HULUSI,
        "Panflute": UnifiedInstrument.PANFLUTE,
        "Xun": UnifiedInstrument.XUN,
        "Matouqin": UnifiedInstrument.MATOUQIN,
        "Ruan": UnifiedInstrument.RUAN,
        "Sanxian": UnifiedInstrument.SANXIAN,
        "Guqin": UnifiedInstrument.GUQIN,
        "Bianzhong": UnifiedInstrument.BIANZHONG,
        "Chinese_Drums": UnifiedInstrument.CHINESE_DRUMS,
        "Chinesedrums": UnifiedInstrument.CHINESE_DRUMS,
        "Synthesizers": UnifiedInstrument.SYNTHESIZERS,
        "Synth_Pluck": UnifiedInstrument.SYNTH_PLUCK,
        "Synth_Lead": UnifiedInstrument.SYNTH_LEAD,
        "Synth_Pad": UnifiedInstrument.SYNTH_PAD,
        "Synth_Effects": UnifiedInstrument.SYNTH_EFFECTS,
        "Plucked_Strings": UnifiedInstrument.PLUCKED_STRINGS,
        "Plucked Strings": UnifiedInstrument.PLUCKED_STRINGS,
        "Ukelele": UnifiedInstrument.UKELELE,
        "Orchestral_Harp": UnifiedInstrument.ORCHESTRAL_HARP,
        "Banjo": UnifiedInstrument.BANJO,
        "Mandolin": UnifiedInstrument.MANDOLIN,
        "Ethnic": UnifiedInstrument.ETHNICS,
        "Bagpipe": UnifiedInstrument.BAGPIPE,
        "Dulcimer": UnifiedInstrument.DULCIMER,
        "Hangdrum": UnifiedInstrument.HANGDRUM,
        "Harmonica": UnifiedInstrument.HARMONICA,
        "Irishwhistle": UnifiedInstrument.IRISHWHISTLE,
        "Ocarina": UnifiedInstrument.OCARINA,
        "Sitar": UnifiedInstrument.SITAR,
        "Whistle": UnifiedInstrument.WHISTLE,
        "Musicbox": UnifiedInstrument.MUSICBOX,
        "Sound_Effects": UnifiedInstrument.SOUND_EFFECTS,
        "Vocal": UnifiedInstrument.VOCALS,
        "Vocals": UnifiedInstrument.VOCALS,
        "Backingvocal": UnifiedInstrument.BACKINGVOCAL,
        "Backing_Vocals": UnifiedInstrument.BACKINGVOCAL,
        "Chorus": UnifiedInstrument.CHORUS,
        "Choir_and_Voice": UnifiedInstrument.CHOIR_AND_VOICE,
        "Vocal_Chops": UnifiedInstrument.VOCAL_CHOPS,
        "Ride_Cymbal": UnifiedInstrument.RIDE_CYMBAL,
        "Harpsichord": UnifiedInstrument.HARPSICHORD,
        "Pipe_Organ": UnifiedInstrument.PIPE_ORGAN,
        "Synth_Bell": UnifiedInstrument.SYNTH_BELL,
        "Double_Bass_Pizzicato": UnifiedInstrument.DOUBLE_BASS_PIZZICATO,
        "Church_Bells": UnifiedInstrument.CHURCH_BELLS,
        "Singing_Bowl": UnifiedInstrument.SINGING_BOWL,
        "Castanets": UnifiedInstrument.CASTANETS,
        "Triangle": UnifiedInstrument.TRIANGLE,
        "Claves": UnifiedInstrument.CLAVES,
        "Cowbell": UnifiedInstrument.COWBELL,
        "Mark_Tree": UnifiedInstrument.MARK_TREE,
        "Congas/Bongos": UnifiedInstrument.CONGAS_BONGOS,
        "Cajón/Box_drum": UnifiedInstrument.CAJON_BOX_DRUM,
        "Wind_Chimes": UnifiedInstrument.WIND_CHIMES,
        "Orchestral_Drums": UnifiedInstrument.ORCHESTRAL_DRUMS,
        "Orchestral_Bassdrum": UnifiedInstrument.ORCHESTRAL_BASSDRUM,
        "Orchestral_Snare": UnifiedInstrument.ORCHESTRAL_SNARE,
        "Orchestral_Cymbals": UnifiedInstrument.ORCHESTRAL_CYMBALS,
        "Taiko_Drums": UnifiedInstrument.TAIKO_DRUMS,
        "Tam_Tam": UnifiedInstrument.TAM_TAM,
        "Body_Percussion": UnifiedInstrument.BODY_PERCUSSION,
        "Finger_snaps": UnifiedInstrument.FINGER_SNAPS,
        "Beat_box": UnifiedInstrument.BEAT_BOX,
        "Chinese_Percussion": UnifiedInstrument.CHINESE_PERCUSSION,
        "Gongs": UnifiedInstrument.GONGS,
        "Wood_Block": UnifiedInstrument.WOOD_BLOCK,
        "Tang_Drums": UnifiedInstrument.TANG_DRUMS,
        "Ban_Drums": UnifiedInstrument.BAN_DRUMS,
        "Chinese_Clappers": UnifiedInstrument.CHINESE_CLAPPERS,
        "Chinese_Cymbals": UnifiedInstrument.CHINESE_CYMBALS,
        "Yunluo": UnifiedInstrument.YUNLUO,
        "Bangzi": UnifiedInstrument.BANGZI,
        "Synth_Chord": UnifiedInstrument.SYNTH_CHORD,
    }
}


class UnifiedArtist(UnifiedVocab):
    # taxonomy from artist tagging model (may subject to change according to Ju-Chiang's experiments)
    EMPTY = auto()
    DENGLIJUN = auto()
    LIUDEHUA = auto()
    ZHANGXUEYOU = auto()
    MOWENWEI = auto()
    ZHOUJIELUN = auto()
    LINJUNJIE = auto()
    ZHOU_SHEN = auto()
    CAIYILIN = auto()
    WANGSULONG = auto()
    CAIJIANYA = auto()
    YINQUESHITING = auto()
    ZHANGSHAOHAN = auto()
    WU_YUE_TIAN = auto()
    WANGXINLING = auto()
    FANGDATONG = auto()
    S_H_E = auto()
    SUNYANZI = auto()
    LINYOUJIA = auto()
    G_E_M_DENGZIQI = auto()
    TAOZHE = auto()
    CHENLI = auto()
    SUDALU = auto()
    LIJIAN = auto()
    LIRONGHAO = auto()
    MAOBUYI = auto()
    VAVA_MAOYANQI = auto()
    HUACHENYU = auto()
    YANGZONGWEI = auto()
    WANGYITAI = auto()
    ERIC_ZHOUXINGZHE = auto()
    SHANYICHUN = auto()
    YUJIAYUN = auto()
    JONY_J = auto()
    YANRENZHONG = auto()
    WANGXINCHEN = auto()
    DUIZHANG = auto()
    LIL_GHOST_XIAOGUI = auto()
    DAVE_MATTHEWS_BAND = auto()
    DAVID_BOWIE = auto()
    DJ_DRAMA = auto()
    MANIC_STREET_PREACHERS = auto()
    JIMMY_WITHERSPOON = auto()
    FRANK_SINATRA = auto()
    BING_CROSBY = auto()
    LIGHTNIN_HOPKINS = auto()
    ERIC_BELLINGER = auto()
    JT_MUSIC = auto()
    LOWELL_FULSON = auto()
    CHAMPION_JACK_DUPREE = auto()
    DR_JOHN = auto()
    BRUCE_SPRINGSTEEN = auto()
    SHIRLEY_BASSEY = auto()
    KYLIE_MINOGUE = auto()
    CHRIS_REA = auto()
    BIG_MAYBELLE = auto()
    ROSEMARY_CLOONEY = auto()
    NANCY_WILSON = auto()
    AMOS_MILBURN = auto()
    LITTLE_MILTON = auto()
    BIG_BILL_BROONZY = auto()
    BARENAKED_LADIES = auto()
    GLENN_MILLER = auto()
    RAY_CHARLES = auto()
    MEL_TORME = auto()
    MADONNA = auto()
    BUDDY_GUY = auto()
    BEYONCE = auto()
    THE_GOO_GOO_DOLLS = auto()
    MARIA_MULDAUR = auto()
    SWITCHFOOT = auto()
    EVERYTHING_BUT_THE_GIRL = auto()
    TINA_TURNER = auto()
    ROY_BROWN = auto()
    JIMMY_REED = auto()
    DORIS_DAY = auto()
    BOB_MARLEY_AND_THE_WAILERS = auto()
    R_KELLY = auto()
    DINAH_WASHINGTON = auto()
    LOU_REED = auto()
    SARAH_VAUGHAN = auto()
    BABYFACE_RAY = auto()
    LINKIN_PARK = auto()
    CHRIS_BROWN = auto()
    CANNED_HEAT = auto()
    TONY_BENNETT = auto()
    SIMPLY_RED = auto()
    JUNIOR_WELLS = auto()
    THE_BLACK_CROWES = auto()
    TREY_SONGZ = auto()
    BONNIE_RAITT = auto()
    MOBY = auto()
    DEPECHE_MODE = auto()
    BOBBY_DARIN = auto()
    NORAH_JONES = auto()
    PHIL_COLLINS = auto()
    JUDY_GARLAND = auto()
    BJORK = auto()
    SCREAMIN_JAY_HAWKINS = auto()
    NICK_CAVE_AND_THE_BAD_SEEDS = auto()
    PERRY_COMO = auto()
    ELECTRIC_SIX = auto()
    MIRACLE_OF_SOUND = auto()
    LONNIE_JOHNSON = auto()
    BILLIE_HOLIDAY = auto()
    BIG_MAMA_THORNTON = auto()
    LOST_FREQUENCIES = auto()
    MEMPHIS_SLIM = auto()
    ALBERT_KING = auto()
    LITTLE_MIX = auto()
    COLDPLAY = auto()
    JOHN_MAYER = auto()
    SONNY_BOY_WILLIAMSON_II = auto()
    PRIMAL_SCREAM = auto()
    CHRIS_ISAAK = auto()
    DIPLO = auto()
    RELIENT_K = auto()
    CELINE_DION = auto()
    IAMX = auto()
    TOM_WAITS = auto()
    RED_HOT_CHILI_PEPPERS = auto()
    AEROSMITH = auto()
    TONY_JOE_WHITE = auto()
    GARBAGE = auto()
    DELBERT_MCCLINTON = auto()
    BLACKBEAR = auto()
    POPA_CHUBBY = auto()
    MICHAEL_BUBLE = auto()
    USHER = auto()
    MEGHAN_TRAINOR = auto()
    PINK = auto()
    ROBERT_CRAY = auto()
    JOHN_MELLENCAMP = auto()
    GOLDFRAPP = auto()
    ROBBIE_WILLIAMS = auto()
    HOT_TUNA = auto()
    JAMES_BLUNT = auto()
    LIL_UZI_VERT = auto()
    NINA_SIMONE = auto()
    EYEDRESS = auto()
    ELLA_FITZGERALD = auto()
    BRITNEY_SPEARS = auto()
    HOT_CHIP = auto()
    LIGHTS = auto()
    AIR_SUPPLY = auto()
    DIDO = auto()
    TORY_LANEZ = auto()
    FEEDER = auto()
    NEFFEX = auto()
    MEMPHIS_MINNIE = auto()
    GROOVE_ARMADA = auto()
    THE_CHAINSMOKERS = auto()
    COLLECTIVE_SOUL = auto()
    INXS = auto()
    JAMES_COTTON = auto()
    ROISIN_MURPHY = auto()
    TAJ_MAHAL = auto()
    WHITNEY_HOUSTON = auto()
    TRAPT = auto()
    CHARLIE_MUSSELWHITE = auto()
    MIKE_BLOOMFIELD = auto()
    CHROMEO = auto()
    JIMMY_ROGERS = auto()
    CRAIG_DAVID = auto()
    CALVIN_HARRIS = auto()
    KATIE_MELUA = auto()
    CREEDENCE_CLEARWATER_REVIVAL = auto()
    AL_BOWLLY = auto()
    ETTA_JAMES = auto()
    SNOOKS_EAGLIN = auto()
    KEB_MO = auto()
    ODETTA = auto()
    TLC = auto()
    PROFESSOR_LONGHAIR = auto()
    BRYAN_ADAMS = auto()
    RY_COODER = auto()
    FATS_DOMINO = auto()
    PAPA_ROACH = auto()
    SHINEDOWN = auto()
    MIDNIGHT_OIL = auto()
    CHRISTINA_AGUILERA = auto()
    PATTI_SMITH = auto()
    LUTHER_ALLISON = auto()
    MIGUEL = auto()
    UNLIKE_PLUTO = auto()
    FOO_FIGHTERS = auto()
    LIVING_COLOUR = auto()
    THE_SHEEPDOGS = auto()
    KELLY_CLARKSON = auto()
    WILLIE_DIXON = auto()
    SKUNK_ANANSIE = auto()
    SEETHER = auto()
    MADRUGADA = auto()
    TOTO = auto()
    POETS_OF_THE_FALL = auto()
    CHRIS_WHITLEY = auto()
    GLENN_MILLER_ORCHESTRA = auto()
    OTIS_SPANN = auto()
    BEN_E_KING = auto()
    KID_ROCK = auto()
    NICKELBACK = auto()
    KEITH_SWEAT = auto()
    CHRIS_SMITHER = auto()
    FATS_WALLER = auto()
    BLACK_REBEL_MOTORCYCLE_CLUB = auto()
    MY_CHEMICAL_ROMANCE = auto()
    PAUL_WELLER = auto()
    HARRY_CONNICK_JR = auto()
    JOYWAVE = auto()
    KENNY_WAYNE_SHEPHERD = auto()
    MILEY_CYRUS = auto()
    CYNDI_LAUPER = auto()
    TOMMY_DORSEY = auto()
    PANIC_AT_THE_DISCO = auto()
    SAN_HOLO = auto()
    SON_LUX = auto()
    THE_CALIFORNIA_HONEYDROPS = auto()
    MISSISSIPPI_FRED_MCDOWELL = auto()
    TINSLEY_ELLIS = auto()
    TOMMY_CASTRO = auto()
    JAGGED_EDGE = auto()
    DUA_LIPA = auto()
    JOHN_MAYALL_AND_THE_BLUESBREAKERS = auto()
    OLLY_MURS = auto()
    ALTER_BRIDGE = auto()
    FAITHLESS = auto()
    PARAMORE = auto()
    HONNE = auto()
    AL_JARREAU = auto()
    BREATHE_CAROLINA = auto()
    RUFUS_DU_SOL = auto()
    JASON_DERULO = auto()
    DEMI_LOVATO = auto()
    CARO_EMERALD = auto()
    RAY_J = auto()
    ESTELLE = auto()
    TINK = auto()
    JAZZANOVA = auto()
    ASH_GRUNWALD = auto()
    BILLY_JOEL = auto()
    KID_TRAVIS = auto()
    ADAM_LAMBERT = auto()
    ELVIN_BISHOP = auto()
    AVRIL_LAVIGNE = auto()
    J_B_LENOIR = auto()
    CAVETOWN = auto()
    CLARENCE_GATEMOUTH_BROWN = auto()
    HIPPIE_SABOTAGE = auto()
    LOUIS_ARMSTRONG = auto()
    BIG_JOE_WILLIAMS = auto()
    TAB_BENOIT = auto()
    ROBERT_GLASPER = auto()
    PARTYNEXTDOOR = auto()
    JOSE_JAMES = auto()
    BIG_JOE_TURNER = auto()
    CROWDED_HOUSE = auto()
    ROBBEN_FORD = auto()
    WEEZER = auto()
    KELIS = auto()
    RAMIREZ = auto()
    MUDDY_WATERS = auto()
    TOKIMONSTA = auto()
    THE_RASMUS = auto()
    BLIND_WILLIE_MCTELL = auto()
    ANA_POPOVIC = auto()
    JMSN = auto()
    JEEZY = auto()
    ANNE_MARIE = auto()
    CHARLIE_PUTH = auto()
    CRASH_TEST_DUMMIES = auto()
    KYGO = auto()
    BENNY_SINGS = auto()
    TINASHE = auto()
    KOKO_TAYLOR = auto()
    DAVID_GUETTA = auto()
    SCOTT_H_BIRAM = auto()
    BABY_TATE = auto()
    BRIAN_MCKNIGHT = auto()
    BASEMENT_JAXX = auto()
    GEORGE_BENSON = auto()
    BOBBY_BLUE_BLAND = auto()
    THE_VERONICAS = auto()
    ACE_OF_BASE = auto()
    COLIN_JAMES = auto()
    EVANESCENCE = auto()
    OUR_LADY_PEACE = auto()
    SHEMEKIA_COPELAND = auto()
    SHAWN_JAMES = auto()
    OMARION = auto()
    JOHNNY_WINTER = auto()
    BON_JOVI = auto()
    LAUV = auto()
    THE_KNOCKS = auto()
    MOTIONLESS_IN_WHITE = auto()
    JAMES_ARTHUR = auto()
    LABRINTH = auto()
    CHRIS_THOMAS_KING = auto()
    MARC_E_BASSY = auto()
    LCD_SOUNDSYSTEM = auto()
    ELVIS_COSTELLO = auto()
    THE_PRESIDENTS_OF_THE_UNITED_STATES_OF_AMERICA = auto()
    BRANDY = auto()
    CHROMATICS = auto()
    DVSN = auto()
    THE_DARKNESS = auto()
    FILTER = auto()
    TRENTEMOLLER = auto()
    SOLANGE = auto()
    MONICA = auto()
    AWOLNATION = auto()
    SIMPLE_PLAN = auto()
    ELMORE_JAMES = auto()
    STEREOPHONICS = auto()
    BLUES_TRAVELER = auto()
    FAITH_EVANS = auto()
    MARINA = auto()
    MIKE_POSNER = auto()
    EN_VOGUE = auto()
    G_LOVE_AND_SPECIAL_SAUCE = auto()
    CHARLI_XCX = auto()
    SAM_FELDT = auto()
    GUSGUS = auto()
    ANDERS_OSBORNE = auto()
    COCO_MONTOYA = auto()
    OLIVER_TREE = auto()
    CLEAN_BANDIT = auto()
    TWENTY_ONE_PILOTS = auto()
    PNB_ROCK = auto()
    MARILYN_MONROE = auto()
    BACKSTREET_BOYS = auto()
    PJ_MORTON = auto()
    SAINT_ETIENNE = auto()
    LADYTRON = auto()
    DJ_CANDLESTICK = auto()
    OG_RON_C = auto()
    INCUBUS = auto()
    BLIND_BOY_FULLER = auto()
    NATASHA_BEDINGFIELD = auto()
    TONI_BRAXTON = auto()
    KHALID = auto()
    SAMANTHA_FISH = auto()
    WHEATUS = auto()
    PHANTOM_PLANET = auto()
    DAWN_RICHARD = auto()
    JUSTICE = auto()
    SOPHIE_ELLIS_BEXTOR = auto()
    STACEY_KENT = auto()
    CIARA = auto()
    ALBERT_CUMMINGS = auto()
    FANTASIA = auto()
    TATE_MCRAE = auto()
    THE_WHITE_STRIPES = auto()
    DAUGHTRY = auto()
    BUKKA_WHITE = auto()
    BRYSON_TILLER = auto()
    SANTANA = auto()
    AVA_MAX = auto()
    I_MONSTER = auto()
    THE_DELTA_SAINTS = auto()
    JULIE_LONDON = auto()
    OTIS_RUSH = auto()
    JJ_GREY_AND_MOFRO = auto()
    HOOTIE_AND_THE_BLOWFISH = auto()
    VEDO = auto()
    JOAN_JETT_AND_THE_BLACKHEARTS = auto()
    GOLDLINK = auto()
    DOJA_CAT = auto()
    ANDREA_STORM_KADEN = auto()
    SLIM_HARPO = auto()
    JENNIFER_LOPEZ = auto()
    FLIGHT_FACILITIES = auto()
    KELELA = auto()
    NAT_KING_COLE = auto()
    JAY_SEAN = auto()
    CARLY_RAE_JEPSEN = auto()
    LION_BABE = auto()
    KAI_STRAW = auto()
    WESTLIFE = auto()
    LADY_A = auto()
    THE_PAUL_BUTTERFIELD_BLUES_BAND = auto()
    GEORGE_MICHAEL = auto()
    RAY_EBERLE = auto()
    CHE_ECRU = auto()
    RIHANNA = auto()
    STEVIE_RAY_VAUGHAN = auto()
    CHEAT_CODES = auto()
    _3OH_3 = auto()
    ALI_GATIE = auto()
    CALVIN_RUSSELL = auto()
    MGMT = auto()
    FRED_ASTAIRE = auto()
    FLEET_FOXES = auto()
    LILY_ALLEN = auto()


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

# A vocab2id lookup table that supports SA tags and Audio tags V3, and V4
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

    # Audio genre V5
    UnifiedGenre.JERSEY_CLUB,
    UnifiedGenre.UK_GARAGE,
    UnifiedGenre.CHINOISERIE_ROCK,
    UnifiedGenre.MODERN_POP_BALLAD,
    UnifiedGenre.CELTIC_FOLK,
    UnifiedGenre.FUTURE_BOUNCE,
    UnifiedGenre.PHONK,
    UnifiedGenre.EARLY_COUNTRY,
    UnifiedGenre.AFROBEATS,

    # audio theme V5
    UnifiedTheme.LIVEHOUSE,
    UnifiedTheme.MUSIC_FESTIVAL,

    # audio timbre V5
    UnifiedTimbre.BREATHY_VOICE,
    UnifiedTimbre.ACG_VOICE,
    UnifiedTimbre.BEL_CANTO,
    UnifiedTimbre.CHINESE_FOLK,
    UnifiedTimbre.GENTLE,
    UnifiedTimbre.DELICATE,
    UnifiedTimbre.ENERGETIC,
    UnifiedTimbre.SASSY,
    UnifiedTimbre.REFRESHING,
    UnifiedTimbre.SOOTHING,
    UnifiedTimbre.COMICAL,
    UnifiedTimbre.UNCLE_LIKE,
    UnifiedTimbre.GRITTY,
    UnifiedTimbre.ROMANTIC,
    UnifiedTimbre.ENTHUSIASTIC,
    UnifiedTimbre.SMOKY,
    UnifiedTimbre.CHILDLIKE,
    UnifiedTimbre.OPERA_TUNE,

    # audio extra V5
    UnifiedExtra.EXPERIMENTAL,
    UnifiedExtra.FASHIONABLE,
    UnifiedExtra.CANNED_MUSIC,

    # audio inst V5
    UnifiedInstrument.EMPTY,
    UnifiedInstrument.WOODWINDS,
    UnifiedInstrument.PICCOLO,
    UnifiedInstrument.FLUTE,
    UnifiedInstrument.CLARINET,
    UnifiedInstrument.OBOE,
    UnifiedInstrument.ENGLISH_HORN,
    UnifiedInstrument.PIPE,
    UnifiedInstrument.SAXOPHONE,
    UnifiedInstrument.SOPRANO_ALTO_SAX,
    UnifiedInstrument.TENOR_SAX,
    UnifiedInstrument.BARITONE_SAX,
    UnifiedInstrument.BASSOON,
    UnifiedInstrument.BRASS,
    UnifiedInstrument.TRUMPET,
    UnifiedInstrument.FRENCH_HORN,
    UnifiedInstrument.TROMBONE,
    UnifiedInstrument.TUBA,
    UnifiedInstrument.BRASS_SECTION,
    UnifiedInstrument.SYNTH_BRASS,
    UnifiedInstrument.PERCUSSION,
    UnifiedInstrument.DRUMS,
    UnifiedInstrument.BASSDRUM,
    UnifiedInstrument.SNARE,
    UnifiedInstrument.HI_HAT,
    UnifiedInstrument.TOM,
    UnifiedInstrument.CYMBALS,
    UnifiedInstrument.CHROMATIC_PERCUSSION,
    UnifiedInstrument.MARIMBA,
    UnifiedInstrument.BELLS,
    UnifiedInstrument.BELLTREE,
    UnifiedInstrument.XYLOPHONE,
    UnifiedInstrument.GLOCKENSPIEL,
    UnifiedInstrument.VIBRAPHONE,
    UnifiedInstrument.CHIMES,
    UnifiedInstrument.PERCUSSIVE,
    UnifiedInstrument.SANDHAMMER,
    UnifiedInstrument.TAMBOURINE,
    UnifiedInstrument.TIMPANI,
    UnifiedInstrument.SYNTH_DRUMS,
    UnifiedInstrument.SYNTH_KICK,
    UnifiedInstrument.SYNTH_SNARE,
    UnifiedInstrument.SYNTH_HI_HAT,
    UnifiedInstrument.SYNTH_TOM,
    UnifiedInstrument.SYNTH_CYMBALS,
    UnifiedInstrument.SYNTH_CLAVE,
    UnifiedInstrument.FX,
    UnifiedInstrument.CLAP,
    UnifiedInstrument.KEYS,
    UnifiedInstrument.ACOUSTIC_PIANO,
    UnifiedInstrument.ELECTRIC_PIANO,
    UnifiedInstrument.ORGAN,
    UnifiedInstrument.ACCORDION,
    UnifiedInstrument.STRINGS,
    UnifiedInstrument.VIOLIN,
    UnifiedInstrument.VIOLA,
    UnifiedInstrument.CELLO,
    UnifiedInstrument.CONTRABASS,
    UnifiedInstrument.STRING_ENSEMBLE,
    UnifiedInstrument.SYNTH_STRINGS,
    UnifiedInstrument.BASS,
    UnifiedInstrument.ELECTRIC_BASS,
    UnifiedInstrument.SYNTH_BASS,
    UnifiedInstrument.GUITAR,
    UnifiedInstrument.ACOUSTIC_GUITAR,
    UnifiedInstrument.ELECTRIC_GUITAR,
    UnifiedInstrument.CLEAN_ELECTRIC_GUITAR,
    UnifiedInstrument.DISTORTED_ELECTRIC_GUITAR,
    UnifiedInstrument.CHINESE_TRADITIONAL_INSTRUMENTS,
    UnifiedInstrument.DI,
    UnifiedInstrument.XIAO,
    UnifiedInstrument.SUONA,
    UnifiedInstrument.ERHU,
    UnifiedInstrument.GUZHENG,
    UnifiedInstrument.PIPA,
    UnifiedInstrument.YANGQIN,
    UnifiedInstrument.SHENG,
    UnifiedInstrument.HULUSI,
    UnifiedInstrument.PANFLUTE,
    UnifiedInstrument.XUN,
    UnifiedInstrument.MATOUQIN,
    UnifiedInstrument.RUAN,
    UnifiedInstrument.SANXIAN,
    UnifiedInstrument.GUQIN,
    UnifiedInstrument.BIANZHONG,
    UnifiedInstrument.CHINESE_DRUMS,
    UnifiedInstrument.SYNTHESIZERS,
    UnifiedInstrument.SYNTH_PLUCK,
    UnifiedInstrument.SYNTH_LEAD,
    UnifiedInstrument.SYNTH_PAD,
    UnifiedInstrument.SYNTH_EFFECTS,
    UnifiedInstrument.PLUCKED_STRINGS,
    UnifiedInstrument.UKELELE,
    UnifiedInstrument.ORCHESTRAL_HARP,
    UnifiedInstrument.BANJO,
    UnifiedInstrument.MANDOLIN,
    UnifiedInstrument.ETHNICS,
    UnifiedInstrument.BAGPIPE,
    UnifiedInstrument.DULCIMER,
    UnifiedInstrument.HANGDRUM,
    UnifiedInstrument.HARMONICA,
    UnifiedInstrument.IRISHWHISTLE,
    UnifiedInstrument.OCARINA,
    UnifiedInstrument.SITAR,
    UnifiedInstrument.WHISTLE,
    UnifiedInstrument.MUSICBOX,
    UnifiedInstrument.SOUND_EFFECTS,
    UnifiedInstrument.VOCALS,
    UnifiedInstrument.BACKINGVOCAL,
    UnifiedInstrument.CHORUS,
    UnifiedInstrument.CHOIR_AND_VOICE,
    UnifiedInstrument.VOCAL_CHOPS,
    UnifiedInstrument.RIDE_CYMBAL,
    UnifiedInstrument.HARPSICHORD,
    UnifiedInstrument.PIPE_ORGAN,
    UnifiedInstrument.SYNTH_BELL,
    UnifiedInstrument.DOUBLE_BASS_PIZZICATO,
    UnifiedInstrument.CHURCH_BELLS,
    UnifiedInstrument.SINGING_BOWL,
    UnifiedInstrument.CASTANETS,
    UnifiedInstrument.TRIANGLE,
    UnifiedInstrument.CLAVES,
    UnifiedInstrument.COWBELL,
    UnifiedInstrument.MARK_TREE,
    UnifiedInstrument.CONGAS_BONGOS,
    UnifiedInstrument.CAJON_BOX_DRUM,
    UnifiedInstrument.WIND_CHIMES,
    UnifiedInstrument.ORCHESTRAL_DRUMS,
    UnifiedInstrument.ORCHESTRAL_BASSDRUM,
    UnifiedInstrument.ORCHESTRAL_SNARE,
    UnifiedInstrument.ORCHESTRAL_CYMBALS,
    UnifiedInstrument.TAIKO_DRUMS,
    UnifiedInstrument.TAM_TAM,
    UnifiedInstrument.BODY_PERCUSSION,
    UnifiedInstrument.FINGER_SNAPS,
    UnifiedInstrument.BEAT_BOX,
    UnifiedInstrument.CHINESE_PERCUSSION,
    UnifiedInstrument.GONGS,
    UnifiedInstrument.WOOD_BLOCK,
    UnifiedInstrument.TANG_DRUMS,
    UnifiedInstrument.BAN_DRUMS,
    UnifiedInstrument.CHINESE_CLAPPERS,
    UnifiedInstrument.CHINESE_CYMBALS,
    UnifiedInstrument.YUNLUO,
    UnifiedInstrument.BANGZI,
    UnifiedInstrument.SYNTH_CHORD,

    # audio tempo V5
    UnifiedTempo.EMPTY,
    UnifiedTempo.GRAVE,
    UnifiedTempo.LARGO,
    UnifiedTempo.ADAGIO,
    UnifiedTempo.ANDANTE,
    UnifiedTempo.MODERATO,
    UnifiedTempo.ALLEGRO,
    UnifiedTempo.VIVACE,
    UnifiedTempo.PRESTO,

    # audio mode V5
    UnifiedMode.EMPTY,
    UnifiedMode.MAJOR,
    UnifiedMode.MINOR,
    UnifiedMode.GREGORIAN,
    UnifiedMode.PENTATONIC,
    UnifiedMode.BLUES,
    UnifiedMode.NATURAL_MAJOR,
    UnifiedMode.HARMONIC_MAJOR,
    UnifiedMode.MELODIC_MAJOR,
    UnifiedMode.NATURAK_MINOR,
    UnifiedMode.HARMONIC_MINOR,
    UnifiedMode.MELODIC_MINOR,
    UnifiedMode.LYDIAN,
    UnifiedMode.MIXOLYDIAN,
    UnifiedMode.DORIAN,
    UnifiedMode.PHRYGIAN,
    UnifiedMode.BLUES_MAJOR,
    UnifiedMode.BLUES_MINOR,
    UnifiedMode.BLUES_COMBINED,
    UnifiedMode.GONG,
    UnifiedMode.SHANG,
    UnifiedMode.YU,
    UnifiedMode.ZHI,
    UnifiedMode.JUE,
    UnifiedMode.DUJIE,

    # audio key V5
    UnifiedKey.EMPTY,
    UnifiedKey.A,
    UnifiedKey.A_SHARP,
    #UnifiedKey.Ab,
    UnifiedKey.B,
    #UnifiedKey.Bb,
    UnifiedKey.C,
    UnifiedKey.C_SHARP,
    UnifiedKey.Cb,
    UnifiedKey.D,
    UnifiedKey.D_SHARP,
    #UnifiedKey.Db,
    UnifiedKey.E,
    #UnifiedKey.Eb,
    UnifiedKey.F,
    UnifiedKey.F_SHARP,
    UnifiedKey.G,
    UnifiedKey.G_SHARP,
    #UnifiedKey.Gb,

    # other V5
    UnifiedGenre.OTHER,
    UnifiedExtra.NON_NOSTALGIC,
    UnifiedExtra.NON_CANNED_MUSIC,
    UnifiedExtra.GRASSROOTS,
    UnifiedExtra.SQUARE_DANCE,
    UnifiedExtra.SENIORS_GRASSROOTS,
    UnifiedExtra.HITS_GRASSROOTS,
    UnifiedExtra.YOUNG_GRASSROOTS,
    UnifiedMood.OTHER,
    UnifiedMood.NO_MOOD,
    UnifiedTheme.OTHER,
    UnifiedGender.NONVOCAL_NONGENDER,
    UnifiedTimbre.NONVOCAL_NONTIMBRE,
    UnifiedLang.OTHER,
    UnifiedInstrument.OTHER,
    UnifiedInstrument.NONVOCAL,
    UnifiedMode.OTHER,
    UnifiedLang.PORTUGUESE,
    UnifiedLang.SPANISH,
    UnifiedLang.INDONESIAN,
    UnifiedLang.THAI,
    UnifiedLang.VIETNAMESE,
    UnifiedExtra.NIGHT_LOVING_SCENE,
    UnifiedExtra.ACG,
    UnifiedExtra.ANIME_OPENING_ENDING,
    UnifiedMode.LOCRIAN,
    UnifiedGenre.SCI_FI,
    UnifiedGenre.MIDDLE_AGES,
    UnifiedGenre.RENAISSANCE,
    UnifiedExtra.MUSICAL_THEATER,
    UnifiedExtra.OPERA,
    UnifiedExtra.SYMPHONY,
    UnifiedExtra.SONATA,
    UnifiedExtra.CONCERTO,
    UnifiedExtra.FUGUE,
    UnifiedExtra.WALTZ,
    UnifiedExtra.CHAMBER_MUSIC,
    UnifiedExtra.SOLO,
    UnifiedMood.BRISK_CAREFREE,
    UnifiedMood.WARM_KIND,
    UnifiedMood.CONFIDENT_DETERMINED,
    UnifiedMood.COOL_SWAG,
    UnifiedMood.ELEGANT_SOPHISTICATED,
    UnifiedMood.SERIOUS_REFLECTIVE,
    UnifiedMood.LYRICAL_BALLAD,
    UnifiedMood.MOVING,
    UnifiedMood.MAGICAL_FAIRYTALE,
    UnifiedMood.DARK,
    UnifiedInstrument.CLAVINET,
    UnifiedGenre.BIG_ROOM_HOUSE,
    UnifiedGenre.MIDTEMPO,
    UnifiedGenre.MELODIC_BASS,
    UnifiedGenre.COLOR_BASS,
    UnifiedGenre.MOOMBAHTON,
    UnifiedTheme.NATURE,
    UnifiedTheme.NATIONAL_PRIDE,
    UnifiedTheme.SELF_AND_GROWTH,
    UnifiedTheme.CELEBRATION_AND_JOY,
    UnifiedTheme.LIFE_AND_MINDSET,
    UnifiedTheme.DREAM_AND_FUTURE,
    UnifiedTheme.MATSURI,
    UnifiedTheme.SUNRISE_DAWN,
    UnifiedTheme.SUNSET_AFTERNOON,
    UnifiedTheme.MIDNIGHT,
    UnifiedTheme.CLOUDY_DAY,
    UnifiedTheme.FOG,
    UnifiedTheme.STORM,
    UnifiedTheme.SKY_HEAVEN_PARADISE,
    UnifiedTheme.OCEAN_SEA,
    UnifiedTheme.FOREST,
    UnifiedTheme.OUTDOOR,
    UnifiedTheme.TEA_ROOM,
    UnifiedTheme.UNDERGOUND,
    UnifiedTheme.CIRCUS_CARNIVAL,
    UnifiedTheme.TEMPLE,
    UnifiedTheme.CHURCH,
    UnifiedTheme.COUNTRYSIDE,
    UnifiedTheme.EMOTIONAL,
    UnifiedTheme.TECHNOLOGY_SCIENCE,
    UnifiedTheme.HIGH_ENERGY_VIDEO,
    UnifiedTheme.DOCUMENTARY,
    UnifiedTheme.DRAMA,
    UnifiedTheme.SUPERNATURAL,
    UnifiedTheme.COMMERCIAL,
    UnifiedTheme.ADVERTISEMENT,
    UnifiedTheme.SENIORS_VIDEO,
    UnifiedTheme.VIRAL_SHORT_VIDEO,
    UnifiedTheme.TRAILER,
    UnifiedTheme.PATRIOTIC_VIDEO,
    UnifiedTheme.CULTURAL_EVENTS,
    UnifiedTheme.WAR_FIGHT,
    UnifiedTheme.ADVENTURE_DISCOVERY,
    UnifiedGenre.MELODIC_HARDCORE,
    UnifiedGenre.DUB,
    UnifiedGenre.CHA_CHA_CHA,
    UnifiedGenre.KAWAII_BASS,
    UnifiedGenre.JAPANESE_JAZZ_FUSION,
    UnifiedGenre.KAWAII_METAL,
    UnifiedGenre.AINU_FOLK,
    UnifiedGenre.OKINAWAN_POP,
    UnifiedInstrument.CELESTA,
    UnifiedExtra.JAPANESE_STYLE,
    UnifiedTimbre.ROUND,
    UnifiedTimbre.FLAT,
    UnifiedTimbre.GROWLING,
    UnifiedTimbre.NASAL_VOICE,
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
        "Coffee Shop": UnifiedTheme.COFFEE_SHOP,
        "Campus": UnifiedTheme.CAMPUS,
        "Christmas": UnifiedTheme.CHRISTMAS,
        "Dance": UnifiedTheme.DANCE,
        "Danceable": UnifiedTheme.DANCEABLE,
        "Date": UnifiedTheme.DATE,
        "Dating": UnifiedTheme.DATE,
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
        "Valentine's day": UnifiedTheme.VALENTINES_DAY,
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

CATEGORY_MAP_AUDIO_V3 = {
    UnifiedCategory.GENRE: {
       "Pop": UnifiedGenre.POP,
       "Electronic": UnifiedGenre.ELECTRONIC,
       "Chinese Style": UnifiedGenre.CHINESE_STYLE,
       "Rock": UnifiedGenre.ROCK,
       "Jazz": UnifiedGenre.JAZZ,
       "Hip Hop": UnifiedGenre.HIP_HOP,
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
       "Funk": UnifiedGenre.FUNK,
       "Contemporary R&B": UnifiedGenre.CONTEMPORARY_RNB,
       "Neo Soul": UnifiedGenre.NEO_SOUL,
       "Soul": UnifiedGenre.SOUL,
       "Pop Soul": UnifiedGenre.POP_SOUL,
       "Black Metal": UnifiedGenre.BLACK_METAL,
       "Death Metal": UnifiedGenre.DEATH_METAL,
       "Glam Metal": UnifiedGenre.GLAM_METAL,
       "Glam Metal（Hair Metal，Pop Metal）": UnifiedGenre.GLAM_METAL,
       "Glam Metal\n（Hair Metal，Pop Metal）": UnifiedGenre.GLAM_METAL,
       "Grindcore": UnifiedGenre.GRINDCORE,
       "Power Metal": UnifiedGenre.POWER_METAL,
       "Progressive Metal": UnifiedGenre.PROGRESSIVE_METAL,
       "Progressive Metal（Prog Metal)": UnifiedGenre.PROGRESSIVE_METAL,
       "Progressive Metal\n（Prog Metal)": UnifiedGenre.PROGRESSIVE_METAL,
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
       "Valentine's Day": UnifiedTheme.VALENTINES_DAY,
       "Qi Xi": UnifiedTheme.QI_XI,
       "Birthday": UnifiedTheme.BIRTHDAY,
       "Wedding": UnifiedTheme.WEDDING,
       "Funeral": UnifiedTheme.FUNERAL,
       "Graduation": UnifiedTheme.GRADUATION,
       "National's Day ": UnifiedTheme.NATIONALS_DAY,
       "National's Day": UnifiedTheme.NATIONALS_DAY,
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
       "Anime": UnifiedTheme.ANIME,
       "Wake up": UnifiedTheme.WAKE_UP,
       "Family time": UnifiedTheme.FAMILY_TIME,
       "Landscape/Scenery": UnifiedTheme.LANDSCAPE_SCENERY,
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
       "Cafe": UnifiedTheme.COFFEE_SHOP,
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
       "Unkonwn": UnifiedGender.EMPTY,
       "Unknown": UnifiedGender.EMPTY,
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
        "Hip Hop": UnifiedGenre.HIP_HOP,
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
        "Funk": UnifiedGenre.FUNK,
        "Contemporary R&B": UnifiedGenre.CONTEMPORARY_RNB,
        "Neo Soul": UnifiedGenre.NEO_SOUL,
        "Soul": UnifiedGenre.SOUL,
        "Pop Soul": UnifiedGenre.POP_SOUL,
        "Black Metal": UnifiedGenre.BLACK_METAL,
        "Death Metal": UnifiedGenre.DEATH_METAL,
        "Glam Metal": UnifiedGenre.GLAM_METAL,
        "Glam Metal（Hair Metal，Pop Metal）": UnifiedGenre.GLAM_METAL,
        "Glam Metal\n（Hair Metal，Pop Metal）": UnifiedGenre.GLAM_METAL,
        "Grindcore": UnifiedGenre.GRINDCORE,
        "Power Metal": UnifiedGenre.POWER_METAL,
        "Progressive Metal": UnifiedGenre.PROGRESSIVE_METAL,
        "Progressive Metal（Prog Metal)": UnifiedGenre.PROGRESSIVE_METAL,
        "Progressive Metal\n（Prog Metal)": UnifiedGenre.PROGRESSIVE_METAL,
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
        "Glam Metal": UnifiedGenre.GLAM_METAL,
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
        "New Wave": UnifiedGenre.NEW_WAVE,
        "No Wave": UnifiedGenre.NO_WAVE,
        "Nu Metal": UnifiedGenre.NU_METAL,
        "Post-hardcore": UnifiedGenre.POST_HARDCORE,
        "Post-Punk": UnifiedGenre.POST_PUNK,
        "Progressive Metal": UnifiedGenre.PROGRESSIVE_METAL,
        "Progressive Metal（Prog Metal)": UnifiedGenre.PROGRESSIVE_METAL,
        "Progressive Metal\n（Prog Metal)": UnifiedGenre.PROGRESSIVE_METAL,
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
        "Valentine's Day": UnifiedTheme.VALENTINES_DAY,
        "Qi Xi": UnifiedTheme.QI_XI,
        "Birthday": UnifiedTheme.BIRTHDAY,
        "Wedding": UnifiedTheme.WEDDING,
        "Funeral": UnifiedTheme.FUNERAL,
        "Graduation": UnifiedTheme.GRADUATION,
        "National's Day ": UnifiedTheme.NATIONALS_DAY,
        "National's Day": UnifiedTheme.NATIONALS_DAY,
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
        "Anime": UnifiedTheme.ANIME,
        "Wake up": UnifiedTheme.WAKE_UP,
        "Family time": UnifiedTheme.FAMILY_TIME,
        "Landscape/Scenery": UnifiedTheme.LANDSCAPE_SCENERY,
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
        "Cafe": UnifiedTheme.COFFEE_SHOP,
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
        "Unkonwn": UnifiedGender.EMPTY,
        "Unknown": UnifiedGender.EMPTY,
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

VOCAB2ID_AUDIO_V4 = Vocab2Id(
    CATEGORY_MAP_AUDIO_V4,
    UNIFIED_VOCAB2ID_LEGACY,
)

CATEGORY_MAP_AUDIO_V5 = {
    UnifiedCategory.GENRE: {
        "Other_AUDIO_GENRE": UnifiedGenre.OTHER,
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
        "Funk": UnifiedGenre.FUNK,
        "Contemporary R&B": UnifiedGenre.CONTEMPORARY_RNB,
        "Neo Soul": UnifiedGenre.NEO_SOUL,
        "Soul": UnifiedGenre.SOUL,
        "Pop Soul": UnifiedGenre.POP_SOUL,
        "Black Metal": UnifiedGenre.BLACK_METAL,
        "Death Metal": UnifiedGenre.DEATH_METAL,
        "Glam Metal": UnifiedGenre.GLAM_METAL,
        "Glam Metal（Hair Metal，Pop Metal）": UnifiedGenre.GLAM_METAL,
        "Glam Metal\n（Hair Metal，Pop Metal）": UnifiedGenre.GLAM_METAL,
        "Grindcore": UnifiedGenre.GRINDCORE,
        "Power Metal": UnifiedGenre.POWER_METAL,
        "Progressive Metal": UnifiedGenre.PROGRESSIVE_METAL,
        "Progressive Metal（Prog Metal)": UnifiedGenre.PROGRESSIVE_METAL,
        "Progressive Metal\n（Prog Metal)": UnifiedGenre.PROGRESSIVE_METAL,
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
        "Glam Metal": UnifiedGenre.GLAM_METAL,
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
        "New Wave": UnifiedGenre.NEW_WAVE,
        "No Wave": UnifiedGenre.NO_WAVE,
        "Nu Metal": UnifiedGenre.NU_METAL,
        "Post-hardcore": UnifiedGenre.POST_HARDCORE,
        "Post-Punk": UnifiedGenre.POST_PUNK,
        "Progressive Metal": UnifiedGenre.PROGRESSIVE_METAL,
        "Progressive Metal（Prog Metal)": UnifiedGenre.PROGRESSIVE_METAL,
        "Progressive Metal\n（Prog Metal)": UnifiedGenre.PROGRESSIVE_METAL,
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
        # audio genre V5
        "Jersey Club": UnifiedGenre.JERSEY_CLUB,
        "UK Garage": UnifiedGenre.UK_GARAGE,
        "Chinoiserie Rock": UnifiedGenre.CHINOISERIE_ROCK,
        "Modern Pop Ballad": UnifiedGenre.MODERN_POP_BALLAD,
        "Celtic Folk": UnifiedGenre.CELTIC_FOLK,
        "Future Bounce": UnifiedGenre.FUTURE_BOUNCE,
        "Phonk": UnifiedGenre.PHONK,
        "Early Country": UnifiedGenre.EARLY_COUNTRY,
        "Afrobeats": UnifiedGenre.AFROBEATS,
        "empty genre": UnifiedGenre.EMPTY,
        "Sci-Fi": UnifiedGenre.SCI_FI,
        "Middle Ages": UnifiedGenre.MIDDLE_AGES,
        "Renaissance": UnifiedGenre.RENAISSANCE,
        "Big Room House": UnifiedGenre.BIG_ROOM_HOUSE,
        "Midtempo": UnifiedGenre.MIDTEMPO,
        "Melodic Bass": UnifiedGenre.MELODIC_BASS,
        "Color Bass": UnifiedGenre.COLOR_BASS,
        "Moombahton": UnifiedGenre.MOOMBAHTON,
        "Melodic Hardcore": UnifiedGenre.MELODIC_HARDCORE,
        "Dub": UnifiedGenre.DUB,
        "Cha-Cha-Cha": UnifiedGenre.CHA_CHA_CHA,
        "Kawaii Bass": UnifiedGenre.KAWAII_BASS,
        "Japanese Jazz Fusion": UnifiedGenre.JAPANESE_JAZZ_FUSION,
        "Kawaii Metal": UnifiedGenre.KAWAII_METAL,
        "Ainu Folk": UnifiedGenre.AINU_FOLK,
        "Okinawan Pop": UnifiedGenre.OKINAWAN_POP,
    },
    UnifiedCategory.MOOD: {
        "Other_AUDIO_MOOD": UnifiedMood.OTHER,
        "No Mood": UnifiedMood.NO_MOOD,
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
        "Magnificent/Epic": UnifiedMood.SHOCKING_MAGNIFICENT_EPIC,
        "Shocking/magnificent/epic": UnifiedMood.SHOCKING_MAGNIFICENT_EPIC,
        "Mysterious": UnifiedMood.MYSTERIOUS,
        # audio mood V5
        "empty mood": UnifiedMood.EMPTY,
        "Brisk/Carefree": UnifiedMood.BRISK_CAREFREE,
        "Warm/Kind": UnifiedMood.WARM_KIND,
        "Confident/Determined": UnifiedMood.CONFIDENT_DETERMINED,
        "Cool/Swag": UnifiedMood.COOL_SWAG,
        "Elegant/Sophisticated": UnifiedMood.ELEGANT_SOPHISTICATED,
        "Serious/Reflective": UnifiedMood.SERIOUS_REFLECTIVE,
        "Lyrical/Ballad": UnifiedMood.LYRICAL_BALLAD,
        "Moving": UnifiedMood.MOVING,
        "Magical/Fairytale": UnifiedMood.MAGICAL_FAIRYTALE,
        "Dark": UnifiedMood.DARK,
    },
    UnifiedCategory.THEME: {
        "Other scene": UnifiedTheme.OTHER,
        "Halloween": UnifiedTheme.HALLOWEEN,
        "Christmas": UnifiedTheme.CHRISTMAS,
        "New Year": UnifiedTheme.NEW_YEAR,
        "Spring Festival": UnifiedTheme.SPRING_FESTIVAL,
        "Valentine's day": UnifiedTheme.VALENTINES_DAY,
        "Valentine's Day": UnifiedTheme.VALENTINES_DAY,
        "Qi Xi": UnifiedTheme.QI_XI,
        "Birthday": UnifiedTheme.BIRTHDAY,
        "Wedding": UnifiedTheme.WEDDING,
        "Funeral": UnifiedTheme.FUNERAL,
        "Graduation": UnifiedTheme.GRADUATION,
        "National's Day ": UnifiedTheme.NATIONALS_DAY,
        "National's Day": UnifiedTheme.NATIONALS_DAY,
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
        "Anime": UnifiedTheme.ANIME,
        "Wake up": UnifiedTheme.WAKE_UP,
        "Family time": UnifiedTheme.FAMILY_TIME,
        "Landscape/Scenery": UnifiedTheme.LANDSCAPE_SCENERY,
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
        "Cafe": UnifiedTheme.COFFEE_SHOP,
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
        "Mid-autumn Festival": UnifiedTheme.MID_AUTUMN_FESTIVAL,
        # audio theme V5
        "Livehouse": UnifiedTheme.LIVEHOUSE,
        "Music Festival": UnifiedTheme.MUSIC_FESTIVAL,
        "empty scene": UnifiedTheme.EMPTY,
        "Nature": UnifiedTheme.NATURE,
        "National Pride": UnifiedTheme.NATIONAL_PRIDE,
        "Self and Growth": UnifiedTheme.SELF_AND_GROWTH,
        "Celebration and Joy": UnifiedTheme.CELEBRATION_AND_JOY,
        "Life and Mindset": UnifiedTheme.LIFE_AND_MINDSET,
        "Dream and Future": UnifiedTheme.DREAM_AND_FUTURE,
        "Matsuri": UnifiedTheme.MATSURI,
        "Sunrise/Dawn": UnifiedTheme.SUNRISE_DAWN,
        "Sunset/Afternoon": UnifiedTheme.SUNSET_AFTERNOON,
        "Midnight": UnifiedTheme.MIDNIGHT,
        "Cloudy Day": UnifiedTheme.CLOUDY_DAY,
        "Fog": UnifiedTheme.FOG,
        "Storm": UnifiedTheme.STORM,
        "Sky/Heaven/Paradise": UnifiedTheme.SKY_HEAVEN_PARADISE,
        "Ocean/Sea": UnifiedTheme.OCEAN_SEA,
        "Forest": UnifiedTheme.FOREST,
        "Outdoor": UnifiedTheme.OUTDOOR,
        "Tea Room": UnifiedTheme.TEA_ROOM,
        "Undergound": UnifiedTheme.UNDERGOUND,
        "Circus/Carnival": UnifiedTheme.CIRCUS_CARNIVAL,
        "Temple": UnifiedTheme.TEMPLE,
        "Church": UnifiedTheme.CHURCH,
        "Countryside": UnifiedTheme.COUNTRYSIDE,
        "Emotional": UnifiedTheme.EMOTIONAL,
        "Technology/Science": UnifiedTheme.TECHNOLOGY_SCIENCE,
        "High-energy video": UnifiedTheme.HIGH_ENERGY_VIDEO,
        "Documentary": UnifiedTheme.DOCUMENTARY,
        "Drama": UnifiedTheme.DRAMA,
        "Supernatural": UnifiedTheme.SUPERNATURAL,
        "Commercial": UnifiedTheme.COMMERCIAL,
        "Advertisement": UnifiedTheme.ADVERTISEMENT,
        "Seniors video": UnifiedTheme.SENIORS_VIDEO,
        "Viral short video": UnifiedTheme.VIRAL_SHORT_VIDEO,
        "Trailer": UnifiedTheme.TRAILER,
        "Patriotic Video": UnifiedTheme.PATRIOTIC_VIDEO,
        "Cultural events": UnifiedTheme.CULTURAL_EVENTS,
        "War/Fight": UnifiedTheme.WAR_FIGHT,
        "Adventure/Discovery": UnifiedTheme.ADVENTURE_DISCOVERY,
    },
    UnifiedCategory.GENDER: {
        "Unkonwn": UnifiedGender.EMPTY,
        "Unknown": UnifiedGender.EMPTY,
        "Female": UnifiedGender.FEMALE,
        "female": UnifiedGender.FEMALE,
        "Male": UnifiedGender.MALE,
        "male": UnifiedGender.MALE,
        "Neutral": UnifiedGender.NEUTRAL,
        "Child": UnifiedGender.CHILD,
        "child": UnifiedGender.CHILD,
        "Adult": UnifiedGender.ADULT,
        "adult": UnifiedGender.ADULT,
        "Chorus_AUDIO_GENDER": UnifiedGender.CHORUS,  # dedup
        # audio tags ver2
        "Multiple": UnifiedGender.MULTIPLE,
        "empty gender": UnifiedGender.EMPTY,
        "nonvocal nongender": UnifiedGender.NONVOCAL_NONGENDER,
    },
    UnifiedCategory.TIMBRE: {
        "Other Timbre": UnifiedTimbre.EMPTY,
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
        # audio timbre V5
        "Breathy voice": UnifiedTimbre.BREATHY_VOICE,
        "ACG voice": UnifiedTimbre.ACG_VOICE,
        "Bel Canto": UnifiedTimbre.BEL_CANTO,
        "Chinese_Folk_AUDIO_TIMBRE": UnifiedTimbre.CHINESE_FOLK,
        "Gentle": UnifiedTimbre.GENTLE,
        "Delicate": UnifiedTimbre.DELICATE,
        "Energetic": UnifiedTimbre.ENERGETIC,
        "Sassy": UnifiedTimbre.SASSY,
        "Refreshing": UnifiedTimbre.REFRESHING,
        "Soothing": UnifiedTimbre.SOOTHING,
        "Comical": UnifiedTimbre.COMICAL,
        "Uncle Like": UnifiedTimbre.UNCLE_LIKE,
        "Gritty": UnifiedTimbre.GRITTY,
        "Romantic_AUDIO_TIMBRE": UnifiedTimbre.ROMANTIC,
        "Enthusiastic": UnifiedTimbre.ENTHUSIASTIC,
        "Smoky": UnifiedTimbre.SMOKY,
        "Cutesy&Squeaky Voice": UnifiedTimbre.CHILDLIKE,
        "Childlike": UnifiedTimbre.CHILDLIKE,
        "Chinese Opera Tune": UnifiedTimbre.OPERA_TUNE,
        "Opera Tune": UnifiedTimbre.OPERA_TUNE,
        "empty timbre": UnifiedTimbre.EMPTY,
        "nonvocal nontimbre": UnifiedTimbre.NONVOCAL_NONTIMBRE,
        "Round": UnifiedTimbre.ROUND,
        "Flat": UnifiedTimbre.FLAT,
        "Growling": UnifiedTimbre.GROWLING,
        "Nasal Voice": UnifiedTimbre.NASAL_VOICE,
    },
    UnifiedCategory.LANG: {
        "Other lang": UnifiedLang.OTHER,
        "Other Lang": UnifiedLang.OTHER,
        "Cantonese": UnifiedLang.CANTONESE,
        "canto": UnifiedLang.CANTONESE,
        "Chinese": UnifiedLang.CHINESE,
        "Chinese Dialects": UnifiedLang.CHINESE_DIALECT,
        "English": UnifiedLang.ENGLISH,
        # audio tags ver2
        "African": UnifiedLang.AFRICAN,
        "Arabic": UnifiedLang.ARABIC,
        "French": UnifiedLang.FRENCH,
        "German": UnifiedLang.GERMAN,
        "Hindi": UnifiedLang.HINDI,
        "indian": UnifiedLang.HINDI,
        "Instrumental/Non-vocal": UnifiedLang.INSTRUMENTAL,
        "Non-vocal": UnifiedLang.INSTRUMENTAL,
        "Italian": UnifiedLang.ITALIAN,
        "Japanese": UnifiedLang.JAPANESE,
        "Korea": UnifiedLang.KOREA,
        "Russian": UnifiedLang.RUSSIAN,
        "Sichuanese": UnifiedLang.SICHUANESE,
        "Taiwanese": UnifiedLang.TAIWANESE,
        "Turkish": UnifiedLang.TURKISH,
        # audio tags V5
        "empty lang": UnifiedLang.EMPTY,
        "Portuguese": UnifiedLang.PORTUGUESE,
        "Spanish": UnifiedLang.SPANISH,
        "Indonesian": UnifiedLang.INDONESIAN,
        "Thai": UnifiedLang.THAI,
        "Vietnamese": UnifiedLang.VIETNAMESE,
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
        # audio extra V5
        "Experimental": UnifiedExtra.EXPERIMENTAL,
        "Fashionable": UnifiedExtra.FASHIONABLE,
        "Canned Music": UnifiedExtra.CANNED_MUSIC,
        "empty extra": UnifiedExtra.EMPTY,
        "non-nostalgic": UnifiedExtra.NON_NOSTALGIC,
        "non-canned music": UnifiedExtra.NON_CANNED_MUSIC,
        "Grassroots": UnifiedExtra.GRASSROOTS,
        "Square Dance": UnifiedExtra.SQUARE_DANCE,
        "Seniors Grassroots": UnifiedExtra.SENIORS_GRASSROOTS,
        "Trends Grassroots": UnifiedExtra.HITS_GRASSROOTS,
        "Hits Grassroots": UnifiedExtra.HITS_GRASSROOTS,
        "Young Grassroots": UnifiedExtra.YOUNG_GRASSROOTS,
        "Night-Loving Scene": UnifiedExtra.NIGHT_LOVING_SCENE,
        "ACG": UnifiedExtra.ACG,
        "Anime Opening/Ending": UnifiedExtra.ANIME_OPENING_ENDING,
        "Musical Theater": UnifiedExtra.MUSICAL_THEATER,
        "Opera": UnifiedExtra.OPERA,
        "Symphony": UnifiedExtra.SYMPHONY,
        "Sonata": UnifiedExtra.SONATA,
        "Concerto": UnifiedExtra.CONCERTO,
        "Fugue": UnifiedExtra.FUGUE,
        "Waltz": UnifiedExtra.WALTZ,
        "Chamber Music": UnifiedExtra.CHAMBER_MUSIC,
        "Solo": UnifiedExtra.SOLO,
        "Japanese Style": UnifiedExtra.JAPANESE_STYLE,
    },
    UnifiedCategory.INSTRUMENT: {
        "Other Inst": UnifiedInstrument.OTHER,
        "Woodwinds": UnifiedInstrument.WOODWINDS,
        "Piccolo": UnifiedInstrument.PICCOLO,
        "Flute": UnifiedInstrument.FLUTE,
        "Clarinet": UnifiedInstrument.CLARINET,
        "Oboe": UnifiedInstrument.OBOE,
        "English_Horn": UnifiedInstrument.ENGLISH_HORN,
        "Pipe": UnifiedInstrument.PIPE,
        "Saxophone": UnifiedInstrument.SAXOPHONE,
        "Soprano/Alto_Sax": UnifiedInstrument.SOPRANO_ALTO_SAX,
        "Tenor_Sax": UnifiedInstrument.TENOR_SAX,
        "Baritone_Sax": UnifiedInstrument.BARITONE_SAX,
        "Bassoon": UnifiedInstrument.BASSOON,
        "Brass": UnifiedInstrument.BRASS,
        "Trumpet": UnifiedInstrument.TRUMPET,
        "French_Horn": UnifiedInstrument.FRENCH_HORN,
        "Trombone": UnifiedInstrument.TROMBONE,
        "Tuba": UnifiedInstrument.TUBA,
        "Brass_Section": UnifiedInstrument.BRASS_SECTION,
        "Synth_Brass": UnifiedInstrument.SYNTH_BRASS,
        "Percussion": UnifiedInstrument.PERCUSSION,
        "Percussions": UnifiedInstrument.PERCUSSION,
        "Drums": UnifiedInstrument.DRUMS,
        "Drum_Set": UnifiedInstrument.DRUMS,    # merge Drums, Drum_Set
        # "Drum_set": UnifiedInstrument.DRUMS,
        "Bassdrum": UnifiedInstrument.BASSDRUM,
        "Bass_Drum": UnifiedInstrument.BASSDRUM,
        "Snare": UnifiedInstrument.SNARE,
        "Hi_hat": UnifiedInstrument.HI_HAT,
        "Tom": UnifiedInstrument.TOM,
        "Tom_Tom": UnifiedInstrument.TOM,
        "Cymbals": UnifiedInstrument.CYMBALS,
        "Crash_Cymbal": UnifiedInstrument.CYMBALS,
        "Chromatic_Percussion": UnifiedInstrument.CHROMATIC_PERCUSSION,
        "Marimba": UnifiedInstrument.MARIMBA,
        "Bells": UnifiedInstrument.BELLS,
        "Belltree": UnifiedInstrument.BELLTREE,
        "Xylophone": UnifiedInstrument.XYLOPHONE,
        "Glockenspiel": UnifiedInstrument.GLOCKENSPIEL,
        "Vibraphone": UnifiedInstrument.VIBRAPHONE,
        "Chimes": UnifiedInstrument.CHIMES,
        "Tubular_Bells ": UnifiedInstrument.CHIMES,
        "Tubular_Bells": UnifiedInstrument.CHIMES,
        "Percussive": UnifiedInstrument.PERCUSSIVE,
        "Sandhammer": UnifiedInstrument.SANDHAMMER,
        "Tambourine": UnifiedInstrument.TAMBOURINE,
        "Timpani": UnifiedInstrument.TIMPANI,
        "Synth_Drums": UnifiedInstrument.SYNTH_DRUMS,
        "Synth_Kick": UnifiedInstrument.SYNTH_KICK,
        "Synth_Snare": UnifiedInstrument.SYNTH_SNARE,
        "Synth_Hi_hat": UnifiedInstrument.SYNTH_HI_HAT,
        "Synth_Tom": UnifiedInstrument.SYNTH_TOM,
        "Synth_Cymbals": UnifiedInstrument.SYNTH_CYMBALS,
        "Synth_Clave": UnifiedInstrument.SYNTH_CLAVE,
        "Fx": UnifiedInstrument.FX,
        "Clap": UnifiedInstrument.CLAP,
        "Keys": UnifiedInstrument.KEYS,
        "Acoustic_Piano": UnifiedInstrument.ACOUSTIC_PIANO,
        "Acousticpiano": UnifiedInstrument.ACOUSTIC_PIANO,
        "Electric_Piano": UnifiedInstrument.ELECTRIC_PIANO,
        "Electric_piano": UnifiedInstrument.ELECTRIC_PIANO,
        "Organ": UnifiedInstrument.ORGAN,
        "Accordion": UnifiedInstrument.ACCORDION,
        "Strings": UnifiedInstrument.STRINGS,
        "Violin": UnifiedInstrument.VIOLIN,
        "Viola": UnifiedInstrument.VIOLA,
        "Cello": UnifiedInstrument.CELLO,
        "Contrabass": UnifiedInstrument.CONTRABASS,
        "Double_Bass": UnifiedInstrument.CONTRABASS,
        "String_Ensemble": UnifiedInstrument.STRING_ENSEMBLE,
        "Synth_Strings": UnifiedInstrument.SYNTH_STRINGS,
        "Bass": UnifiedInstrument.BASS,
        "Electric_Bass": UnifiedInstrument.ELECTRIC_BASS,
        "Synth_Bass": UnifiedInstrument.SYNTH_BASS,
        "Guitar": UnifiedInstrument.GUITAR,
        "Acoustic_Guitar": UnifiedInstrument.ACOUSTIC_GUITAR,
        "Electric_Guitar": UnifiedInstrument.ELECTRIC_GUITAR,
        "Clean_Electric_Guitar": UnifiedInstrument.CLEAN_ELECTRIC_GUITAR,
        "Distorted_Electric_Guitar": UnifiedInstrument.DISTORTED_ELECTRIC_GUITAR,
        "Chinese_Traditional_Instruments": UnifiedInstrument.CHINESE_TRADITIONAL_INSTRUMENTS,
        "Chinese Traditional Instruments": UnifiedInstrument.CHINESE_TRADITIONAL_INSTRUMENTS,
        "Di": UnifiedInstrument.DI,
        "Xiao": UnifiedInstrument.XIAO,
        "Suona": UnifiedInstrument.SUONA,
        "Erhu": UnifiedInstrument.ERHU,
        "Guzheng": UnifiedInstrument.GUZHENG,
        "Pipa": UnifiedInstrument.PIPA,
        "Yangqin": UnifiedInstrument.YANGQIN,
        "Sheng": UnifiedInstrument.SHENG,
        "Hulusi": UnifiedInstrument.HULUSI,
        "Panflute": UnifiedInstrument.PANFLUTE,
        "Xun": UnifiedInstrument.XUN,
        "Matouqin": UnifiedInstrument.MATOUQIN,
        "Ruan": UnifiedInstrument.RUAN,
        "Sanxian": UnifiedInstrument.SANXIAN,
        "Guqin": UnifiedInstrument.GUQIN,
        "Bianzhong": UnifiedInstrument.BIANZHONG,
        "Chinese_Drums": UnifiedInstrument.CHINESE_DRUMS,
        "Chinesedrums": UnifiedInstrument.CHINESE_DRUMS,
        "Synthesizers": UnifiedInstrument.SYNTHESIZERS,
        "Synth_Pluck": UnifiedInstrument.SYNTH_PLUCK,
        "Synth_Lead": UnifiedInstrument.SYNTH_LEAD,
        "Synth_Pad": UnifiedInstrument.SYNTH_PAD,
        "Synth_Effects": UnifiedInstrument.SYNTH_EFFECTS,
        "Plucked_Strings": UnifiedInstrument.PLUCKED_STRINGS,
        "Plucked Strings": UnifiedInstrument.PLUCKED_STRINGS,
        "Ukelele": UnifiedInstrument.UKELELE,
        "Orchestral_Harp": UnifiedInstrument.ORCHESTRAL_HARP,
        "Banjo": UnifiedInstrument.BANJO,
        "Mandolin": UnifiedInstrument.MANDOLIN,
        "Ethnic": UnifiedInstrument.ETHNICS,
        "Bagpipe": UnifiedInstrument.BAGPIPE,
        "Dulcimer": UnifiedInstrument.DULCIMER,
        "Hangdrum": UnifiedInstrument.HANGDRUM,
        "Harmonica": UnifiedInstrument.HARMONICA,
        "Irishwhistle": UnifiedInstrument.IRISHWHISTLE,
        "Ocarina": UnifiedInstrument.OCARINA,
        "Sitar": UnifiedInstrument.SITAR,
        "Whistle": UnifiedInstrument.WHISTLE,
        "Musicbox": UnifiedInstrument.MUSICBOX,
        "Sound_Effects": UnifiedInstrument.SOUND_EFFECTS,
        "Vocal": UnifiedInstrument.VOCALS,
        "Vocals": UnifiedInstrument.VOCALS,
        "Backingvocal": UnifiedInstrument.BACKINGVOCAL,
        "Backing_Vocals": UnifiedInstrument.BACKINGVOCAL,
        "Chorus": UnifiedInstrument.CHORUS,
        "Choir_and_Voice": UnifiedInstrument.CHOIR_AND_VOICE,
        "Vocal_Chops": UnifiedInstrument.VOCAL_CHOPS,
        "Ride_Cymbal": UnifiedInstrument.RIDE_CYMBAL,
        "Harpsichord": UnifiedInstrument.HARPSICHORD,
        "Pipe_Organ": UnifiedInstrument.PIPE_ORGAN,
        "Synth_Bell": UnifiedInstrument.SYNTH_BELL,
        "Double_Bass_Pizzicato": UnifiedInstrument.DOUBLE_BASS_PIZZICATO,
        "Church_Bells": UnifiedInstrument.CHURCH_BELLS,
        "Singing_Bowl": UnifiedInstrument.SINGING_BOWL,
        "Castanets": UnifiedInstrument.CASTANETS,
        "Triangle": UnifiedInstrument.TRIANGLE,
        "Claves": UnifiedInstrument.CLAVES,
        "Cowbell": UnifiedInstrument.COWBELL,
        "Mark_Tree": UnifiedInstrument.MARK_TREE,
        "Congas/Bongos": UnifiedInstrument.CONGAS_BONGOS,
        "Cajón/Box_drum": UnifiedInstrument.CAJON_BOX_DRUM,
        "Wind_Chimes": UnifiedInstrument.WIND_CHIMES,
        "Orchestral_Drums": UnifiedInstrument.ORCHESTRAL_DRUMS,
        "Orchestral_Bassdrum": UnifiedInstrument.ORCHESTRAL_BASSDRUM,
        "Orchestral_Snare": UnifiedInstrument.ORCHESTRAL_SNARE,
        "Orchestral_Cymbals": UnifiedInstrument.ORCHESTRAL_CYMBALS,
        "Taiko_Drums": UnifiedInstrument.TAIKO_DRUMS,
        "Tam_Tam": UnifiedInstrument.TAM_TAM,
        "Body_Percussion": UnifiedInstrument.BODY_PERCUSSION,
        "Finger_snaps": UnifiedInstrument.FINGER_SNAPS,
        "Beat_box": UnifiedInstrument.BEAT_BOX,
        "Chinese_Percussion": UnifiedInstrument.CHINESE_PERCUSSION,
        "Gongs": UnifiedInstrument.GONGS,
        "Wood_Block": UnifiedInstrument.WOOD_BLOCK,
        "Tang_Drums": UnifiedInstrument.TANG_DRUMS,
        "Ban_Drums": UnifiedInstrument.BAN_DRUMS,
        "Chinese_Clappers": UnifiedInstrument.CHINESE_CLAPPERS,
        "Chinese_Cymbals": UnifiedInstrument.CHINESE_CYMBALS,
        "Yunluo": UnifiedInstrument.YUNLUO,
        "Bangzi": UnifiedInstrument.BANGZI,
        "Synth_Chord": UnifiedInstrument.SYNTH_CHORD,
        "empty instrument": UnifiedInstrument.EMPTY,
        "nonvocal": UnifiedInstrument.NONVOCAL,
        "Clavinet": UnifiedInstrument.CLAVINET,
        "Celesta": UnifiedInstrument.CELESTA,
    },
    UnifiedCategory.TEMPO: {
        "Grave": UnifiedTempo.GRAVE,
        "Largo": UnifiedTempo.LARGO,
        "Adagio": UnifiedTempo.ADAGIO,
        "Andante": UnifiedTempo.ANDANTE,
        "Moderato": UnifiedTempo.MODERATO,
        "Allegro": UnifiedTempo.ALLEGRO,
        "Vivace": UnifiedTempo.VIVACE,
        "Presto": UnifiedTempo.PRESTO,
        "empty tempo": UnifiedTempo.EMPTY,
    },
    UnifiedCategory.MODE: {
        "Other Mode": UnifiedMode.OTHER,
        "Major": UnifiedMode.MAJOR,
        "Minor": UnifiedMode.MINOR,
        "Maj": UnifiedMode.MAJOR,
        "Min": UnifiedMode.MINOR,
        "Gregorian": UnifiedMode.GREGORIAN,
        "Pentatonic": UnifiedMode.PENTATONIC,
        "Blues_AUDIO_MODE": UnifiedMode.BLUES,
        "Natural_Major": UnifiedMode.NATURAL_MAJOR,
        "Harmonic_Major": UnifiedMode.HARMONIC_MAJOR,
        "Melodic_Major": UnifiedMode.MELODIC_MAJOR,
        "Natural_Minor": UnifiedMode.NATURAK_MINOR,
        "Harmonic_Minor": UnifiedMode.HARMONIC_MINOR,
        "Melodic_Minor": UnifiedMode.MELODIC_MINOR,
        "Lydian": UnifiedMode.LYDIAN,
        "Mixolydian": UnifiedMode.MIXOLYDIAN,
        "Dorian": UnifiedMode.DORIAN,
        "Phrygian": UnifiedMode.PHRYGIAN,
        "Blues_Major": UnifiedMode.BLUES_MAJOR,
        "Blues_Minor": UnifiedMode.BLUES_MINOR,
        "Blues_Combined": UnifiedMode.BLUES_COMBINED,
        "Gong": UnifiedMode.GONG,
        "Shang": UnifiedMode.SHANG,
        "Yu": UnifiedMode.YU,
        "Zhi": UnifiedMode.ZHI,
        "Jue": UnifiedMode.JUE,
        "DuJie": UnifiedMode.DUJIE,
        "empty mode": UnifiedMode.EMPTY,
        "Locrian": UnifiedMode.LOCRIAN,
    },
    UnifiedCategory.KEY: {
        "A": UnifiedKey.A,
        "A#": UnifiedKey.A_SHARP,
        "Ab": UnifiedKey.G_SHARP,
        "B": UnifiedKey.B,
        "Bb": UnifiedKey.A_SHARP,
        "C": UnifiedKey.C,
        "C#": UnifiedKey.C_SHARP,
        "Cb": UnifiedKey.Cb,
        "D": UnifiedKey.D,
        "D#": UnifiedKey.D_SHARP,
        "Db": UnifiedKey.C_SHARP,
        "E": UnifiedKey.E,
        "Eb": UnifiedKey.D_SHARP,
        "F": UnifiedKey.F,
        "F#": UnifiedKey.F_SHARP,
        "G": UnifiedKey.G,
        "G#": UnifiedKey.G_SHARP,
        "Gb": UnifiedKey.F_SHARP,
        "empty key": UnifiedKey.EMPTY,
    }
}

VOCAB2ID_AUDIO_V5 = Vocab2Id(
    CATEGORY_MAP_AUDIO_V5,
    UNIFIED_VOCAB2ID_LEGACY,
)


CATEGORY_MAP_ARTIST_V1 = {
    UnifiedCategory.ARTIST: {
        "邓丽君": UnifiedArtist.DENGLIJUN,
        "刘德华": UnifiedArtist.LIUDEHUA,
        "张学友": UnifiedArtist.ZHANGXUEYOU,
        "莫文蔚": UnifiedArtist.MOWENWEI,
        "周杰伦": UnifiedArtist.ZHOUJIELUN,
        "林俊杰": UnifiedArtist.LINJUNJIE,
        "周深": UnifiedArtist.ZHOU_SHEN,
        "蔡依林": UnifiedArtist.CAIYILIN,
        "汪苏泷": UnifiedArtist.WANGSULONG,
        "蔡健雅": UnifiedArtist.CAIJIANYA,
        "音阙诗听": UnifiedArtist.YINQUESHITING,
        "张韶涵": UnifiedArtist.ZHANGSHAOHAN,
        "五月天": UnifiedArtist.WU_YUE_TIAN,
        "王心凌": UnifiedArtist.WANGXINLING,
        "方大同": UnifiedArtist.FANGDATONG,
        "s.h.e": UnifiedArtist.S_H_E,
        "孙燕姿": UnifiedArtist.SUNYANZI,
        "林宥嘉": UnifiedArtist.LINYOUJIA,
        "g.e.m. 邓紫棋": UnifiedArtist.G_E_M_DENGZIQI,
        "陶喆": UnifiedArtist.TAOZHE,
        "陈粒": UnifiedArtist.CHENLI,
        "苏打绿": UnifiedArtist.SUDALU,
        "李健": UnifiedArtist.LIJIAN,
        "李荣浩": UnifiedArtist.LIRONGHAO,
        "毛不易": UnifiedArtist.MAOBUYI,
        "vava毛衍七": UnifiedArtist.VAVA_MAOYANQI,
        "华晨宇": UnifiedArtist.HUACHENYU,
        "杨宗纬": UnifiedArtist.YANGZONGWEI,
        "王以太": UnifiedArtist.WANGYITAI,
        "eric周兴哲": UnifiedArtist.ERIC_ZHOUXINGZHE,
        "单依纯": UnifiedArtist.SHANYICHUN,
        "余佳运": UnifiedArtist.YUJIAYUN,
        "jony j": UnifiedArtist.JONY_J,
        "颜人中": UnifiedArtist.YANRENZHONG,
        "王忻辰": UnifiedArtist.WANGXINCHEN,
        "队长": UnifiedArtist.DUIZHANG,
        "lil ghost小鬼": UnifiedArtist.LIL_GHOST_XIAOGUI,
        "dave matthews band": UnifiedArtist.DAVE_MATTHEWS_BAND,
        "david bowie": UnifiedArtist.DAVID_BOWIE,
        "dj drama": UnifiedArtist.DJ_DRAMA,
        "manic street preachers": UnifiedArtist.MANIC_STREET_PREACHERS,
        "jimmy witherspoon": UnifiedArtist.JIMMY_WITHERSPOON,
        "frank sinatra": UnifiedArtist.FRANK_SINATRA,
        "bing crosby": UnifiedArtist.BING_CROSBY,
        "lightnin' hopkins": UnifiedArtist.LIGHTNIN_HOPKINS,
        "eric bellinger": UnifiedArtist.ERIC_BELLINGER,
        "jt music": UnifiedArtist.JT_MUSIC,
        "lowell fulson": UnifiedArtist.LOWELL_FULSON,
        "champion jack dupree": UnifiedArtist.CHAMPION_JACK_DUPREE,
        "dr. john": UnifiedArtist.DR_JOHN,
        "bruce springsteen": UnifiedArtist.BRUCE_SPRINGSTEEN,
        "shirley bassey": UnifiedArtist.SHIRLEY_BASSEY,
        "kylie minogue": UnifiedArtist.KYLIE_MINOGUE,
        "chris rea": UnifiedArtist.CHRIS_REA,
        "big maybelle": UnifiedArtist.BIG_MAYBELLE,
        "rosemary clooney": UnifiedArtist.ROSEMARY_CLOONEY,
        "nancy wilson": UnifiedArtist.NANCY_WILSON,
        "amos milburn": UnifiedArtist.AMOS_MILBURN,
        "little milton": UnifiedArtist.LITTLE_MILTON,
        "big bill broonzy": UnifiedArtist.BIG_BILL_BROONZY,
        "barenaked ladies": UnifiedArtist.BARENAKED_LADIES,
        "glenn miller": UnifiedArtist.GLENN_MILLER,
        "ray charles": UnifiedArtist.RAY_CHARLES,
        "mel tormé": UnifiedArtist.MEL_TORME,
        "madonna": UnifiedArtist.MADONNA,
        "buddy guy": UnifiedArtist.BUDDY_GUY,
        "beyoncé": UnifiedArtist.BEYONCE,
        "the goo goo dolls": UnifiedArtist.THE_GOO_GOO_DOLLS,
        "maria muldaur": UnifiedArtist.MARIA_MULDAUR,
        "switchfoot": UnifiedArtist.SWITCHFOOT,
        "everything but the girl": UnifiedArtist.EVERYTHING_BUT_THE_GIRL,
        "tina turner": UnifiedArtist.TINA_TURNER,
        "roy brown": UnifiedArtist.ROY_BROWN,
        "jimmy reed": UnifiedArtist.JIMMY_REED,
        "doris day": UnifiedArtist.DORIS_DAY,
        "bob marley & the wailers": UnifiedArtist.BOB_MARLEY_AND_THE_WAILERS,
        "r. kelly": UnifiedArtist.R_KELLY,
        "dinah washington": UnifiedArtist.DINAH_WASHINGTON,
        "lou reed": UnifiedArtist.LOU_REED,
        "sarah vaughan": UnifiedArtist.SARAH_VAUGHAN,
        "babyface ray": UnifiedArtist.BABYFACE_RAY,
        "linkin park": UnifiedArtist.LINKIN_PARK,
        "chris brown": UnifiedArtist.CHRIS_BROWN,
        "canned heat": UnifiedArtist.CANNED_HEAT,
        "tony bennett": UnifiedArtist.TONY_BENNETT,
        "simply red": UnifiedArtist.SIMPLY_RED,
        "junior wells": UnifiedArtist.JUNIOR_WELLS,
        "the black crowes": UnifiedArtist.THE_BLACK_CROWES,
        "trey songz": UnifiedArtist.TREY_SONGZ,
        "bonnie raitt": UnifiedArtist.BONNIE_RAITT,
        "moby": UnifiedArtist.MOBY,
        "depeche mode": UnifiedArtist.DEPECHE_MODE,
        "bobby darin": UnifiedArtist.BOBBY_DARIN,
        "norah jones": UnifiedArtist.NORAH_JONES,
        "phil collins": UnifiedArtist.PHIL_COLLINS,
        "judy garland": UnifiedArtist.JUDY_GARLAND,
        "björk": UnifiedArtist.BJORK,
        "screamin' jay hawkins": UnifiedArtist.SCREAMIN_JAY_HAWKINS,
        "nick cave & the bad seeds": UnifiedArtist.NICK_CAVE_AND_THE_BAD_SEEDS,
        "perry como": UnifiedArtist.PERRY_COMO,
        "electric six": UnifiedArtist.ELECTRIC_SIX,
        "miracle of sound": UnifiedArtist.MIRACLE_OF_SOUND,
        "lonnie johnson": UnifiedArtist.LONNIE_JOHNSON,
        "billie holiday": UnifiedArtist.BILLIE_HOLIDAY,
        "big mama thornton": UnifiedArtist.BIG_MAMA_THORNTON,
        "lost frequencies": UnifiedArtist.LOST_FREQUENCIES,
        "memphis slim": UnifiedArtist.MEMPHIS_SLIM,
        "albert king": UnifiedArtist.ALBERT_KING,
        "little mix": UnifiedArtist.LITTLE_MIX,
        "coldplay": UnifiedArtist.COLDPLAY,
        "john mayer": UnifiedArtist.JOHN_MAYER,
        "sonny boy williamson ii": UnifiedArtist.SONNY_BOY_WILLIAMSON_II,
        "primal scream": UnifiedArtist.PRIMAL_SCREAM,
        "chris isaak": UnifiedArtist.CHRIS_ISAAK,
        "diplo": UnifiedArtist.DIPLO,
        "reLIENT k": UnifiedArtist.RELIENT_K,
        "céline dion": UnifiedArtist.CELINE_DION,
        "iamx": UnifiedArtist.IAMX,
        "tom waits": UnifiedArtist.TOM_WAITS,
        "red hot chili peppers": UnifiedArtist.RED_HOT_CHILI_PEPPERS,
        "aerosmith": UnifiedArtist.AEROSMITH,
        "tony joe white": UnifiedArtist.TONY_JOE_WHITE,
        "garbage": UnifiedArtist.GARBAGE,
        "delbert mcclinton": UnifiedArtist.DELBERT_MCCLINTON,
        "blackbear": UnifiedArtist.BLACKBEAR,
        "popa chubby": UnifiedArtist.POPA_CHUBBY,
        "michael bublé": UnifiedArtist.MICHAEL_BUBLE,
        "usher": UnifiedArtist.USHER,
        "meghan trainor": UnifiedArtist.MEGHAN_TRAINOR,
        "p!nk": UnifiedArtist.PINK,
        "robert cray": UnifiedArtist.ROBERT_CRAY,
        "john mellencamp": UnifiedArtist.JOHN_MELLENCAMP,
        "goldfrapp": UnifiedArtist.GOLDFRAPP,
        "robbie williams": UnifiedArtist.ROBBIE_WILLIAMS,
        "hot tuna": UnifiedArtist.HOT_TUNA,
        "james blunt": UnifiedArtist.JAMES_BLUNT,
        "lil uzi vert": UnifiedArtist.LIL_UZI_VERT,
        "nina simone": UnifiedArtist.NINA_SIMONE,
        "eyedress": UnifiedArtist.EYEDRESS,
        "ella fitzgerald": UnifiedArtist.ELLA_FITZGERALD,
        "britney spears": UnifiedArtist.BRITNEY_SPEARS,
        "hot chip": UnifiedArtist.HOT_CHIP,
        "lights": UnifiedArtist.LIGHTS,
        "air supply": UnifiedArtist.AIR_SUPPLY,
        "dido": UnifiedArtist.DIDO,
        "tory lanez": UnifiedArtist.TORY_LANEZ,
        "feeder": UnifiedArtist.FEEDER,
        "neffex": UnifiedArtist.NEFFEX,
        "memphis minnie": UnifiedArtist.MEMPHIS_MINNIE,
        "groove armada": UnifiedArtist.GROOVE_ARMADA,
        "the chainsmokers": UnifiedArtist.THE_CHAINSMOKERS,
        "collective soul": UnifiedArtist.COLLECTIVE_SOUL,
        "inxs": UnifiedArtist.INXS,
        "james cotton": UnifiedArtist.JAMES_COTTON,
        "róisín murphy": UnifiedArtist.ROISIN_MURPHY,
        "taj mahal": UnifiedArtist.TAJ_MAHAL,
        "whitney houston": UnifiedArtist.WHITNEY_HOUSTON,
        "trapt": UnifiedArtist.TRAPT,
        "charlie musselwhite": UnifiedArtist.CHARLIE_MUSSELWHITE,
        "mike bloomfield": UnifiedArtist.MIKE_BLOOMFIELD,
        "chromeo": UnifiedArtist.CHROMEO,
        "jimmy rogers": UnifiedArtist.JIMMY_ROGERS,
        "craig david": UnifiedArtist.CRAIG_DAVID,
        "calvin harris": UnifiedArtist.CALVIN_HARRIS,
        "katie melua": UnifiedArtist.KATIE_MELUA,
        "creedence clearwater revival": UnifiedArtist.CREEDENCE_CLEARWATER_REVIVAL,
        "al bowlly": UnifiedArtist.AL_BOWLLY,
        "etta james": UnifiedArtist.ETTA_JAMES,
        "snooks eaglin": UnifiedArtist.SNOOKS_EAGLIN,
        "keb' mo'": UnifiedArtist.KEB_MO,
        "odetta": UnifiedArtist.ODETTA,
        "tlc": UnifiedArtist.TLC,
        "professor longhair": UnifiedArtist.PROFESSOR_LONGHAIR,
        "bryan adams": UnifiedArtist.BRYAN_ADAMS,
        "ry cooder": UnifiedArtist.RY_COODER,
        "fats domino": UnifiedArtist.FATS_DOMINO,
        "papa roach": UnifiedArtist.PAPA_ROACH,
        "shinedown": UnifiedArtist.SHINEDOWN,
        "midnight oil": UnifiedArtist.MIDNIGHT_OIL,
        "christina aguilera": UnifiedArtist.CHRISTINA_AGUILERA,
        "patti smith": UnifiedArtist.PATTI_SMITH,
        "luther allison": UnifiedArtist.LUTHER_ALLISON,
        "miguel": UnifiedArtist.MIGUEL,
        "unlike pluto": UnifiedArtist.UNLIKE_PLUTO,
        "foo fighters": UnifiedArtist.FOO_FIGHTERS,
        "living colour": UnifiedArtist.LIVING_COLOUR,
        "the sheepdogs": UnifiedArtist.THE_SHEEPDOGS,
        "kelly clarkson": UnifiedArtist.KELLY_CLARKSON,
        "willie dixon": UnifiedArtist.WILLIE_DIXON,
        "skunk anansie": UnifiedArtist.SKUNK_ANANSIE,
        "seether": UnifiedArtist.SEETHER,
        "madrugada": UnifiedArtist.MADRUGADA,
        "toto": UnifiedArtist.TOTO,
        "poets of the fall": UnifiedArtist.POETS_OF_THE_FALL,
        "chris whitley": UnifiedArtist.CHRIS_WHITLEY,
        "glenn miller orchestra": UnifiedArtist.GLENN_MILLER_ORCHESTRA,
        "otis spann": UnifiedArtist.OTIS_SPANN,
        "ben e. king": UnifiedArtist.BEN_E_KING,
        "kid rock": UnifiedArtist.KID_ROCK,
        "nickelback": UnifiedArtist.NICKELBACK,
        "keith sweat": UnifiedArtist.KEITH_SWEAT,
        "chris smither": UnifiedArtist.CHRIS_SMITHER,
        "fats waller": UnifiedArtist.FATS_WALLER,
        "black rebel motorcycle club": UnifiedArtist.BLACK_REBEL_MOTORCYCLE_CLUB,
        "my chemical romance": UnifiedArtist.MY_CHEMICAL_ROMANCE,
        "paul weller": UnifiedArtist.PAUL_WELLER,
        "harry connick, jr.": UnifiedArtist.HARRY_CONNICK_JR,
        "joywave": UnifiedArtist.JOYWAVE,
        "kenny wayne shepherd": UnifiedArtist.KENNY_WAYNE_SHEPHERD,
        "miley cyrus": UnifiedArtist.MILEY_CYRUS,
        "cyndi lauper": UnifiedArtist.CYNDI_LAUPER,
        "tommy dorsey": UnifiedArtist.TOMMY_DORSEY,
        "panic! at the disco": UnifiedArtist.PANIC_AT_THE_DISCO,
        "san holo": UnifiedArtist.SAN_HOLO,
        "son lux": UnifiedArtist.SON_LUX,
        "the california honeydrops": UnifiedArtist.THE_CALIFORNIA_HONEYDROPS,
        "mississippi fred mcdowell": UnifiedArtist.MISSISSIPPI_FRED_MCDOWELL,
        "tinsley ellis": UnifiedArtist.TINSLEY_ELLIS,
        "tommy castro": UnifiedArtist.TOMMY_CASTRO,
        "jagged edge": UnifiedArtist.JAGGED_EDGE,
        "dua lipa": UnifiedArtist.DUA_LIPA,
        "john mayall & the bluesbreakers": UnifiedArtist.JOHN_MAYALL_AND_THE_BLUESBREAKERS,
        "olly murs": UnifiedArtist.OLLY_MURS,
        "alter bridge": UnifiedArtist.ALTER_BRIDGE,
        "faithless": UnifiedArtist.FAITHLESS,
        "paramore": UnifiedArtist.PARAMORE,
        "honne": UnifiedArtist.HONNE,
        "al jarreau": UnifiedArtist.AL_JARREAU,
        "breathe carolina": UnifiedArtist.BREATHE_CAROLINA,
        "rüfüs du sol": UnifiedArtist.RUFUS_DU_SOL,
        "jason derulo": UnifiedArtist.JASON_DERULO,
        "demi lovato": UnifiedArtist.DEMI_LOVATO,
        "caro emerald": UnifiedArtist.CARO_EMERALD,
        "ray j": UnifiedArtist.RAY_J,
        "estelle": UnifiedArtist.ESTELLE,
        "tink": UnifiedArtist.TINK,
        "jazzanova": UnifiedArtist.JAZZANOVA,
        "ash grunwald": UnifiedArtist.ASH_GRUNWALD,
        "billy joel": UnifiedArtist.BILLY_JOEL,
        "kid travis": UnifiedArtist.KID_TRAVIS,
        "adam lambert": UnifiedArtist.ADAM_LAMBERT,
        "elvin bishop": UnifiedArtist.ELVIN_BISHOP,
        "avril lavigne": UnifiedArtist.AVRIL_LAVIGNE,
        "j.b. lenoir": UnifiedArtist.J_B_LENOIR,
        "cavetown": UnifiedArtist.CAVETOWN,
        'clarence "gatemouth" brown': UnifiedArtist.CLARENCE_GATEMOUTH_BROWN,
        "hippie sabotage": UnifiedArtist.HIPPIE_SABOTAGE,
        "louis armstrong": UnifiedArtist.LOUIS_ARMSTRONG,
        "big joe williams": UnifiedArtist.BIG_JOE_WILLIAMS,
        "tab benoit": UnifiedArtist.TAB_BENOIT,
        "robert glasper": UnifiedArtist.ROBERT_GLASPER,
        "partynextdoor": UnifiedArtist.PARTYNEXTDOOR,
        "josé james": UnifiedArtist.JOSE_JAMES,
        "big joe turner": UnifiedArtist.BIG_JOE_TURNER,
        "crowded house": UnifiedArtist.CROWDED_HOUSE,
        "robben ford": UnifiedArtist.ROBBEN_FORD,
        "weezer": UnifiedArtist.WEEZER,
        "kelis": UnifiedArtist.KELIS,
        "ramirez": UnifiedArtist.RAMIREZ,
        "muddy waters": UnifiedArtist.MUDDY_WATERS,
        "tokimonsta": UnifiedArtist.TOKIMONSTA,
        "the rasmus": UnifiedArtist.THE_RASMUS,
        "blind willie mctell": UnifiedArtist.BLIND_WILLIE_MCTELL,
        "ana popovic": UnifiedArtist.ANA_POPOVIC,
        "jmsn": UnifiedArtist.JMSN,
        "jeezy": UnifiedArtist.JEEZY,
        "anne-marie": UnifiedArtist.ANNE_MARIE,
        "charlie puth": UnifiedArtist.CHARLIE_PUTH,
        "crash test dummies": UnifiedArtist.CRASH_TEST_DUMMIES,
        "kygo": UnifiedArtist.KYGO,
        "benny sings": UnifiedArtist.BENNY_SINGS,
        "tinashe": UnifiedArtist.TINASHE,
        "koko taylor": UnifiedArtist.KOKO_TAYLOR,
        "david guetta": UnifiedArtist.DAVID_GUETTA,
        "scott h. biram": UnifiedArtist.SCOTT_H_BIRAM,
        "baby tate": UnifiedArtist.BABY_TATE,
        "brian mcknight": UnifiedArtist.BRIAN_MCKNIGHT,
        "basement jaxx": UnifiedArtist.BASEMENT_JAXX,
        "george benson": UnifiedArtist.GEORGE_BENSON,
        'bobby "blue" bland': UnifiedArtist.BOBBY_BLUE_BLAND,
        "the veronicas": UnifiedArtist.THE_VERONICAS,
        "ace of base": UnifiedArtist.ACE_OF_BASE,
        "colin james": UnifiedArtist.COLIN_JAMES,
        "evanescence": UnifiedArtist.EVANESCENCE,
        "our lady peace": UnifiedArtist.OUR_LADY_PEACE,
        "shemekia copeland": UnifiedArtist.SHEMEKIA_COPELAND,
        "shawn james": UnifiedArtist.SHAWN_JAMES,
        "omarion": UnifiedArtist.OMARION,
        "johnny winter": UnifiedArtist.JOHNNY_WINTER,
        "bon jovi": UnifiedArtist.BON_JOVI,
        "lauv": UnifiedArtist.LAUV,
        "the knocks": UnifiedArtist.THE_KNOCKS,
        "motionless in white": UnifiedArtist.MOTIONLESS_IN_WHITE,
        "james arthur": UnifiedArtist.JAMES_ARTHUR,
        "labrinth": UnifiedArtist.LABRINTH,
        "chris thomas king": UnifiedArtist.CHRIS_THOMAS_KING,
        "marc e. bassy": UnifiedArtist.MARC_E_BASSY,
        "lcd soundsystem": UnifiedArtist.LCD_SOUNDSYSTEM,
        "elvis costello": UnifiedArtist.ELVIS_COSTELLO,
        "the presidents of the united states of america": UnifiedArtist.THE_PRESIDENTS_OF_THE_UNITED_STATES_OF_AMERICA,
        "brandy": UnifiedArtist.BRANDY,
        "chromatics": UnifiedArtist.CHROMATICS,
        "dvsn": UnifiedArtist.DVSN,
        "the darkness": UnifiedArtist.THE_DARKNESS,
        "filter": UnifiedArtist.FILTER,
        "trentemøller": UnifiedArtist.TRENTEMOLLER,
        "solange": UnifiedArtist.SOLANGE,
        "monica": UnifiedArtist.MONICA,
        "awolnation": UnifiedArtist.AWOLNATION,
        "simple plan": UnifiedArtist.SIMPLE_PLAN,
        "elmore james": UnifiedArtist.ELMORE_JAMES,
        "stereophonics": UnifiedArtist.STEREOPHONICS,
        "blues traveler": UnifiedArtist.BLUES_TRAVELER,
        "faith evans": UnifiedArtist.FAITH_EVANS,
        "marina": UnifiedArtist.MARINA,
        "mike posner": UnifiedArtist.MIKE_POSNER,
        "en vogue": UnifiedArtist.EN_VOGUE,
        "g. love & special sauce": UnifiedArtist.G_LOVE_AND_SPECIAL_SAUCE,
        "charli xcx": UnifiedArtist.CHARLI_XCX,
        "sam feldt": UnifiedArtist.SAM_FELDT,
        "gusgus": UnifiedArtist.GUSGUS,
        "anders osborne": UnifiedArtist.ANDERS_OSBORNE,
        "coco montoya": UnifiedArtist.COCO_MONTOYA,
        "oliver tree": UnifiedArtist.OLIVER_TREE,
        "clean bandit": UnifiedArtist.CLEAN_BANDIT,
        "twenty one pilots": UnifiedArtist.TWENTY_ONE_PILOTS,
        "pnb rock": UnifiedArtist.PNB_ROCK,
        "marilyn monroe": UnifiedArtist.MARILYN_MONROE,
        "backstreet boys": UnifiedArtist.BACKSTREET_BOYS,
        "pj morton": UnifiedArtist.PJ_MORTON,
        "saint etienne": UnifiedArtist.SAINT_ETIENNE,
        "ladytron": UnifiedArtist.LADYTRON,
        "dj candlestick": UnifiedArtist.DJ_CANDLESTICK,
        "og ron c": UnifiedArtist.OG_RON_C,
        "incubus": UnifiedArtist.INCUBUS,
        "blind boy fuller": UnifiedArtist.BLIND_BOY_FULLER,
        "natasha bedingfield": UnifiedArtist.NATASHA_BEDINGFIELD,
        "toni braxton": UnifiedArtist.TONI_BRAXTON,
        "khalid": UnifiedArtist.KHALID,
        "samantha fish": UnifiedArtist.SAMANTHA_FISH,
        "wheatus": UnifiedArtist.WHEATUS,
        "phantom planet": UnifiedArtist.PHANTOM_PLANET,
        "dawn richard": UnifiedArtist.DAWN_RICHARD,
        "justice": UnifiedArtist.JUSTICE,
        "sophie ellis-bextor": UnifiedArtist.SOPHIE_ELLIS_BEXTOR,
        "stacey kent": UnifiedArtist.STACEY_KENT,
        "ciara": UnifiedArtist.CIARA,
        "albert cummings": UnifiedArtist.ALBERT_CUMMINGS,
        "fantasia": UnifiedArtist.FANTASIA,
        "tate mcrae": UnifiedArtist.TATE_MCRAE,
        "the white stripes": UnifiedArtist.THE_WHITE_STRIPES,
        "daughtry": UnifiedArtist.DAUGHTRY,
        "bukka white": UnifiedArtist.BUKKA_WHITE,
        "bryson tiller": UnifiedArtist.BRYSON_TILLER,
        "santana": UnifiedArtist.SANTANA,
        "ava max": UnifiedArtist.AVA_MAX,
        "i monster": UnifiedArtist.I_MONSTER,
        "the delta saints": UnifiedArtist.THE_DELTA_SAINTS,
        "julie london": UnifiedArtist.JULIE_LONDON,
        "otis rush": UnifiedArtist.OTIS_RUSH,
        "jj grey & mofro": UnifiedArtist.JJ_GREY_AND_MOFRO,
        "hootie & the blowfish": UnifiedArtist.HOOTIE_AND_THE_BLOWFISH,
        "vedo": UnifiedArtist.VEDO,
        "joan jett & the blackhearts": UnifiedArtist.JOAN_JETT_AND_THE_BLACKHEARTS,
        "goldlink": UnifiedArtist.GOLDLINK,
        "doja cat": UnifiedArtist.DOJA_CAT,
        "andrea storm kaden": UnifiedArtist.ANDREA_STORM_KADEN,
        "slim harpo": UnifiedArtist.SLIM_HARPO,
        "jennifer lopez": UnifiedArtist.JENNIFER_LOPEZ,
        "flight facilities": UnifiedArtist.FLIGHT_FACILITIES,
        "kelela": UnifiedArtist.KELELA,
        "nat king cole": UnifiedArtist.NAT_KING_COLE,
        "jay sean": UnifiedArtist.JAY_SEAN,
        "carly rae jepsen": UnifiedArtist.CARLY_RAE_JEPSEN,
        "lion babe": UnifiedArtist.LION_BABE,
        "kai straw": UnifiedArtist.KAI_STRAW,
        "westlife": UnifiedArtist.WESTLIFE,
        "lady a": UnifiedArtist.LADY_A,
        "the paul butterfield blues band": UnifiedArtist.THE_PAUL_BUTTERFIELD_BLUES_BAND,
        "george michael": UnifiedArtist.GEORGE_MICHAEL,
        "ray eberle": UnifiedArtist.RAY_EBERLE,
        "che ecru": UnifiedArtist.CHE_ECRU,
        "rihanna": UnifiedArtist.RIHANNA,
        "stevie ray vaughan": UnifiedArtist.STEVIE_RAY_VAUGHAN,
        "cheat codes": UnifiedArtist.CHEAT_CODES,
        "3oh!3": UnifiedArtist._3OH_3,
        "ali gatie": UnifiedArtist.ALI_GATIE,
        "calvin russell": UnifiedArtist.CALVIN_RUSSELL,
        "mgmt": UnifiedArtist.MGMT,
        "fred astaire": UnifiedArtist.FRED_ASTAIRE,
        "fleet foxes": UnifiedArtist.FLEET_FOXES,
        "lily allen": UnifiedArtist.LILY_ALLEN,
    }
}

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

# SA + Audio V0, V1, V2, V3, V4 (each version is an expansion of the previous one)
CATEGORY_MAP_MIX_V3 = {
    category: _mix_map([
        CATEGORY_MAP_SA[category] if category in CATEGORY_MAP_SA else {},
        CATEGORY_MAP_AUDIO_V4[category] if category in CATEGORY_MAP_AUDIO_V4 else {},
    ])
    for category in set(list(CATEGORY_MAP_SA.keys()) + list(CATEGORY_MAP_AUDIO_V4.keys()))
}

VOCAB2ID_MIX_V3 = Vocab2Id(
    CATEGORY_MAP_MIX_V3,
    UNIFIED_VOCAB2ID_V3,
)

# SA + Audio V0, V1, V2, V3, V4, V5 (each version is an expansion of the previous one)
CATEGORY_MAP_MIX_V4 = {
    category: _mix_map([
        CATEGORY_MAP_SA[category] if category in CATEGORY_MAP_SA else {},
        CATEGORY_MAP_AUDIO_V5[category] if category in CATEGORY_MAP_AUDIO_V5 else {},
    ])
    for category in set(list(CATEGORY_MAP_SA.keys()) + list(CATEGORY_MAP_AUDIO_V5.keys()))
}

VOCAB2ID_MIX_V4 = Vocab2Id(
    CATEGORY_MAP_MIX_V4,
    UNIFIED_VOCAB2ID_V4,
)

_UNIFIED_VOCAB2ID_INST_LST = [UnifiedDumpster.NONE] + list(CATEGORY_MAP_INSTRUMENT_V1[UnifiedCategory.INSTRUMENT].values())
UNIFIED_VOCAB2ID_INST = lst_to_unified_vocab2id(_UNIFIED_VOCAB2ID_INST_LST)

VOCAB2ID_INST_V1 = Vocab2Id(
    CATEGORY_MAP_INSTRUMENT_V1, 
    UNIFIED_VOCAB2ID_INST)

# =========================================================
# Compatibility Check
# Make sure the following vocab2id is compatible with SA vocab2id
# =========================================================

def _sa_vocab_compatibility_check(vocal2ids):
    sa_vocab2id = VOCAB2ID_SA.to_dict()
    for name, vocab2id in vocal2ids.items():
        vocab2id_dict = vocab2id.to_dict()
        for k, v in sa_vocab2id.items():
            if k not in vocab2id_dict:
                raise ZhVocabError(f"Key {k} is not in {name}")
            if v != vocab2id_dict[k]:
                print(vocab2id_dict)
                print(sa_vocab2id)
                raise ZhVocabError(f"For key {k}, value {vocab2id_dict[k]} does not match {v} in VOCAB2ID_SA")

_sa_vocab_compatibility_check({"mix_v4": VOCAB2ID_MIX_V4})


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

AUDIO_V4_TO_SA_TAG_MAP = {
    "Pop": "Pop",
    "Chinese Ballad Pop": "Pop",
    "Chinese Pop": "Pop",
    "Modern Pop Ballad": "Pop",
    "K-Pop": "Pop",
    "J Pop": "Pop",
    "Indian Pop": "Pop",
    "African Pop": "Pop",
    "French Pop": "Pop",
    "Italian Pop": "Pop",
    "German Pop": "Pop",
    "Russian Pop": "Pop",
    "Turkish Pop": "Pop",
    "Arabic Pop": "Pop",
    "Teen Pop": "Pop",
    "Indie Pop": "Pop",
    "Dream Pop": "Pop",
    "City Pop": "Pop",
    "Synth Pop": "Pop",
    "Celtic Pop": "Pop",
    "Dance Pop": "Pop",
    "Deep Dance Pop": "Pop",
    "Electropop": "Pop",
    "Chamber Pop": "Pop",
    "Acapella": "Pop",
    "Country": "Country",
    "Country Pop": "Country",
    "Blue Grass": "Country",
    "Country Rock": "Country",
    "Yodeling": "Country",
    "Country Folk": "Country",
    "Early Country": "Country",
    "Folk": "Folk",
    "Chinese Folk": "Folk",
    "Folk Pop": "Folk",
    "Indie Folk": "Folk",
    "Contemporary Folk": "Folk",
    "English Folk": "Folk",
    "Irish Folk": "Folk",
    "French Folk": "Folk",
    "Korean Folk": "Folk",
    "Japanese Folk": "Folk",
    "African Folk": "Folk",
    "Arabic Folk": "Folk",
    "Traditional Folk": "Folk",
    "Celtic Folk": "Folk",
    "Electronic": "Electronic",
    "EDM": "Electronic",
    "EDM Trap": "Electronic",
    "Deep Pop Edm": "Electronic",
    "House": "Electronic",
    "Tropical House": "Electronic",
    "Future House": "Electronic",
    "Bass House": "Electronic",
    "Lo-Fi House": "Electronic",
    "Melbourne Bounce": "Electronic",
    "Future Bounce": "Electronic",
    "Big Room House": "Electronic",
    "Future Bass": "Electronic",
    "Kawaii Bass": "Electronic",
    "Bubblegum Bass": "Electronic",
    "Downtempo": "Electronic",
    "Trip Hop": "Electronic",
    "Chillout": "Electronic",
    "Midtempo": "Electronic",
    "Vaporwave": "Electronic",
    "Synthwave": "Electronic",
    "Chillwave": "Electronic",
    "Melodic Bass": "Electronic",
    "Dubstep": "Electronic",
    "Color Bass": "Electronic",
    "Ambient": "Electronic",
    "Dark Ambient": "Electronic",
    "Breakbeat": "Electronic",
    "Trance": "Electronic",
    "Techno": "Electronic",
    "Drum&Bass": "Electronic",
    "8 Bit": "Electronic",
    "Folktronica": "Electronic",
    "Glitch": "Electronic",
    "Hardstyle": "Electronic",
    "Hardcore": "Electronic",
    "Hi-NRG": "Electronic",
    "IDM": "Electronic",
    "Disco": "Electronic",
    "Phonk": "Electronic",
    "Jersey Club": "Electronic",
    "UK Garage": "Electronic",
    "Moombahton": "Electronic",
    "Sci-Fi": "Electronic",
    "Chinese Style": "Chinese Style",
    "China-Wave": "Chinese Style",
    "GuFeng Music": "Chinese Style",
    "Chinoiserie Rap": "Chinese Style",
    "Chinoiserie Electronic": "Chinese Style",
    "Chinoiserie Rock": "Chinese Style",
    "Rock": "Rock",
    "Hard Rock": "Rock",
    "Psychedelic Rock": "Rock",
    "Pop Rock": "Rock",
    "Brit Pop": "Rock",
    "Alternative Rock": "Rock",
    "Indie Rock": "Rock",
    "Post-Rock": "Rock",
    "Lo-Fi Rock": "Rock",
    "J Rock": "Rock",
    "Shoegaze-Rock": "Rock",
    "Math Rock": "Rock",
    "Surf Rock": "Rock",
    "Progressive Rock": "Rock",
    "Soft Rock": "Rock",
    "Grunge Rock": "Rock",
    "Punk": "Punk",
    "Pop Punk": "Punk",
    "Post-Punk": "Punk",
    "New Wave ": "Punk",
    "New Wave": "Punk",
    "Hardcore Punk ": "Punk",
    "Hardcore Punk": "Punk",
    "Ska Punk": "Punk",
    "No Wave": "Punk",
    "Dance Punk": "Punk",
    "Emo Punk": "Punk",
    "Melodic Hardcore": "Punk",
    "Jazz": "Jazz",
    "Jazz Pop": "Jazz",
    "Bebop": "Jazz",
    "Ragtime": "Jazz",
    "Traditional Jazz": "Jazz",
    "Jazz Fusion": "Jazz",
    "Bossa Nova": "Jazz",
    "Avant-Garde Jazz": "Jazz",
    "Swing": "Jazz",
    "Big Band": "Jazz",
    "Post-Bop": "Jazz",
    "Smooth Jazz": "Jazz",
    "Cool Jazz": "Jazz",
    "Vocal Jazz": "Jazz",
    "Nu Jazz": "Jazz",
    "Free Jazz": "Jazz",
    "Acid Jazz": "Jazz",
    "Gypsy Jazz": "Jazz",
    "Hard Bop": "Jazz",
    "Japanese Jazz Fusion": "Jazz",
    "Hip Hop": "Hip Hop/Rap",
    "Trap Rap": "Hip Hop/Rap",
    "Old School": "Hip Hop/Rap",
    "R&B Rap": "Hip Hop/Rap",
    "Boombap": "Hip Hop/Rap",
    "Jazz Hip Hop": "Hip Hop/Rap",
    "Alternative Hip Hop": "Hip Hop/Rap",
    "Pop Rap": "Hip Hop/Rap",
    "Hardcore Rap": "Hip Hop/Rap",
    "Comedy Hip Hop": "Hip Hop/Rap",
    "Hip House": "Hip Hop/Rap",
    "Chill Beats": "Hip Hop/Rap",
    "Gangsta Rap": "Hip Hop/Rap",
    "West Coast Hip Hop": "Hip Hop/Rap",
    "East Coast Hip Hop": "Hip Hop/Rap",
    "Southern Hip Hop": "Hip Hop/Rap",
    "Midwest Hip Hop": "Hip Hop/Rap",
    "Drill Rap": "Hip Hop/Rap",
    "Emo Rap": "Hip Hop/Rap",
    "Afrobeats": "Hip Hop/Rap",
    "Classical": "Classical",
    "Middle Ages": "Classical",
    "Renaissance": "Classical",
    "Baroque": "Classical",
    "Classical period": "Classical",
    "Romantic music": "Classical",
    "Impressionism": "Classical",
    "Modernism": "Classical",
    "Contemporary classical music": "Classical",
    "R&B/Soul": "R&B/Soul",
    "Funk": "R&B/Soul",
    "Contemporary R&B": "R&B/Soul",
    "Neo Soul": "R&B/Soul",
    "Soul": "R&B/Soul",
    "Pop Soul": "R&B/Soul",
    "Doo-Wop": "R&B/Soul",
    "Gospel": "Devotional",
    "Classic R&B / Soul": "R&B/Soul",
    "Progressive R&B": "R&B/Soul",
    "Neo Funk": "R&B/Soul",
    "Blues": "Blues",
    "Country Blues": "Blues",
    "Jazz Blues": "Blues",
    "Contemporary Blues": "Blues",
    "Rock Blues": "Blues",
    "Traditional Blues": "Blues",
    "Boogie Woogie": "Blues",
    "Reggae": "Reggae",
    "Dancehall": "Reggae",
    "Dub": "Reggae",
    "Metal": "Metal",
    "Black Metal": "Metal",
    "Death Metal": "Metal",
    "Doom Metal": "Metal",
    "Glam Metal\n\uff08Hair Metal\uff0cPop Metal\uff09": "Metal",
    "Glam Metal (Hair Metal, Pop Metal)": "Metal",
    "Grindcore ": "Metal",
    "Grindcore": "Metal",
    "Power Metal": "Metal",
    "Progressive Metal\uff08Prog Metal)": "Metal",
    "Progressive Metal (Prog Metal)": "Metal",
    "Speed Metal": "Metal",
    "Thrash Metal": "Metal",
    "Industrial Metal": "Metal",
    "Folk Metal": "Metal",
    "Rap Metal": "Metal",
    "Heavy Metal": "Metal",
    "Nu Metal": "Metal",
    "Metalcore": "Metal",
    "Post-hardcore": "Metal",
    "Deathcore": "Metal",
    "Gothic Metal": "Metal",
    "Kawaii Metal": "Metal",
    "Childhood": "Childhood",
    "Devotional": "Devotional",
    "Holiday Music": "Devotional",
    "Christian Music": "Devotional",
    "Buddhist music": "Devotional",
    "Chinese Tradition": "Chinese Tradition",
    "Chinese Opera": "Chinese Tradition",
    "Traditional Chinese Folk": "Chinese Tradition",
    "Chinese Traditional Instrumental Music": "Chinese Tradition",
    "Chinese Quyi": "Chinese Tradition",
    "Red Song": "Chinese Tradition",
    "New Chinese Folk": "Chinese Tradition",
    "Shi Dai Qu": "Chinese Tradition",
    "Easy Listening": "Easy Listening",
    "New Age": "New Age",
    "Latin": "Latin",
    "Latin Pop": "Latin",
    "Latin Rock": "Latin",
    "Tango": "Latin",
    "Reggaeton": "Latin",
    "Brazilian Funk Style": "Latin",
    "Cha-Cha-Cha": "Latin",
    "Samba": "Latin",
    "Rumba": "Latin",
    "Alternative/Indie": "Alternative/Indie",
    "Epic": "Epic",
    "World Music": "World Music",
    "Korea Trot": "World Music",
    "Shima Uta": "World Music",
    "Enka": "World Music",
    "Japanese Traditional Music": "World Music",
    "Bhangra": "World Music",
    "Flamenco": "World Music",
    "Son cubano": "World Music",
    "Worldbeat": "World Music",
    "Mongolian Folk Songs": "World Music",
    "Uygur Folk Songs": "World Music",
    "Fado": "World Music",
    "Ainu Folk": "World Music",
    "Okinawan Pop": "World Music",
    "DJ": "DJ",
    "VinaHouse": "DJ",
    "MC": "MC",
    "Score": "Score",
    "Other": "Other Genre",
    "Other Genre": "Other Genre",
    # lang
    "Sichuanese": "Chinese Dialects",
    "Instrumental/Non-vocal": "Non-vocal",
}
