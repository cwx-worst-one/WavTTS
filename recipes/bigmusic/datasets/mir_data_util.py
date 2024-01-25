
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

chinese_genre1_vocab = [
    "流行",     # use genre2
    "嘻哈",     # use genre2
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
chinese_genre2_vocab = [
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
chinese_mood_vocab = [
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

chinese_scene_vocab = [
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

gender_vocab = ["Male", "Female"]

lang_vocab = ["普通话", "粤语", "闽南话"]
 
NONE_LABEL = 'None'

def get_mir_vocab(lang='Zh'):
    all_values = []
    all_values.append(NONE_LABEL)
    for categories in [
        chinese_genre1_vocab, chinese_genre2_vocab, 
        chinese_mood_vocab, chinese_scene_vocab, 
        gender_vocab, lang_vocab]:
        all_values.extend(categories)
    id2vocab = { idx: value for idx, value in enumerate(all_values) }
    vocab2id = { value: idx for idx, value in enumerate(all_values) }
    return id2vocab, vocab2id
