from recipes.bigmusic.utils.format_utils import rewrite_metadata

text_pool_1_toplevel = [
    'Blues',
    "Children's Music",
    'Classical Music',
    'Country',
    'Devotional',
    'EDM',
    'Electronic Music',
    'Experimental',
    'Folk',
    'Hip Hop',
    'Jazz',
    'Latin',
    'MC',
    'Metal',
    'New Age',
    'Pop',
    'Punk',
    'R&B',
    'Reggae',
    'Rock',
    'Sound Effect',
    'Sound Track',
    'World Music',
    'Trap Rap'
]


text_pool_2 = [
    'Angry',
    'Relaxing',
    'Chill',
    'Cute',
    'Dreamy',
    'Dynamic',
    'Excited',
    'Funny',
    'Funky',
    'Happy',
    'Healing',
    'Inspirational',
    'Miss',
    'Mysterious',
    'No Mood',
    'Memory',
    'Romantic',
    'Lonely',
    'Shocking',
    'Sorrow',
    'Sweet'
    'Tense',
    'Weird',
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

text_pool_5_zh_genre = [
        "流行",
        "摇滚",
        "说唱",
        "民谣",
        "古风",
        "电子",
        "抒情",
        "民歌",
        "儿童歌曲",
        "佛教",
        "情歌",
        "舞曲",
]
text_pool_6_zh_mood = [
        "温馨",
        "忧伤",
        "欢快",
        "思念",
        "浪漫",
        "轻松",
        "怀旧",
        "激情",
        "感人",
        "无奈",
        "深情",
        # "甜蜜",
        # "振奋",
        # "迷茫",
        # "励志",
        # "温柔",
        # "豪迈",
        # "自信",
]

text_pool_7_zh_vocal = ["女声", "男声", "童声"]
text_pool_8_zh_vocal_timbre = ["温暖", "空灵", "低沉", "甜美", '沙哑', '高亢', '明亮', '性感', '可爱']
text_pool_9_zh_lang = ["粤语", "闽南语", "普通话"]

MCC_MOOD = ['Angry', 'Chill', 'Cute', 'Dynamic', 'Excited', 'Happy', 'Romantic', 'Sorrow', 'Tense', 'Weird']
MCC_GENRE = ['Blues', 'Childhood', 'Classical', 'Country', 'Devotional', 'Electronic', 'Experimental', 'Folk', 'Hip Hop/Rap', 'Jazz', 'Metal', 'New Age', 'Pop', 'R&B/Soul', 'Reggae', 'Rock', 'SoundTrack', 'Trap Rap']
MCC_GENDER = ['Female', 'Male']
MCC_LANG = ['English', 'Chinese']
NONE_LABEL = 'None'

class MulanTagger:
    def __init__(self, mulan_tag_type="mulan_genres"):
        self._tag2embed = None
        self.mulan_tag_type = mulan_tag_type
        self.none_label = NONE_LABEL
        self._all_tags = None

        if mulan_tag_type == "mulan_genres":
            self._tag2text_pool = {
                "genre": text_pool_1_toplevel, 
                "mood": text_pool_2, 
                "gender": text_pool_4
            }
        elif mulan_tag_type == "mcc_genres":
            self._tag2text_pool = {
                "genre": MCC_GENRE, 
                "mood": MCC_MOOD, 
                "gender": MCC_GENDER
            }
        elif mulan_tag_type == "cn_tags":
            self._tag2text_pool = {
                "genre": text_pool_5_zh_genre, 
                "mood": text_pool_6_zh_mood, 
                "gender": text_pool_7_zh_vocal,                
                "voice": text_pool_8_zh_vocal_timbre,
                "lang": text_pool_9_zh_lang,
            }
        else:
            raise ValueError(f"Unhandled tag_type: {mulan_tag_type}")
        
        self.id2vocab, self.vocab2id = self.get_vocab()

    def get_label_id(self, label):
        if label in ['nan', 'None', None, '']:
            label = self.none_label
        elif label not in self.vocab2id:
            label = self.none_label
        return self.vocab2id[label]


    def get_vocab(self):
        all_values = [NONE_LABEL]
        for categories in self._tag2text_pool.values():
            all_values.extend(categories)
        id2vocab = { idx: value for idx, value in enumerate(all_values) }
        vocab2id = { value: idx for idx, value in enumerate(all_values) }
        return id2vocab, vocab2id

    def vocab_size(self):
        return len(self.id2vocab)

    def get_tag_embeds(self, requires):
        # TODO: (AS) move this out of inner function. Currently here to remove circular dependency
        from recipes.bigmusic.lightning.embedding_modules import get_mulan_embeds
        # default text pool embeddings
        if self._tag2embed is None:
            self._tag2embed = {}
            for tag_label, category_labels in self._tag2text_pool.items():
                category_embeds = get_mulan_embeds(
                    requires, category_labels, data_type='text'
                )
                self._tag2embed[tag_label] = { 'labels': category_labels, 'embeds': category_embeds }
        return self._tag2embed

    def tag_to_style_text(self, metadata):        
        if self.mulan_tag_type == 'cn_tags':
            # TODO: (QQ) pull out necessary tags from metadata such as: language_id            
            genre, mood, gender, voice, lang = metadata['genre'], metadata['mood'], metadata['gender'], metadata['voice'], metadata['lang']
            return '，'.join([genre, mood, gender, voice, lang])
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
