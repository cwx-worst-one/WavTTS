from samantha.dataio.bigmusic.base_transform import MusicMetaRWTransform
from samantha.dataio.bigmusic.transforms.utils import *

MAX_LLM_TAG_LEN = 10


def normalize_lang(lang):
    if "non-vocal" in lang.lower():
        return "Instrumental"
    return lang


def filter_other(obj):
    def is_other(tags):
        if not isinstance(tags, list):
            if isinstance(tags, str):
                return tags.lower().startswith("other")
            else:
                return False
        for tag in tags:
            if isinstance(tag, str) and tag.lower().startswith("other"):
                return True
        return False

    for k, v in obj.items():
        if is_other(v):
            obj[k] = []

    return obj


def filter_null(keywords):
    return [k for k in keywords if k]


def parse_apm_facet(facets_list_str):
    facet_aspects = {}
    try:
        # It's safer to try JSON first, then fallback to eval if needed.
        facet_list = json.loads(facets_list_str)
    except (json.JSONDecodeError, TypeError):
        try:
            facet_list = eval(facets_list_str)
        except Exception:
            return {}  # Return empty if both fail

    if not isinstance(facet_list, list):
        return {}

    for facet in facet_list:
        if isinstance(facet, dict):
            for k, v in facet.items():
                facet_aspects[k] = v
    return facet_aspects


def process_keywords(data):
    keywords = []

    # Helper to extend keywords from various sources
    def extend_keywords(source):
        if source:
            if isinstance(source, list):
                keywords.extend(source)
            elif isinstance(source, str):
                keywords.append(source)

    # Genius Tags
    extend_keywords(get_nested_value(data, "genius.tags"))

    # Raw Data Sources (v1, raw, raw_merged)
    raw_sources = [
        get_nested_value(data, "raw.v1.genres"),
        get_nested_value(data, "meta.raw.v1.genres"),
        get_nested_value(data, "raw.genres"),
        get_nested_value(data, "raw.moods"),
        get_nested_value(data, "raw.themes"),
        get_nested_value(data, "meta.raw.genres"),
        get_nested_value(data, "meta.raw.moods"),
        get_nested_value(data, "meta.raw.themes"),
        get_nested_value(data, "raw.tags"),
    ]
    for source in raw_sources:
        if isinstance(source, str) and source.startswith(
            "["
        ):  # Handle stringified lists
            try:
                source = eval(source)
            except Exception:
                pass
        extend_keywords(source)

    keywords = keywords + extract_keywords(data)
    keywords = flatten_nested_list(keywords)
    keywords = dedup_with_order(filter_null(keywords), ignore_case=True)

    return keywords


def extract_keywords(meta: Dict) -> List[str]:

    sstk = extract_keywords_sstk(meta)
    everynoise = extract_keywords_everynoise(meta)
    wyy = extract_keywords_wyy(meta)
    apm = extract_keywords_apm(meta)
    rym = extract_keywords_rym(meta)
    freeform_tags = extract_freeform_tags(meta)

    return sstk + everynoise + wyy + apm + rym + freeform_tags


def extract_keywords_sstk(meta: Dict) -> List[str]:

    try:
        instruments = meta.get("instruments", "")
        if not instruments:
            instruments = meta.get("raw", {}).get("instruments", "")
        instruments = instruments.split(", ") if instruments else []
        instruments = [x for x in instruments if x != "\\N"]
        keywords = meta.get("keywords", "")
        if not keywords:
            keywords = meta.get("raw", {}).get("keywords", "")
        keywords = keywords.split(", ") if keywords else []
        all_tags = [tag for tag in instruments + keywords if tag]
        keywords = list(set(all_tags))
    except Exception:
        keywords = []

    return keywords


def extract_keywords_everynoise(meta: Dict) -> List[str]:
    """meta.raw.genres, meta.everynoise_genre, meta.everynoise_trending.vantage/genre"""

    try:
        raw_genres = meta.get("raw", {}).get("genres", [])
        everynoise_genre = meta.get("everynoise_genre", "")
        everynoise_trending_vantage = meta.get("everynoise_trending", {}).get(
            "vantage", ""
        )  # sometimes 'everynoise_trending' exists but is None
        everynoise_trending_genre = meta.get("everynoise_trending", {}).get("genre", "")
        all_tags = [
            tag
            for tag in [
                raw_genres,
                everynoise_genre,
                everynoise_trending_genre,
                everynoise_trending_vantage,
            ]
            if tag
        ]
        flat_list = [
            tag
            for sublist in all_tags
            for tag in (sublist if isinstance(sublist, list) else [sublist])
        ]
        keywords = list(set(flat_list))
    except Exception:
        keywords = []
    return keywords


def extract_keywords_wyy(meta: Dict) -> List[str]:

    try:
        tags = ast.literal_eval(meta.get("raw", {}).get("tags", "[]"))
        category = meta.get("raw", {}).get("category", "")
        song_tag = meta.get("raw", {}).get("song_tag", "")
        song_biz_tag = meta.get("raw", {}).get("song_biz_tag", "")
        category = category.split(",") if category else []
        if song_tag:
            _song_tag = song_tag.split(",")
            song_tag = []
            for x in _song_tag:
                if x:
                    song_tag.extend(x.split("-", 1))
        else:
            song_tag = []
        song_biz_tag = song_biz_tag.split(",") if song_biz_tag else []

        keywords = tags + category + song_tag + song_biz_tag
        keywords = list(set(keywords))
    except Exception:
        keywords = []

    return keywords


def extract_keywords_apm(meta: Dict) -> List[str]:

    try:
        facets_list = json.loads(meta.get("raw", {}).get("facets_list", "[]"))
        keywords = [
            item
            for d in facets_list
            for value in d.values()
            for item in value
            if item is not None
        ]
        term = meta.get("raw", {}).get("term")
        term = term.split(",") if term else []

        keywords += term
    except Exception:
        keywords = []
    return keywords


def extract_keywords_rym(meta: Dict) -> List[str]:

    try:
        track_genres = meta.get("raw", {}).get("track_genres", [])
        keywords = []
        keywords.extend(track_genres)
        keywords = list(set(keywords))
    except Exception:
        keywords = []
    return keywords


def extract_freeform_tags(meta: Dict) -> List[str]:
    freeform_tags = meta.get("raw_merged", {}).get("freeform_tags", [])
    return freeform_tags


class ParseWebKeywords(MusicMetaRWTransform):
    def __init__(self, in_key: str = "meta", out_key: str = "keywords", **kwargs):
        super().__init__(in_key, out_key, allow_empty_in=False, **kwargs)

    def call(self, meta, **kwargs):
        keywords = process_keywords(meta)
        return keywords


def _parse_gemini_v2(gemini_v2: dict) -> dict:
    if not gemini_v2:
        return {}
    gemini_meta = {}
    gemini_meta["genre"] = gemini_v2.get("genre", {}).get("primary", [])
    gemini_meta["genre_extra"] = gemini_v2.get("genre", {}).get("additional", [])
    gemini_meta["mood"] = gemini_v2.get("mood", {}).get("keywords", [])
    gemini_meta["language"] = gemini_v2.get("language", [])
    gemini_meta["tempo"] = gemini_v2.get("musical_features", {}).get(
        "tempo_keywords", {}
    ).get("tempo", []) + gemini_v2.get("musical_features", {}).get(
        "tempo_keywords", {}
    ).get(
        "BPM", []
    )
    try:
        gemini_meta["gender"] = [
            gender
            for keyword in gemini_v2.get("musical_features", {})
            .get("vocal", {})
            .get("keywords", [])
            for gender in keyword.get("gender", [])
        ]
    except Exception:
        pass
    gemini_meta["scene"] = gemini_v2.get("scene", {}).get("keywords", [])
    scene_phrases = gemini_v2.get("scene", {}).get("phrases", [])
    # remove tailing .
    scene_phrases = [phrase.rstrip(".") for phrase in scene_phrases]
    gemini_meta["scene_phrase"] = scene_phrases
    try:
        gemini_meta["timbre"] = [
            timbre
            for keyword in gemini_v2.get("musical_features", {})
            .get("vocal", {})
            .get("keywords", [])
            for timbre in keyword.get("timbre", [])
        ]
    except Exception:
        pass
    gemini_meta["instrument"] = (
        gemini_v2.get("musical_features", {}).get("instruments", {}).get("keywords", [])
    )
    gemini_meta["era"] = gemini_v2.get("additional", {}).get("era_style", [])
    arrangement_keywords = (
        gemini_v2.get("musical_features", {}).get("arrangement", {}).get("keywords", [])
    )
    if isinstance(arrangement_keywords, dict):
        new_arrangement_keywords = []
        for _, v in arrangement_keywords.items():
            new_arrangement_keywords.extend(v)
        arrangement_keywords = new_arrangement_keywords
    gemini_meta["arrangement"] = arrangement_keywords
    gemini_meta["melody"] = (
        gemini_v2.get("musical_features", {}).get("melody", {}).get("keywords", [])
    )
    gemini_meta["rhythm"] = (
        gemini_v2.get("musical_features", {}).get("rhythm", {}).get("keywords", [])
    )
    gemini_meta["key"] = gemini_v2.get("musical_features", {}).get("key", [])
    gemini_meta["imagery"] = gemini_v2.get("abstract_descriptors", {}).get(
        "imagery", []
    )
    gemini_meta["synesthesia_tags"] = gemini_v2.get("abstract_descriptors", {}).get(
        "synesthesia_tags", []
    )
    gemini_meta["vibe"] = gemini_v2.get("abstract_descriptors", {}).get("vibe", [])
    gemini_meta["audio_features"] = gemini_v2.get("audio_features", {}).get(
        "audio_feature_keywords", []
    )
    gemini_meta["additional"] = gemini_v2.get("additional", {}).get("features", [])
    return gemini_meta


class ParseGemini(MusicMetaRWTransform):

    def __init__(
        self,
        in_key: list = [
            "meta.standard_music_meta.gemini",
            "meta.gemini",
            "meta.standard_meta.llm_augmented_meta",
        ],
        out_key: list = ["standard_music_meta.gemini", "standard_music_meta.gemini_v2"],
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=True, **kwargs)

    def call(self, item: dict, **kwargs) -> dict:
        if item is None:
            return [None, None]

        gemini1, gemini2, gemini_v2 = item
        gemini = gemini1 or gemini2  # in some cases, gemini v1 is in meta.gemini

        if gemini is not None and isinstance(
            gemini, dict
        ):  # in some cases, gemini v1 is 'NULL'
            gemini_meta = {k: v for k, v in gemini.items() if k != "scene"}
            if "scene" in gemini:
                gemini_meta["scene_phrase"] = gemini["scene"]
            gemini_meta["description"] = [gemini["character"]]  # it's a str
        else:
            gemini_meta = {}

        try:
            gemini_meta_v2 = _parse_gemini_v2(gemini_v2)
        except Exception:
            gemini_meta_v2 = {}

        return [gemini_meta, gemini_meta_v2]
