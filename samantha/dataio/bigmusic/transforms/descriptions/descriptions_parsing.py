from samantha.dataio.bigmusic.base_transform import MusicMetaRWTransform
from samantha.dataio.bigmusic.transforms.utils import *


class ParseGeminiDesc(MusicMetaRWTransform):

    def __init__(
        self,
        in_key: list = [
            "meta.standard_music_meta.gemini",
            "meta.gemini",
            "meta.standard_meta.llm_augmented_meta",
        ],
        out_key: str = "descriptions",
        **kwargs,
    ):
        super().__init__(in_key, out_key, allow_empty_in=True, **kwargs)

    def call(self, item: dict, **kwargs) -> dict:
        def update_description(source, key, descriptions, out_key):
            description = get_nested_value(source, key)
            if not description:
                return
            descriptions[out_key] = description

        gemini1, gemini2, gemini_v2 = item
        gemini_v1 = gemini1 or gemini2  # in some cases, gemini v1 is in meta.gemini

        descriptions = {}

        update_description(gemini_v1, "character", descriptions, "v1_global")
        update_description(gemini_v2, "mood.description", descriptions, "mood")
        update_description(
            gemini_v2,
            "musical_features.arrangement.description",
            descriptions,
            "arrangement",
        )
        update_description(
            gemini_v2, "musical_features.melody.description", descriptions, "melody"
        )
        update_description(
            gemini_v2, "musical_features.rhythm.description", descriptions, "rhythm"
        )
        update_description(
            gemini_v2, "musical_features.motif_description", descriptions, "motif"
        )
        update_description(
            gemini_v2, "musical_features.vocal.description", descriptions, "vocal"
        )
        update_description(
            gemini_v2,
            "musical_features.instruments.description",
            descriptions,
            "inst_desc",
        )
        update_description(
            gemini_v2,
            "musical_features.instruments.development",
            descriptions,
            "inst_dev",
        )
        update_description(
            gemini_v2, "musical_features.harmony.description", descriptions, "harmony"
        )
        update_description(
            gemini_v2,
            "audio_features.audio_features_description",
            descriptions,
            "audio_features",
        )
        update_description(
            gemini_v2,
            "abstract_descriptors.auditory_storytelling",
            descriptions,
            "auditory_storytelling",
        )
        update_description(
            gemini_v2, "description.global_description", descriptions, "global"
        )
        update_description(
            gemini_v2,
            "description.global_description_long",
            descriptions,
            "global_long",
        )
        update_description(
            gemini_v2, "description.visual_description", descriptions, "visual"
        )
        update_description(
            gemini_v2, "description.picture_description", descriptions, "picture"
        )
        update_description(
            gemini_v2, "description.highlight_description", descriptions, "highlight"
        )
        update_description(
            gemini_v2,
            "description.user_prompt_description",
            descriptions,
            "user_prompt",
        )

        return descriptions
