import pytest

from samantha.dataio.bigmusic.music_meta_transform import *


class TestMusicMetaRWTransform:
    def test_call(self):
        data = {"in_key": {"meta": {"key": "C#"}}}
        transform = MusicMetaRWTransform(in_key="in_key.meta.key", out_key="out_key")
        data = transform(data)
        assert data["out_key"] == "C#"

    @pytest.mark.parametrize(
        "data,output",
        [
            (
                {"in_key": {"meta": {"key": "C#"}}},
                {"in_key": {"meta": {"key": "C#"}}, "out_key": "C#"},
            ),
            (
                {"in_key": {"meta": {"key": ""}}},
                {"in_key": {"meta": {"key": ""}}, "out_key": ""},
            ),
            ({"in_key": {"meta": {}}}, {"in_key": {"meta": {}}, "out_key": None}),
        ],
        ids=[
            "Key exists",
            "Key exists, value is an empty string",
            "No key, but still allowed and put None to out_key",
        ],
    )
    def test_allow_empty_in(self, data, output):
        assert (
            MusicMetaRWTransform(
                in_key="in_key.meta.key", out_key="out_key", allow_empty_in=True
            )(data)
            == output
        )

    @pytest.mark.parametrize(
        "data,output",
        [
            (
                {"in_key": {"meta": {"key": "C#"}}},
                {"in_key": {"meta": {"key": "C#"}}, "out_key": "C#"},
            ),
            (
                {"in_key": {"meta": {"key": ""}}},
                {"in_key": {"meta": {"key": ""}}, "out_key": ""},
            ),
            ({"in_key": {"meta": {}}}, None),
            (
                {"in_key": {"meta": {"key": None}}},
                {"in_key": {"meta": {"key": None}}, "out_key": None},
            ),
        ],
        ids=[
            "Key exists",
            "Key exists, value is an empty string",
            "No key, not allowed, return None",
            "Key exists, value is None, still allowed",
        ],
    )
    def test_prohibit_empty_in(self, data, output):
        assert (
            MusicMetaRWTransform(
                in_key="in_key.meta.key", out_key="out_key", allow_empty_in=False
            )(data)
            == output
        )

    def test_remove_in_key(self):
        assert MusicMetaRWTransform(in_key="a", out_key="b", remove_in_key=True)(
            {"a": 1}
        ) == {"b": 1}

    def test_optional_keys_prohibit_empty_in(self):
        assert (
            MusicMetaRWTransform(
                in_key=["a", "opt_b", "opt_c"],
                out_key="output",
                optional_keys=["opt_b", "opt_c"],
                allow_empty_in=False,
            )({"a": 1, "opt_b": 2})
            == None
        ), 'key "opt_c" is missing, not allowed, return None'

    def test_optional_keys_allow_empty_in(self):
        assert MusicMetaRWTransform(
            in_key=["a", "opt_b", "opt_c"],
            out_key="output",
            optional_keys=["opt_b", "opt_c"],
            allow_empty_in=True,
        )({"a": 1, "opt_b": 2}) == {
            "a": 1,
            "opt_b": 2,
            "output": [1, 2, None],
        }, 'key "opt_c" is missing, still allowed, return None'

    def test_optional_keys_allow_empty_in_mandatory_key_missed(self):
        assert (
            MusicMetaRWTransform(
                in_key=["a", "opt_b", "opt_c"],
                out_key="output",
                optional_keys=["opt_b", "opt_c"],
                allow_empty_in=True,
            )({"a": None, "opt_b": 2})
            == None
        ), 'Non-optional key "a" is None, item is invalid, return None'


class TestGroupItems:
    def test_call(self):
        assert GroupItems(in_key=["a", "b"], out_key="out", remove_in_key=True)(
            {"a": 1, "b": 2}
        ) == {"out": [1, 2]}

    def test_invalid_arg(self):
        with pytest.raises(AssertionError):
            GroupItems(
                in_key="a", out_key="out"
            )  # string in_key is not allowed, use ["a"] instead


class TestMaflFromatPrefix:
    def test_call(self):
        assert MaflFormatPrefix(
            in_key=["prompt", "lyrics"], out_key="out", remove_in_key=True
        )({"prompt": "[my prompt] 123", "lyrics": "[verse]my lyrics"}) == {
            "out": "<SECTION> my prompt </SECTION> 123<SECTION> verse </SECTION>my lyrics"
        }


class TestMaflFormatPrefixHard:
    def test_call(self):
        assert MaflFormatPrefixHard(
            in_key=["prompt", "lyrics"],
            out_key="out",
            remove_in_key=True,
            hard_sep_start="<START>",
            hard_sep_end="<END>",
        )({"prompt": "[my prompt] 123", "lyrics": "[verse]my lyrics"}) == {
            "out": "<START>[my prompt] 123[verse]my lyrics<END>"
        }


class TestTokenIdsToString:
    def test_call(self):
        assert TokenIdsToString(
            in_key="a", out_key="b", template="<test_%s>", remove_in_key=True
        )({"a": [1, 2, 3]}) == {"b": "<test_1><test_2><test_3>"}


class TestSimpleTokenLengthEstimation:
    def test_call(self):
        assert SimpleTokenLengthEstimation(
            in_key=["a", "b"], out_key="out", remove_in_key=True
        )({"a": [1, 2, 3], "b": "abcdefg"}) == {"out": 10}


class TestStandardMetaParser:
    def test_consolidate(self):
        assert StandardMetaParser(
            in_key="music_standard_meta",
            out_key="out",
            remove_in_key=True,
            meta_types=["web", "llm"],
            mode="consolidate",
        )(
            {
                # This transform should be field name agnostic
                "music_standard_meta": {
                    "web": {"field1": ["Pop", "Chinese Pop"], "field2": ["Sad/Sorrow"]},
                    "llm": {"field1": ["Pop", "Jazz"], "field2": ["Happy"]},
                    "tagging_model": {"field1": ["Rock"], "field2": ["Neutral"]},
                }
            }
        ) == {
            "out": {
                # combine web and llm, ignore tagging_model
                "field1": ["Pop", "Chinese Pop", "Jazz"],
                "field2": ["Sad/Sorrow", "Happy"],
            }
        }

    def test_sample(self, mocker):
        # patch random.choices:
        mock_choices = mocker.patch(
            "samantha.dataio.bigmusic.music_meta_transform.random.choices"
        )
        mock_choices.side_effect = lambda lst, *args, **kwargs: [lst[0]]
        assert StandardMetaParser(
            in_key="music_standard_meta",
            out_key="out",
            remove_in_key=True,
            meta_types=["web", "llm"],
            mode="sample",
            meta_weights=[1, 1],
        )(
            {
                "music_standard_meta": {
                    "web": {"field1": ["Pop", "Chinese Pop"], "field2": ["Sad/Sorrow"]},
                    "llm": {"field1": ["Jazz"], "field2": ["Happy"]},
                    "tagging_model": {"field1": ["Rock"], "field2": ["Neutral"]},
                }
            }
        ) == {
            "out": {"field1": ["Pop", "Chinese Pop"], "field2": ["Sad/Sorrow"]}
        }

    def test_norm_sep(self):
        assert StandardMetaParser(
            in_key="music_standard_meta",
            out_key="out",
            remove_in_key=True,
            meta_types=["web"],
            norm_sep=True,
        )(
            {
                # This transform should be field name agnostic
                "music_standard_meta": {
                    "web": {
                        "field1": ["Pop / Chinese Pop", "Jazz"],
                        "field2": ["Sad/Sorrow / Happy"],
                    }
                }
            }
        ) == {
            "out": {
                # combine web and llm, ignore tagging_model
                "field1": ["Pop", "Chinese Pop", "Jazz"],
                "field2": ["Sad/Sorrow", "Happy"],
            }
        }


class TestPackMeta2Prompt:
    def test_call(self):
        assert PackMeta2Prompt(
            in_key="standard_meta", out_key="out", remove_in_key=True
        )({"standard_meta": {"field1": ["123", "456"], "field2": ["abc", "def"]}}) == {
            "out": "[FIELD1: 123|456][FIELD2: abc|def]"
        }


class TestTempoParser:
    def test_normal_tempo(self):
        assert TempoParser(
            in_key="meta.tempo", out_key="tempo", remove_in_key=True, dropout_rate=0.0
        )({"meta": {"tempo": "120.5"}}) == {"meta": {}, "tempo": 120.5}

    def test_musicfm_beat(self):
        assert TempoParser(
            in_key="meta.musicfm_plus.beat",
            out_key="tempo",
            remove_in_key=True,
            dropout_rate=0.0,
            # 0.5 spb -> 2 bps -> 120 bpm
        )({"meta": {"musicfm_plus": {"beat": [[1.5, 2], [2.0, 3]]}}}) == {
            "meta": {"musicfm_plus": {}},
            "tempo": 120.0,
        }

    def test_standard_music_meta_nonbpe(self):
        assert TempoParser(
            in_key="meta.standard_music_meta_nonbpe",
            out_key="tempo",
            remove_in_key=True,
            dropout_rate=0.0,
        )({"meta": {"standard_music_meta_nonbpe": {"extra_info": {"bpm": "120.5"}}}}) == {
            "meta": {},
            "tempo": 120.5,
        }

    def test_multi_in_keys(self):
        assert TempoParser(
            in_key=["meta.musicfm_plus.beat", "meta.tempo"],
            out_key="tempo",
            remove_in_key=True,
            dropout_rate=0.0,
        )({"meta": {"tempo": "120.5"}}) == {"meta": {}, "tempo": 120.5}

    def test_empty(self):
        assert TempoParser(
            in_key=["meta.musicfm_plus.beat", "meta.tempo"],
            out_key="tempo",
            remove_in_key=True,
            dropout_rate=0.0,
        )({"meta":{}}) == {"meta": {}, "tempo": -1.0}


class TestKeyModeParser:
    def test_normal_key(self):
        assert KeyModeParser(
            in_key="meta.key", out_key="key_mode", remove_in_key=True, dropout_rate=0.0
        )({"meta": {"key": "C#:Maj"}}) == {"meta": {}, "key_mode": "C#:Major"}

    def test_musicfm_key(self):
        assert KeyModeParser(
            in_key="meta.musicfm_plus.key",
            out_key="key_mode",
            remove_in_key=True,
            dropout_rate=0.0,
        )(
            {
                "meta": {
                    "musicfm_plus": {"key": [[1, "F:Min"], [3, "F:Min"], [5, "F:Min"]]}
                }
            }
        ) == {
            "meta": {"musicfm_plus": {}},
            "key_mode": "F:Minor",
        }

    def test_multi_in_keys(self):
        assert KeyModeParser(
            in_key=["meta.musicfm_plus.key", "meta.key"],
            out_key="key_mode",
            remove_in_key=True,
            dropout_rate=0.0,
        )(
            {
                "meta": {
                    "musicfm_plus": {"key": [[1, "F:Min"], [3, "F:Min"], [5, "F:Min"]]}
                }
            }
        ) == {
            "meta": {"musicfm_plus": {}},
            "key_mode": "F:Minor",
        }

    @pytest.mark.parametrize("key", ["" "A", "A:", "A:B:C"])
    def test_invalid_key(self, key):
        assert KeyModeParser(
            in_key="meta.key", out_key="key_mode", remove_in_key=True, dropout_rate=0.0
        )({"meta": {"key": key}}) == {"meta": {}, "key_mode": ":"}


class TestInstParser:
    def test_single_in_key(self):
        assert InstParser(
            in_key="meta.inst", out_key="inst", remove_in_key=True, dropout_rate=0.0
        )({"meta": {"inst": "guitar, bass"}}) == (
            {"meta": {}, "inst": ["guitar", "bass"]}
        )

    def test_multi_in_keys(self):
        assert InstParser(
            in_key=["meta.inst1", "meta.inst2", "meta.inst3"],
            out_key="inst",
            remove_in_key=True,
            dropout_rate=0.0,
        )({"meta": {"inst1": "", "inst2": "guitar, bass", "inst3": "drums"}}) == (
            {"meta": {}, "inst": ["guitar", "bass"]}
        )

    def test_src_musicfm_tagging(self):
        assert InstParser(
            in_key=(
                "meta.instruments",
                "meta.musicfm_tagging.instrument_section.global.instruments",
            ),
            out_key="inst",
            remove_in_key=True,
            dropout_rate=0.0,
            confidence_threshold=0.3,
        )(
            {
                "meta": {
                    "instruments": "",
                    "musicfm_tagging": {
                        "instrument_section": {
                            "global": {
                                "instruments": {
                                    "Guitar": 0.5,
                                    "Bass": 0.1,
                                    "Drums": 0.4,
                                }
                            }
                        }
                    },
                }
            }
        ) == (
            {
                "meta": {"musicfm_tagging": {"instrument_section": {"global": {}}}},
                "inst": ["Guitar", "Drums"],
            }
        )


class TestRegexDropoutTransform:

    def test_all_mode(self):
        meta = {"lyrics": "this is my lyrics"}
        transform = RegexDropoutTransform(in_key="lyrics", modes=["all"], weights=[1.0])
        assert transform(meta) == {"lyrics": ""}

    def test_none_mode(self):
        meta = {"lyrics": "this is my lyrics"}
        transform = RegexDropoutTransform(
            in_key="lyrics", modes=["none"], weights=[1.0]
        )
        assert transform(meta) == {"lyrics": "this is my lyrics"}

    def test_positive_mode(self):
        meta = {"lyrics": "[verse]\nthis is my lyrics"}
        transform = RegexDropoutTransform(
            in_key="lyrics", modes=["positive"], weights=[1.0]
        )
        assert transform(meta) == {"lyrics": "this is my lyrics"}

    def test_negative_mode(self):
        meta = {"lyrics": "[verse]\nthis is my lyrics"}
        transform = RegexDropoutTransform(
            in_key="lyrics", modes=["negative"], weights=[1.0]
        )
        assert transform(meta) == {"lyrics": "[verse]\n"}


class TestKeywordExpansion:

    def test_str(self):
        keyword_expansion = KeywordExpansion(in_key="key", out_key="key")
        assert keyword_expansion({"key": "爵士"}) == {"key": ["Jazz"]}
        assert keyword_expansion({"key": "TV Music"}) == {
            "key": ["TV Music", "Soundtrack"]
        }  # case insensitive mapping

    def test_list(self):
        keyword_expansion = KeywordExpansion(in_key="key", out_key="key")
        assert keyword_expansion({"key": ["爵士", "TV Music"]}) == {
            "key": ["Jazz", "TV Music", "Soundtrack"]
        }
    
    def test_list_rep(self):
        keyword_expansion = KeywordExpansion(in_key="key", out_key="key")
        assert keyword_expansion({"key": ["爵士", "TV Music", "Soundtrack"]}) == {
            "key": ["Jazz", "TV Music", "Soundtrack"]  # no duplication
        }

    def test_dict(self):
        keyword_expansion = KeywordExpansion(in_key="key", out_key="key")
        assert keyword_expansion(
            {"key": {"key_a": ["爵士"], "key_b": ["TV Music"]}}
        ) == {"key": {"key_a": ["Jazz"], "key_b": ["TV Music", "Soundtrack"]}}

        
class TestMultiDropoutTransform:
    def test_multi_tasks(self):
        meta = {"a": "item a", "b": "item b", "c": "item c"}
        transform = MultiDropoutTransform(
            in_key=["a", "b", "c"],
            out_key="out",
            tasks=[
                {"task": ["a", "b"], "weight": 1.0},
                {"task": ["b", "c"], "weight": 0.0},
            ],
        )
        assert transform(meta) == {
            "a": "item a",
            "b": "item b",
            "c": "item c",
            "out": {"a": "item a", "b": "item b", "c": ""},
        }
