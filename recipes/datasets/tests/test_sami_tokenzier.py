from itertools import product

import pytest
import numpy as np

from recipes.datasets.mcc.sami_tokenizer import (
    Phrase,
    SamiOfflineTokenizer,
    SamiTokenizerError,
    singer_tags,
    section_tags,
    phone_to_int,
    tone_to_int,
    convert_labels_to_text_id,
)


PHONE_MOCK_INPUT = [
    "sil\t0\tS\t0\tO\t\tS",
    "C0uo\t3\tS\t0\tO\t我\tS",
    "C0c\t2\tS\t1\tO\t曾\tB",
    "C0eng\t2\tS\t1\tO\t曾\tE",
    "C0j\t1\tS\t0\tO\t将\tB",
    "C0iang\t1\tS\t0\tO\t将\tE",
    "C0n\t3\tS\t1\tO\t你\tB",
    "C0i\t3\tS\t1\tO\t你\tE",
    "C0ch\t1\tB\t0\tO\t春天\tB",
    "C0uen\t1\tB\t0\tO\t春天\tE",
    "C0t\t1\tE\t0\tO\t\tB",
    "C0ian\t1\tE\t0\tO\t\tE",
    "C0d\t5\tS\t1\tO\t的\tB",
    "C0e\t5\tS\t1\tO\t的\tE",
    "C0n\t4\tB\t0\tO\t诺言\tB",
    "C0uo\t4\tB\t0\tO\t诺言\tE",
    "C0ian\t2\tE\t4\tO\t\tS",
    "。\t0\tS\t4\tO\t\tS",
]

PHONE_MOCK_OUTPUT = (
    np.array(
        [
            [
                2,
                250,
                265,
                263,
                166,
                200,
                265,
                263,
                172,
                207,
                265,
                263,
                176,
                204,
                265,
                263,
                167,
                245,
                265,
                182,
                206,
                265,
                263,
                168,
                196,
                265,
                263,
                176,
                250,
                265,
                206,
                265,
                263,
                85,
            ],
            [
                2,
                5,
                19,
                17,
                4,
                4,
                19,
                17,
                3,
                3,
                19,
                17,
                5,
                5,
                19,
                17,
                3,
                3,
                19,
                3,
                3,
                19,
                17,
                7,
                7,
                19,
                17,
                6,
                6,
                19,
                4,
                19,
                17,
                2,
            ],
        ]
    ),
    [
        "sil",
        "C0uo",
        "syl_sep",
        "zh_word_sep",
        "C0c",
        "C0eng",
        "syl_sep",
        "zh_word_sep",
        "C0j",
        "C0iang",
        "syl_sep",
        "zh_word_sep",
        "C0n",
        "C0i",
        "syl_sep",
        "zh_word_sep",
        "C0ch",
        "C0uen",
        "syl_sep",
        "C0t",
        "C0ian",
        "syl_sep",
        "zh_word_sep",
        "C0d",
        "C0e",
        "syl_sep",
        "zh_word_sep",
        "C0n",
        "C0uo",
        "syl_sep",
        "C0ian",
        "syl_sep",
        "zh_word_sep",
        "。",
    ],
    [
        "0",
        "3",
        "syl_sep",
        "zh_word_sep",
        "2",
        "2",
        "syl_sep",
        "zh_word_sep",
        "1",
        "1",
        "syl_sep",
        "zh_word_sep",
        "3",
        "3",
        "syl_sep",
        "zh_word_sep",
        "1",
        "1",
        "syl_sep",
        "1",
        "1",
        "syl_sep",
        "zh_word_sep",
        "5",
        "5",
        "syl_sep",
        "zh_word_sep",
        "4",
        "4",
        "syl_sep",
        "2",
        "syl_sep",
        "zh_word_sep",
        "0",
    ],
)


def test_sami_tokenizer():
    tokens, phonemes, tones = convert_labels_to_text_id(PHONE_MOCK_INPUT)
    gt_tokens, gt_phonemes, gt_tones = PHONE_MOCK_OUTPUT
    assert np.array_equal(tokens, gt_tokens)
    assert phonemes == gt_phonemes
    assert tones == gt_tones


def _get_mock_phrase(section_tag, singer_tag):
    # prepend the singer tag to the beginning of the first line
    if section_tag and singer_tag:
        mock_input = [f"[{section_tag}] {singer_tag}:{PHONE_MOCK_INPUT[0]}"] + PHONE_MOCK_INPUT[1:]
    elif section_tag:
        mock_input = [f"[{section_tag}] {PHONE_MOCK_INPUT[0]}"] + PHONE_MOCK_INPUT[1:]
    elif singer_tag:
        mock_input = [f"{singer_tag}:{PHONE_MOCK_INPUT[0]}"] + PHONE_MOCK_INPUT[1:]
    else:
        mock_input = PHONE_MOCK_INPUT
    return Phrase.parse(phonemes="\n".join(mock_input))


@pytest.mark.parametrize("prefix_tags", product([None] + section_tags, [None] + singer_tags))
def test_sami_tokenizer_with_prefix_tags(prefix_tags):
    section_tag, singer_tag = prefix_tags
    phrase = _get_mock_phrase(section_tag, singer_tag)
    tokens, phonemes, tones = convert_labels_to_text_id(phrase.phonemes.split("\n"), phrase.prefix_tags)
    gt_tokens, gt_phonemes, gt_tones = PHONE_MOCK_OUTPUT

    section_phone_id = phone_to_int.get(section_tag)
    section_tone_id = tone_to_int.get(section_tag)
    singer_phone_id = phone_to_int.get(singer_tag)
    singer_tone_id = tone_to_int.get(singer_tag)

    prepend_tokens = list(filter(None, [section_tag, singer_tag]))
    prepend_phone_ids = list(filter(None, [section_phone_id, singer_phone_id]))
    prepend_tone_ids = list(filter(None, [section_tone_id, singer_tone_id]))

    assert np.array_equal(tokens, np.hstack([[prepend_phone_ids, prepend_tone_ids], gt_tokens]))
    assert phonemes == prepend_tokens + gt_phonemes
    assert tones == prepend_tokens + gt_tones


def test_tokenize_phrase():
    section_tag = "verse"
    singer_tag = "合"
    phrase = _get_mock_phrase(section_tag, singer_tag)

    section_phone_id = phone_to_int.get(section_tag)
    singer_phone_id = phone_to_int.get(singer_tag)

    prepend_phone_ids = [section_phone_id, singer_phone_id]

    gt_tokens, _, _ = PHONE_MOCK_OUTPUT

    tokenizer = SamiOfflineTokenizer()
    assert np.array_equal(tokenizer.tokenize_phrase(phrase), np.concatenate([prepend_phone_ids, gt_tokens[0]]))


def test_tokenize_phrase_invalid_inputs():
    tokenizer = SamiOfflineTokenizer()

    with pytest.raises(SamiTokenizerError):
        tokenizer.tokenize_phrase(Phrase.parse(""))  # empty phrase

    with pytest.raises(SamiTokenizerError):
        tokenizer.tokenize_phrase(Phrase.parse("[verse] abc"))  # invalid phoneme label


ALT_VERSE_SECTION_TAGS = [
    "[verse]",
    "(VERSE)",
    "（Verse）",
    "<Verse>",
    "{Verse}",
    "【verse】",
    "《verse》",
    "「verse」",
]

@pytest.mark.parametrize("section_tag", ALT_VERSE_SECTION_TAGS)
def test_tag_normalization_verse(section_tag):
    phrase = Phrase.parse(text=f"{section_tag} 我的歌词", normalize_tag=True)
    assert phrase.section_tag == "verse"
    assert phrase.text == "我的歌词"


def test_tag_normalization_silent_none():
    phrase = Phrase.parse(text=f"(no_such_section) 我的歌词", normalize_tag=True)
    assert phrase.section_tag is None
    assert phrase.text == "我的歌词"


def test_parse_text():
    phrases = Phrase.parse_text("(Verse)第一句 [CHORUS] 第二句<bridge> 第三句")
    assert (phrases[0].text, phrases[0].section_tag) == ("第一句", "verse")
    assert (phrases[1].text, phrases[1].section_tag) == ("第二句", "chorus")
    assert (phrases[2].text, phrases[2].section_tag) == ("第三句", "bridge")

# TODO (Yilin) Test SamiTokenizer