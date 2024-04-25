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
    phonetone_to_int,
    VOCAB_TYPES,
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

PHONE_TYPE_MOCK_OUTPUT = (
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
            ]
        ]
        * 2
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
)


PHONE_TONE_TYPE_MOCK_OUTPUT = (
    np.array(
        [
            [
                2,
                1401,
                1595,
                1593,
                418,
                650,
                1595,
                1593,
                424,
                754,
                1595,
                1593,
                428,
                711,
                1595,
                1593,
                419,
                1324,
                1595,
                434,
                739,
                1595,
                1593,
                420,
                593,
                1595,
                1593,
                428,
                1402,
                1595,
                740,
                1595,
                1593,
                85,
            ]
        ]
        * 2
    ),
    [
        "sil",
        "C0uo@3",
        "syl_sep",
        "zh_word_sep",
        "C0c",
        "C0eng@2",
        "syl_sep",
        "zh_word_sep",
        "C0j",
        "C0iang@1",
        "syl_sep",
        "zh_word_sep",
        "C0n",
        "C0i@3",
        "syl_sep",
        "zh_word_sep",
        "C0ch",
        "C0uen@1",
        "syl_sep",
        "C0t",
        "C0ian@1",
        "syl_sep",
        "zh_word_sep",
        "C0d",
        "C0e@5",
        "syl_sep",
        "zh_word_sep",
        "C0n",
        "C0uo@4",
        "syl_sep",
        "C0ian@2",
        "syl_sep",
        "zh_word_sep",
        "。",
    ],
)


MOCK_OUTPUTS = {
    "phoneme": PHONE_TYPE_MOCK_OUTPUT,
    "phoneme+tone": PHONE_TONE_TYPE_MOCK_OUTPUT,
}


VOCAB_MAP = {"phoneme": phone_to_int, "phoneme+tone": phonetone_to_int}


@pytest.mark.parametrize("vocab_type", VOCAB_TYPES)
def test_sami_tokenizer(vocab_type):
    tokens, phonemes, _ = convert_labels_to_text_id(
        PHONE_MOCK_INPUT, vocab_type=vocab_type
    )
    gt_tokens, gt_phonemes = MOCK_OUTPUTS[vocab_type]
    assert np.array_equal(tokens, gt_tokens)
    assert phonemes == gt_phonemes


def _get_mock_phrase(section_tag, singer_tag):
    # prepend the singer tag to the beginning of the first line
    if section_tag and singer_tag:
        mock_input = [
            f"[{section_tag}] {singer_tag}:{PHONE_MOCK_INPUT[0]}"
        ] + PHONE_MOCK_INPUT[1:]
    elif section_tag:
        mock_input = [f"[{section_tag}] {PHONE_MOCK_INPUT[0]}"] + PHONE_MOCK_INPUT[1:]
    elif singer_tag:
        mock_input = [f"{singer_tag}:{PHONE_MOCK_INPUT[0]}"] + PHONE_MOCK_INPUT[1:]
    else:
        mock_input = PHONE_MOCK_INPUT
    return Phrase.parse(phonemes="\n".join(mock_input))


@pytest.mark.parametrize(
    "prefix_tags", product([None] + section_tags, [None] + singer_tags)
)
def test_sami_tokenizer_with_prefix_tags(prefix_tags):
    # test on "phoneme" vocab type only
    section_tag, singer_tag = prefix_tags
    phrase = _get_mock_phrase(section_tag, singer_tag)
    tokens, phonemes, _ = convert_labels_to_text_id(
        phrase.phonemes.split("\n"), phrase.prefix_tags
    )
    gt_tokens, gt_phonemes = PHONE_TYPE_MOCK_OUTPUT

    section_phone_id = None if section_tag is None else phone_to_int[section_tag]
    singer_phone_id = None if singer_tag is None else phone_to_int[singer_tag]

    prepend_tokens = list(filter(None, [section_tag, singer_tag]))
    prepend_phone_ids = list(filter(None, [section_phone_id, singer_phone_id]))

    assert np.array_equal(
        tokens, np.hstack([[prepend_phone_ids, prepend_phone_ids], gt_tokens])
    )
    assert phonemes == prepend_tokens + gt_phonemes


@pytest.mark.parametrize("vocab_type", VOCAB_TYPES)
def test_tokenize_phrase(vocab_type):
    section_tag = "verse"
    singer_tag = "合"
    phrase = _get_mock_phrase(section_tag, singer_tag)

    section_phone_id = VOCAB_MAP[vocab_type][section_tag]
    singer_phone_id = VOCAB_MAP[vocab_type][singer_tag]

    prepend_phone_ids = [section_phone_id, singer_phone_id]

    gt_tokens, _ = MOCK_OUTPUTS[vocab_type]

    tokenizer = SamiOfflineTokenizer(vocab_type=vocab_type)
    assert np.array_equal(
        tokenizer.tokenize_phrase(phrase),
        np.concatenate([prepend_phone_ids, gt_tokens[0]]),
    )


@pytest.mark.parametrize("vocab_type", VOCAB_TYPES)
def test_tokenize_phrase_invalid_inputs(vocab_type):
    tokenizer = SamiOfflineTokenizer(vocab_type=vocab_type)

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


# TODO (Yilin) Test SamiTokenizer
