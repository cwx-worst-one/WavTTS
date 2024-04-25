from recipes.bigmusic.datasets.inference import process_zh_lyrics, multitags_to_speaker_ids, process_zh_style_text


def test_multitags_to_speaker_ids():
    speaker_ids = [0, 3, 0, 0]
    assert multitags_to_speaker_ids(speaker_ids, [49, 48, 48, 47]) == [49, 3, 48, 47]


def test_zh_style_text():
    style_text_list = [
        "Pop|Chill|Female",  # Female
        "MC|Chill|Male",  # Female
    ]
    tags, keys, tempo_labels, speaker_ids = process_zh_style_text(style_text_list, "multi_tag")
    for tag in tags:
        assert "|" in tag
    # TODO: support key and tempo completion
    assert keys == ["N", "N"]
    assert tempo_labels == ["", ""]

    assert speaker_ids == [49, 48]
