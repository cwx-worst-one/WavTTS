from recipes.bigmusic.datasets.inference import process_user_lyrics


def test_process_user_lyrics():
    # This test only tests overwrite + reformat logic.
    # Refromat algorithm test is coverted in test_foramt_utils.
    lyrics = [
        "hello",
        "",
        "hello",
    ]
    user_lyrics = [
        "[intro]\n[verse]\nsinger1:how are you\nhello",
        "[verse]\noverwrite",
        "",
    ]
    output_lyrics_a = [
        "[intro]\n[verse] singer1:how are you\n[verse] hello",  # overwrite + reformat
        "[verse] overwrite",  # overwrite + reformat
        "hello",  # no overwrite
    ]
    output_lyrics_b = output_lyrics_a[:-1] + [""]  # the last one is empty

    assert process_user_lyrics(lyrics, user_lyrics) == output_lyrics_a  # optionally override
    assert process_user_lyrics(None, user_lyrics) == output_lyrics_b    # full override