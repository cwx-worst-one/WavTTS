from recipes.bigmusic.datasets.inference import split_lyrics_by_section_tags


def test_split_and_normalize_lyrics():
    lyrics = [
        "\n".join([
            "[verse] This is verse (Chorus)This is chorus",
            "<bridge> This is bridge",
        ]),
        "\n".join([
            "[verse] This is verse (Chorus)This is chorus<bridge> This is bridge",
            "[OUTRO]"
        ])
    ]

    reformatted_lyrics = split_lyrics_by_section_tags(lyrics)

    assert reformatted_lyrics == [
        "\n".join([
            "[verse] This is verse",
            "(Chorus)This is chorus",
            "<bridge> This is bridge"
        ]),
        "\n".join([
            "[verse] This is verse",
            "(Chorus)This is chorus",
            "<bridge> This is bridge",
            "[OUTRO]",
        ])
    ]