from typing import List, Optional, Union

from recipes.bigmusic.datasets.utils.zh_lyrics_proc import SongLyrics 


def select_genre_in_list(
    genre: Optional[Union[str, List[str]]], supported_genres, logger=None
) -> str:
    default_genre = "Pop"
    input_genre = genre
    # logger.info("Input genre {}".format(input_genre))

    # Use default if the input genre is empty
    if not genre:
        # logger.info("Input genre is empty, use default {}".format(default_genre))
        return default_genre

    # Filter genre list according to the given supported genres
    if isinstance(genre, str):
        genre = [genre]
    genre = [tag for tag in genre if tag in supported_genres]
    if not genre:
        genre = default_genre
        # logger.info(
        #     'input genre {} is not supported, use "{}"'.format(
        #         input_genre, default_genre
        #     )
        # )
    else:
        genre = genre[0]
    # logger.info('Select genre "{}"'.format(genre))
    return genre


def infer_is_full_song_from_structure(n_vocal_sections: int) -> bool:
    return n_vocal_sections > 3


def infer_is_full_song_from_lyrics(lyrics: str) -> bool:
    song_lyrics = SongLyrics.parse(lyrics)
    return infer_is_full_song_from_structure(
        len([paragraph for paragraph in song_lyrics if paragraph.has_utterance])
    )