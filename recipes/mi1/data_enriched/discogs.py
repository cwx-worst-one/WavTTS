import os
import time

import discogs_client
import pandas as pd
from discogs_client.exceptions import HTTPError
from joblib import Parallel, delayed
import logging
from tqdm import tqdm
from typing import List
from uuid import uuid4

logger = logging.getLogger(__name__)

class DiscogsScraper:
    def __init__(self):
        self._client = discogs_client.Client(
            "TheRecordIndustry/0.1", user_token="wMDhArCrEPkajsNzYlTVhTIoxiCboLpxHnsertLz"
        )

    def search(self, album_name: str, artist_name: str):
        results = self._client.search(
            release_title=album_name, artist=artist_name, type="release"
        )
        if not len(results):
            return None
        return results


import re
def album_name_variations(album_name: str) -> List[str]:
    album_name_no_brackets = re.sub(r'\([^)]*\)', '', album_name)
    return [album_name, album_name_no_brackets]


def get_discogs_data(scraper: DiscogsScraper, album_name: str, artist_name: str) -> dict:
    discogs_entries = []
    for album_name in album_name_variations(album_name):
        # try a few variations, break if we found one

        results = scraper.search(album_name, artist_name)
        if results is None:
            continue

        # print(len(results), result, result.genres, result.styles)
        for result_idx, result in enumerate(results):
            discogs_data = {}
            discogs_data["title"] = result.title
            discogs_data["artists"] = result.artists[0].name
            discogs_data["year"] = result.year
            discogs_data["year"] = result.year
            discogs_data["genres"] = result.genres
            discogs_data["styles"] = result.styles
            discogs_data["result_search_idx"] = result_idx
            discogs_entries.append(discogs_data)
            break # TODO: we'll do a single result for the prototype
        break

    return discogs_entries



def parallel_fetch(idx: int, row, scraper: DiscogsScraper, n_attempts: int = 5, sleep_sec: int = 5):
    global discogs_data_df
    spotify_album_id = row.metadata["spotify_album_id"]
    album_name = row.metadata["spotify_album_name"]
    artist_name = row.metadata["spotify_primary_artist_name"]

    attempt = 0
    discogs_entries = None
    while attempt < n_attempts:
        try:
            discogs_entries = get_discogs_data(scraper, album_name, artist_name)
            if len(discogs_entries):
                break

        except Exception as e:
            print(e)
            time.sleep(sleep_sec)

        attempt += 1

    if discogs_entries is None:
        return

    for idx in range(len(discogs_entries)):
        discogs_entries[idx]["spotify_album_id"] = spotify_album_id

    discogs_data_df = pd.concat(
    (
        discogs_data_df,
        pd.DataFrame(discogs_entries),
    ))

    if idx % 50 == 0:
        discogs_data_df.to_pickle(OUT_FP)

if __name__ == "__main__":
    fp = "billboard_200_train_index.p"
    index = pd.read_pickle(fp)
    OUT_FP = fp + ".discogs"

    index["spotify_album_id"] = index.metadata.apply(lambda r: r["spotify_album_id"])
    dedup = index.drop_duplicates("spotify_album_id")

    if os.path.exists(OUT_FP):
        discogs_data_df = pd.read_pickle(OUT_FP)
        dedup = dedup[
            ~dedup["spotify_album_id"].isin(discogs_data_df["spotify_album_id"])
        ]
    else:
        discogs_data_df = pd.DataFrame()

    scraper = DiscogsScraper()

    Parallel(n_jobs=8, prefer="threads")(
        delayed(parallel_fetch)(idx, row, scraper)
        for idx, (_, row) in enumerate(tqdm(dedup.iterrows(), total=len(dedup)))
    )

    print(discogs_data_df)
    discogs_data_df.to_pickle(OUT_FP)
