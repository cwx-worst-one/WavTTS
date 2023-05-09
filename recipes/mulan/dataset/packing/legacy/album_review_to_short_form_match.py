import pickle

from big_csv_loader import parallel_load_and_process
from tqdm import tqdm

# load pickle
with open("album_data_dict.pkl", "rb") as f:
    data_dict = pickle.load(f)

ks = list(data_dict.keys())
print(len(ks))
cnt_dict = {}
for idx in range(200, 300):
    print(ks[idx])
    for k in data_dict[ks[idx]]:
        print(k, data_dict[ks[idx]][k])
    print("-----------------")
print(len(cnt_dict.keys()))
assert 1 == 2

sf_path = "../2000w_music_info_meta_v2.csv"
df_sf = parallel_load_and_process(sf_path)
# df_sf = pd.read_csv(sf_path, nrows=50000)
review_path = "../allmusic_album.csv"
df_r = parallel_load_and_process(review_path)
# df_r = pd.read_csv(review_path, nrows=50000)


print("parsing short form data")
sf_dict = {}
df_title = df_sf["meta_song_title"].to_list()
df_author = df_sf["meta_song_author"].to_list()
df_album = df_sf["meta_song_album_name"].to_list()
df_song_id = df_sf["meta_song_id"].to_list()

for i in tqdm(range(len(df_sf))):
    title = str(df_title[i]).lower()
    author = str(df_author[i]).lower()
    album = str(df_album[i]).lower()
    song_id = str(df_song_id[i])

    sf_dict[author] = sf_dict.get(author, {})
    sf_dict[author][album] = sf_dict[author].get(album, {})
    if title not in sf_dict[author][album]:
        sf_dict[author][album][title] = song_id

print("parsing review data")
r_dict = {}
df_artist = df_r["artist"].to_list()
df_album = df_r["album"].to_list()
df_review = df_r["review"].to_list()
df_genre = df_r["genre"].to_list()
df_styles = df_r["styles"].to_list()
df_release_date = df_r["release_date"].to_list()
df_album_moods = df_r["album_moods"].to_list()
df_album_themes = df_r["album_themes"].to_list()

for i in tqdm(range(len(df_r))):
    artist = str(df_artist[i]).lower()
    album = str(df_album[i]).lower()
    review = str(df_review[i])
    genre = str(df_genre[i])
    styles = str(df_styles[i])
    release_date = str(df_release_date[i])
    album_moods = str(df_album_moods[i])
    album_themes = str(df_album_themes[i])

    r_dict[artist] = r_dict.get(artist, {})
    if album not in r_dict[artist]:
        r_dict[artist][album] = {
            "review": review,
            "genre": genre,
            "styles": styles,
            "release_date": release_date,
            "album_moods": album_moods,
            "album_themes": album_themes,
        }

print("matching")
data_dict = {}
for artist in tqdm(r_dict):
    if artist in sf_dict:
        for album in r_dict[artist]:
            if album in sf_dict[artist]:
                for title in sf_dict[artist][album]:
                    song_id = sf_dict[artist][album][title]
                    data_dict[song_id] = {
                        "review": r_dict[artist][album]["review"].replace("\\", ""),
                        "genre": r_dict[artist][album]["genre"]
                        .replace("[", "")
                        .replace("]", "")
                        .replace("'", ""),
                        "styles": r_dict[artist][album]["styles"]
                        .replace("[", "")
                        .replace("]", "")
                        .replace("'", ""),
                        "release_date": r_dict[artist][album]["release_date"],
                        "album_moods": r_dict[artist][album]["album_moods"]
                        .replace("[", "")
                        .replace("]", "")
                        .replace("'", ""),
                        "album_themes": r_dict[artist][album]["album_themes"]
                        .replace("[", "")
                        .replace("]", "")
                        .replace("'", ""),
                    }

# save data_dict to pickle
with open("data_dict.pkl", "wb") as f:
    pickle.dump(data_dict, f)
