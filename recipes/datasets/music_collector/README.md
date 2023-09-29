# SpotDL Dataset


## Quickstart
```bash

# Substitute the `spotify_id` with the artist's Spotify ID
python3 -m recipes.datasets.spotdl.create_webdataset --download --spotify_id 74ASZWbe4lXaubB36ztrGX --spotify_type artist

# To only create webdataset:
python3 -m recipes.datasets.spotdl.create_webdataset --spotify_id 74ASZWbe4lXaubB36ztrGX --spotify_type artist
```