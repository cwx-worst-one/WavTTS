import gradio as gr
from io import BytesIO
import base64

from recipes.musiclm.datasets.mcc import PGCDataset
from webdataset.pipeline import DataPipeline
import webdataset as wds
import numpy as np
import wavio

DURATION=30

dataset = DataPipeline(
    PGCDataset(
        duration=DURATION,
        normalize_audio=True,
        resampled=True,
        shardshuffle=True,
        max_num_crops=1,
    ),
    wds.shuffle(100),
)
it = iter(dataset)


def fetch_music():
    item = next(it)
    audio_bytes = BytesIO()
    wavio.write(audio_bytes, item["audio"][0].numpy().astype(np.float32), 24000, sampwidth=4)
    audio_bytes.seek(0)

    audio_base64 = base64.b64encode(audio_bytes.read()).decode("utf-8")
    audio_player = f'<audio src="data:audio/mpeg;base64,{audio_base64}" controls autoplay></audio>'
    audio_player += (
        f'<tr><br><td>song_title</td><br><td>{item["meta"]["song_title"]}</td><br></tr><br><tr><br><td>artist_name</td><br><td>{item["meta"]["artist_name"]}</td><br></tr><br><tr><br><td>meta_song_id</td><br><td>{item["meta"]["meta_song_id"]}</td><br></tr><br><tr><br><td>genre</td><br><td>{item["meta"]["genre"]}</td><br></tr><br><tr><br><td>theme</td><br><td>{item["meta"]["theme"]}</td><br></tr><br><tr><br><td>mood</td><br><td>{item["meta"]["mood"]}</td><br></tr><br>'
    )
    return audio_player


with gr.Blocks() as demo:
    html = gr.HTML()
    demo.load(fetch_music, inputs=None, outputs=[html], every=DURATION)

demo.queue(concurrency_count=10).launch(share=True)
