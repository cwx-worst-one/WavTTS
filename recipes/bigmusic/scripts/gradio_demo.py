import sys
from time import perf_counter
import torch
import gradio as gr

from hyperpyyaml import load_hyperpyyaml

from samantha.utils.hdfs_tools import hdfs_open
from samantha.utils.hparams import DotDict
from recipes.bigmusic.datasets.inference import inference_dataset_from_prompt


def torch_fp32_to_numpy_int16(audio: torch.Tensor):
    return (audio * 32767).to(torch.int16).T.cpu().numpy()

def generate_audio(
    lyrics: str,
    genre: str,
    mood: str,
    gender: str,
):
    global vocal_model
    params = vocal_model._hparams.extra_params
    sample_rate = params["sample_rate"]
    tik = perf_counter()

    n_samples = 4
    style_prompt_metadata = {}
    style_prompt_metadata['final_genre'] = genre
    style_prompt_metadata['final_mood'] = mood
    style_prompt_metadata['merge_aed'] = gender

    prompts = {'metadata': [style_prompt_metadata] * n_samples,
                'lyrics': [lyrics] * n_samples}
    inference_dataset = inference_dataset_from_prompt(
        prompts, conditions="style_text,lyrics_tokens",
        batch_size=n_samples,
        lyrics_max_seq_len=params["lyrics_max_seq_len"])
    batch = next(iter(inference_dataset))
    out_dict = vocal_model.predict_step(batch)
    sampled_audio = out_dict['generated_audio']

    tok = perf_counter()
    rtf = (tok - tik) / params["duration"]
    sampled_audio_0 = torch_fp32_to_numpy_int16(sampled_audio[0])
    sampled_audio_1 = torch_fp32_to_numpy_int16(sampled_audio[1])
    sampled_audio_2 = torch_fp32_to_numpy_int16(sampled_audio[2])
    sampled_audio_3 = torch_fp32_to_numpy_int16(sampled_audio[3])

    return ((sample_rate, sampled_audio_0),
            (sample_rate, sampled_audio_1),
            (sample_rate, sampled_audio_2),
            (sample_rate, sampled_audio_3), rtf)


def get_model(hparams_file):
    if hparams_file.startswith("hdfs"):
        with hdfs_open(hparams_file, "r") as fin:
            hparams = load_hyperpyyaml(fin)
    else:
        with open(hparams_file, "r", encoding="utf-8") as fin:
            hparams = load_hyperpyyaml(fin)
    cfg = DotDict(hparams)
    # pl_datamodule = cfg.pl_datamodule
    pl_module = cfg.pl_module
    pl_module = pl_module.to('cuda')
    # ref_sample = next(iter(pl_datamodule.predict_dataset))
    return pl_module


if __name__ == "__main__":
    lyrics = gr.Textbox(
        value="Imagine there's no countries\nIt isn't hard to do\nNothing to kill or die for\nAnd no religion, too\nImagine all the people\nLivin' life in peace",
        label="Lyrics",
    )
    genre = gr.Dropdown(
        choices=['nan', 'Rock', 'Pop', 'EDM', 'R&B', 'Country', 'Jazz',
                 'Reggae', 'Blues', 'Trap Rap', 'Metal', 'New Age'],
                 value='Pop', label="genre"
    )
    mood = gr.Dropdown(
        choices=['nan', 'Happy', 'Chill', 'Cute', 'Sweet', 'Romantic',
                 'Excited', 'Dynamic', 'Lonely', 'Sorrow', 'Angry', 'Tense'],
                 value='Chill', label="mood"
    )
    gender = gr.Dropdown(
        choices=['nan', 'Female', 'Male'], value='Female', label="gender"
    )
    demo = gr.Interface(
        fn=generate_audio,
        inputs=[
            lyrics,
            genre,
            mood,
            gender
        ],
        outputs=[
            gr.Audio(label="Generated 0"),
            gr.Audio(label="Generated 1"),
            gr.Audio(label="Generated 2"),
            gr.Audio(label="Generated 3"),
            gr.Textbox(label="RTF"),
        ],
        title="vocal music",
        description="vocal music",
    )
    demo.queue(concurrency_count=4)

    # initialize model
    vocal_model = get_model(sys.argv[1])
    print("model loading ready")

    demo.launch(share=True)
