import torch
import gradio as gr

from recipes.byteformers_example.scripts.generate import generate
from recipes.musiclm.inference.utils import load_config

device = "cuda"
cfg = load_config("./recipes/byteformers_example/conf/finetune.yaml")

tokenizer = cfg.tokenizer
pl_module = cfg.pl_module.to(device)
pl_module.eval()


with gr.Blocks() as demo:
    top_p = gr.Slider(
        minimum=0,
        maximum=1.0,
        value=0.95,
        step=0.05,
        interactive=True,
        label="Top-p",
        visible=True,
    )
    temperature = gr.Slider(
        minimum=0,
        maximum=5.0,
        value=0.8,
        step=0.1,
        interactive=True,
        label="Temperature",
        visible=True,
    )
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def respond_stream(message, top_p, temperature, history=[]):
        history = list(map(tuple, history))
        stream = generate(
            message,
            tokenizer,
            pl_module,
            max_new_tokens=512,
            top_k=None,
            top_p=top_p,
            temperature=temperature,
        )
        response = []
        for token in stream:
            response.append(token)
            yield "", [
                (message, tokenizer.decode(torch.tensor(response, dtype=torch.long)))
            ]

    msg.submit(respond_stream, [msg, top_p, temperature], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue(concurrency_count=16)
demo.launch(share=True)
