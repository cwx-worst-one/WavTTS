import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def plot_mel(mel, t=None):
    fig, ax = plt.subplots(figsize=(10,2))
    im = ax.imshow(mel, aspect="auto", origin="lower",
                    interpolation='none')
    if t is not None:
        ax.text(mel.shape[1] / 2, mel.shape[0] / 2, f't={t}', 
                color='black', fontsize=16, ha='center', va='center')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data

def plot_mel_local(mel, output_path):
    fig, ax = plt.subplots(figsize=(10,2))
    im = ax.imshow(mel, aspect="auto", origin="lower",
                    interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    plt.savefig(output_path)
    return

class ode_wrapper(torch.nn.Module):
    """
    ODE solver only accepts two arguments, t and x. However, some models may have other conditional inputs, which are time-invariant.
    Here, time refers to the time step of the ODE solver, which is different from the time dimension of the data (e.g., audio time).
    Therefore, we need to wrap the model to a ode compatible format.
    """
    def __init__(self, model, conditional_inputs=None, use_guidance=False, guidance_strength=0.3):
        """
        Args:
            model: model to be wrapped
            conditional_inputs: a dictionary of conditional inputs, e.g., control inputs
            use_guidance: whether to use classifier-free guidance according to section 3.5 in https://dl.fbaipublicfiles.com/voicebox/paper.pdf
            guidance_strength: the strength of the guidance defined in eq. (8) in https://dl.fbaipublicfiles.com/voicebox/paper.pdf
        """
        super().__init__()
        self.model = model
        self.conditional_inputs = conditional_inputs
        self.use_guidance = use_guidance
        self.guidance_strength = guidance_strength

    def forward(self, t, x):
        """
        Args:
            t: time step, single number
            x: input tensor, shape (batch_size, ...)
        """
        t = t.repeat(x.shape[0])[:, None]
        if self.conditional_inputs is not None:
            if self.use_guidance:
                # generate zero condtional inputs
                zero_conditional_inputs = {}
                for key, value in self.conditional_inputs.items():
                    if type(value) == dict:
                        zero_conditional_inputs[key] = {}
                        for k, v in value.items():
                            zero_conditional_inputs[key][k] = torch.zeros_like(v)
                    else:
                        zero_conditional_inputs[key] = torch.zeros_like(value)
                # # eq. (8) in https://dl.fbaipublicfiles.com/voicebox/paper.pdf

                x = ( 1 + self.guidance_strength) * self.model.inference(t=t, x=x, **self.conditional_inputs) \
                    - self.guidance_strength * self.model.inference(t=t, x=x, **zero_conditional_inputs) 
            else:
                x = self.model.inference(t=t, x=x, **self.conditional_inputs)
            return x
        else:
            return self.model.inference(x=x, t=t)


@rank_zero_only
def log_audio(wandb_logger: WandbLogger, key, audios, uttids, sample_rate, step, rank):
    metrics = {key:[wandb.Audio(audio,caption=f"{step=}/{rank=}/{uttid}",sample_rate=sample_rate) for uttid, audio in zip(uttids, audios)]}
    wandb_logger.log_metrics(metrics, step)