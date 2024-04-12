import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb

def get_wandb_logger(loggers):
    """Extract the wandb logger from a list of loggers."""
    assert loggers is not None
    if isinstance(loggers, pl.loggers.wandb.WandbLogger):
        # loggers contains only one logger of type WandbLogger. Return this one.
        return loggers
    else:
        # loggers is a list with wandb and tensorboard. It is not clear which index belongs to wandb.
        for logger in loggers:
            if isinstance(logger, pl.loggers.wandb.WandbLogger):
                # only get the wandb logger
                return logger
        return None

def get_list_of_mel_spec_plots_to_log(mel_spec, num_samples_to_plot):
    """Get a list of Mel Specs for logging into one window on ByteDance Merlin."""
    return get_list_of_spectrogram_plots_to_log(mel_spec, num_samples_to_plot, title="Mel Spec")

def get_list_of_chroma_spec_plots_to_log(chroma_spec, num_samples_to_plot):
    """Get a list of Chroma Specs for logging into one window on ByteDance Merlin."""
    return get_list_of_spectrogram_plots_to_log(chroma_spec, num_samples_to_plot, title="Chroma Spec")

def get_list_of_spectrogram_plots_to_log(spectrograms, num_samples_to_plot, title):
    """
    Return a list of wandb.Image() objects which can be directly logged to a wandb_logger.

    12APR2024 @hanoihantrakul
    I have to do it this way because the plt object has to be opened and closed for each
    separate plot. Then each plot is converted to its own unique wandb.Image() instance.
    By returning a list and passing this into the wandb_logger, each individual spectrogram
    will be printed to the Merlin console in the same window in a neat manner. 
    """
    wandb_image_list = []
    for i in range(num_samples_to_plot):
        plt.figure(figsize=(16, 8))
        plt.pcolor(spectrograms[i].cpu().T, vmin=-6, vmax=0.5)
        plt.title(f"{title} {i}")
        wandb_image_list.append(wandb.Image(plt))
        plt.close()
    return wandb_image_list


def deprecated_create_spectrogram_plt(x):
    """
    12APR2024 @hanoihantrakul 
    DO NOT USE THIS FUNCTION
    I leave this as an example here to show how NOT to
    save spectrograms to wandb on ByteDance infrastructure.
    The issue here is the plt object does not get closed properly
    and so multiple plots end up looking the same.
    """
    plt.close() 
    plt.figure().clear()
    fig = plt.figure(figsize=(16, 8))
    plt.pcolor(x.cpu().T, vmin=-6, vmax=0.5)
    return plt

def get_list_of_callable_methods(x):
    return [func for func in dir(x) if callable(getattr(x, func))]
    

