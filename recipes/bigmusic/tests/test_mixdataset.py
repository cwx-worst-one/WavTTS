from tqdm import tqdm
from recipes.bigmusic.datasets.mix import MusicCollectorDataset, MusicCollectorWebDataModule


def test_mixdataset_music_collector():
    datamodule = MusicCollectorWebDataModule(
        dataset_names=["Shania Twain"],
        dataset_weights=[1],
        num_workers=8,
    )

    train_loader = datamodule.train_dataloader()

    for batch in tqdm(train_loader):
        continue