from recipes.mir.datasets.single import DialectDataModule, ChineseGenreDataModule

batch_size = 8
shuffle_buffer_size = 10
num_workers = 0


def test_dialect_dataset():
    datamodule = DialectDataModule(
        batch_size, shuffle_buffer_size, num_workers=num_workers
    )
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

def test_chinese_genre_dataset():
    datamodule = ChineseGenreDataModule(
        batch_size, shuffle_buffer_size, num_workers=num_workers
    )
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    breakpoint()