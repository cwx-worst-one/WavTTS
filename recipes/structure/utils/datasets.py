import functools

import braceexpand
import webdataset
from torch.utils.data import DataLoader

from recipes.structure.preprocess.common import StructurePreprocessor
from samantha.utils.webdataset import apply_webdataset_pipeline
from recipes.beat.utils.dataset import MCC_dataset, GalaxyArk_dataset, hot_galaxy_dataset, GalaxyArk_new_dataset


def get_train_pipeline(preprocessor, shuffle_buffer, batch_size, num_iter, to_tuple):
    pipeline = []

    pipeline.append(functools.partial(webdataset.WebDataset.shuffle, shuffle_buffer))
    pipeline.append(functools.partial(webdataset.WebDataset.decode, handler=webdataset.warn_and_stop))

    proprocess = functools.partial(
        StructurePreprocessor.train_batch_preprocess, preprocessor
    )
    pipeline.append(functools.partial(webdataset.WebDataset.compose, proprocess))
    pipeline.append(functools.partial(webdataset.WebDataset.shuffle, shuffle_buffer))
    pipeline.append(functools.partial(webdataset.WebDataset.to_tuple, to_tuple))
    pipeline.append(functools.partial(webdataset.WebDataset.batched, batch_size))
    pipeline.append(functools.partial(webdataset.WebDataset.with_epoch, num_iter))
    return pipeline


def get_val_pipeline(preprocessor):
    pipeline = []
    pipeline.append(functools.partial(webdataset.WebDataset.decode))
    proprocess = functools.partial(StructurePreprocessor.val_preprocess, preprocessor)
    pipeline.append(functools.partial(webdataset.WebDataset.map, proprocess))
    pipeline.append(
        functools.partial(
            webdataset.WebDataset.to_tuple,
            "audio boundary_label function_label chorus_only \
             boundary_interval chorus_interval segment_type __key__",
        )
    )
    pipeline.append(functools.partial(webdataset.WebDataset.batched, 1))
    return pipeline


def supervised_dataset():
    supervised_urls = []
    urls = [
        "/mnt/bn/mir-tasks/structure/billboard_structure/train/shards-{0000..0002}.tar",
        "/mnt/bn/mir-tasks/structure/isophonic_structure/train/shards-{0000..0001}.tar",
        "/mnt/bn/mir-tasks/structure/salamijazz_structure/train/" \
            "shards-{0000..0002}.tar",
        "/mnt/bn/mir-tasks/structure/salamipop_structure/train/shards-{0000..0001}.tar",
        "/mnt/bn/mir-tasks/structure/harmonix_structure/train/shards-{0000..0003}.tar",
        "/mnt/bn/mir-tasks/structure/rwc_structure/train/shards-{0000..0001}.tar",
        "/mnt/bn/mir-tasks/structure/salamilive_structure/train/" \
            "shards-{0000..0004}.tar",
        "/mnt/bn/mir-tasks/structure/bytechorus_structure/train/" \
            "shards-{0000..0011}.tar",
        "/mnt/bn/mir-tasks/structure/pop909_structure/train/shards-{0000..0004}.tar",
        "/mnt/bn/mir-tasks/structure/hooktheory_structure/train/" \
            "shards-{0000..0015}.tar",
    ]
    for url in urls:
        supervised_urls = supervised_urls + list(braceexpand.braceexpand(url))

    return supervised_urls


def validation_dataset():
    supervised_urls = []
    urls = [
        "/mnt/bn/mir-tasks/structure/salamipop_structure/validation/shards-0000.tar",
        "/mnt/bn/mir-tasks/structure/harmonix_structure/validation/shards-0000.tar",
        "/mnt/bn/mir-tasks/structure/rwc_structure/validation/shards-0000.tar",
        "/mnt/bn/mir-tasks/structure/bytechorus_structure/validation/" \
            "shards-{0000..0001}.tar",
        "/mnt/bn/mir-tasks/structure/hooktheory_structure/validation/shards-0000.tar",
    ]
    for url in urls:
        supervised_urls = supervised_urls + list(braceexpand.braceexpand(url))

    return supervised_urls


def testing_dataset():
    url = ["/mnt/bn/mir-tasks/structure/pop909_structure/test/shards-0000.tar"]
    return url


def get_train_dataset(
    dataset_names,
    preprocessor,
    shuffle_buffer,
    batch_size,
    num_iter,
    num_workers,
    pin_memory,
    to_tuple,
):
    dataset_urls = []
    for dataset_name in dataset_names:
        if dataset_name == "mcc":
            dataset_urls += MCC_dataset()
        elif dataset_name == "labeled":
            dataset_urls += supervised_dataset()
        elif dataset_name == 'hot_galaxy':
            dataset_urls += hot_galaxy_dataset()
        elif dataset_name == 'galaxyark':
            dataset_urls += GalaxyArk_dataset()
        elif dataset_name == 'new_galaxyark':
            dataset_urls += GalaxyArk_new_dataset()
        else:
            raise "Only support these datasets: [mcc, labeled, hot_galaxy, galaxyark, new_galaxyark]"

    dataset = webdataset.WebDataset(urls=dataset_urls, resampled=True)
    dataset = dataset.with_length(num_iter * num_workers)
    train_transformed = apply_webdataset_pipeline(
        wds_dataset=dataset,
        pipeline=get_train_pipeline(
            preprocessor, shuffle_buffer, batch_size, num_iter, to_tuple
        ),
    )
    train_dataloader = DataLoader(
        dataset=train_transformed,
        shuffle=False,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    return train_dataloader


def split_by_node(src):
    for s in src:
        yield s


def get_validation_dataset(preprocessor, num_workers, pin_memory):
    dataset_urls = validation_dataset()
    dataset = webdataset.WebDataset(urls=dataset_urls, nodesplitter=split_by_node)
    validate_transformed = apply_webdataset_pipeline(
        wds_dataset=dataset, pipeline=get_val_pipeline(preprocessor)
    )
    validate_dataloader = DataLoader(
        dataset=validate_transformed,
        shuffle=False,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    return validate_dataloader


def get_testing_dataset(preprocessor, num_workers, pin_memory):
    dataset_urls = testing_dataset()
    dataset = webdataset.WebDataset(urls=dataset_urls, nodesplitter=split_by_node)
    test_transformed = apply_webdataset_pipeline(
        wds_dataset=dataset, pipeline=get_val_pipeline(preprocessor)
    )
    test_dataloader = DataLoader(
        dataset=test_transformed,
        shuffle=False,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    return test_dataloader
