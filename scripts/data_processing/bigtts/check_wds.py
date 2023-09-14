import logging

from samantha.dataio.utils import parse_data_urls
from samantha.dataio.webdataset.ra_wds import WebDataset


def test_new_dataset(base):

    logger = logging.getLogger(__name__)
    path = [
        f"hdfs://harunava/home/byte_data_seed_us/hdd_va/speech/data/datastore/bigtts/resso_podcast_5-10s_v2/package/wvae_1_web_dataset_1/data/part={i:05d}/*.tar"  # noqa
        for i in range(base, base + 5)
    ]
    urls = parse_data_urls(data_urls=path)
    dataset = WebDataset(urls=urls).decode()
    url = None
    for idx, item in enumerate(dataset):
        if item["__url__"] != url:
            url = item["__url__"]
            logger.info(f"{base=} iterated {idx} samples, cur_url: {url}")


def main():
    from multiprocessing import Process

    procs = []
    for base in range(0, 100, 5):
        p = Process(target=test_new_dataset, args=(base,))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()


main()
