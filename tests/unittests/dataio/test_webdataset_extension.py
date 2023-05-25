from samantha.dataio.webdataset.extension import IndexedWebDataset


def _load_url2index(url2index_path, global_datadir):
    url2index = {}
    with open(url2index_path, "r") as f:
        for line in f:
            line = line.strip() % (global_datadir, global_datadir)
            line = line.split("\t")
            url2index[line[0]] = line[1]
    return url2index


def _verify_items(items):
    assert len(items) == 2
    assert items[0]["__key__"] == "compressed/0001"
    assert items[1]["__key__"] == "compressed/0003"
    assert items[1]["__index_data__"]["metadata"] == 1


def test_indexed_webdataset(global_datadir):
    url2index_path = "tests/data/webdataset/compressed.url2index"
    # Load from in-memory dict
    url2index = _load_url2index(url2index_path, global_datadir)
    dataset = IndexedWebDataset(url2index=url2index).decode()
    _verify_items([item for item in dataset])
    