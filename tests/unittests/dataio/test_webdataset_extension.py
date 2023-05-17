from samantha.dataio.webdataset.extension import IndexedWebDataset


def _load_url2index(url2index_path):
    url2index = {}
    with open(url2index_path, "r") as f:
        for line in f:
            ary = line.strip().split("\t")
            url2index[ary[0]] = ary[1]
    return url2index


def _verify_items(items):
    assert len(items) == 2
    assert items[0]["__key__"] == "compressed/0001"
    assert items[1]["__key__"] == "compressed/0003"
    assert items[1]["__index_data__"]["metadata"] == 1


def test_indexed_webdataset():
    # Load from file
    url2index_path = "tests/data/webdataset/compressed.url2index"
    dataset1 = IndexedWebDataset(url2index=url2index_path).decode()
    _verify_items([item for item in dataset1])
    # Load from in-memory dict
    url2index = _load_url2index(url2index_path)
    dataset2 = IndexedWebDataset(url2index=url2index).decode()
    _verify_items([item for item in dataset2])
    