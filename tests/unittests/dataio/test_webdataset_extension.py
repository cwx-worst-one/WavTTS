from samantha.dataio.webdataset.extension import IndexedWebDataset


def test_indexed_webdataset():
    tar_path = "pipe:cat tests/data/webdataset/compressed.tar"
    index_path = "tests/data/webdataset/compressed.index"
    dataset = IndexedWebDataset(urls=[tar_path], index_files=[index_path]).decode()
    items = [item for item in dataset]
    assert len(items) == 2
    assert items[0]["__key__"] == "compressed/0001"
    assert items[1]["__key__"] == "compressed/0003"
    assert items[1]["__index_data__"]["metadata"] == 1
