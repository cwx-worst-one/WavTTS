import os
from functools import partial

from webdataset import WebDataset

from samantha.utils.webdataset import apply_webdataset_pipeline


def test_apply_webdataset_pipeline(global_datadir):
    wds_dataset = WebDataset(os.path.join(global_datadir, "webdataset/compressed.tar"))
    pipeline = [partial(WebDataset.decode, "rgb")]
    out_wds_dataset = apply_webdataset_pipeline(wds_dataset, pipeline)
    assert next(iter(out_wds_dataset))["txt.gz"] == "hello\n"
