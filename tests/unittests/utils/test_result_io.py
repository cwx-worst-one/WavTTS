import json
import os

from samantha.utils.result_io import prep_training_output


def test_prep_training_output():
    this_file_dir_path = os.path.dirname(__file__)
    print(this_file_dir_path)

    export_dir = os.path.join(
        this_file_dir_path,
        "../../recipes/sample_project/test_sample_project/logs/sample_project/0.1/",
    )

    os.environ["ARNOLD_OUTPUT"] = "./outputs"

    metadata_json_golden_loc = os.path.join(export_dir, "train_meta.json")
    with open(metadata_json_golden_loc) as fp:
        golden = json.load(fp)

    result = prep_training_output(export_dir)

    golden_ckpts = golden["train"]["checkpoints"]
    assert len(golden_ckpts) == len(result)
    for i in range(len(golden_ckpts)):
        cur_golden_ckpt = golden_ckpts[i]
        collect_metas = result[i].metas
        assert cur_golden_ckpt["acc/val"] == float(collect_metas["acc/val"])

    golden_events = golden["train"]["tfevents"]
    collect_tfevents = result[-1].metas["tfevents"].split(",")
    assert len(golden_events) == len(collect_tfevents)
    for i in range(len(golden_events)):
        cur_gold_tfevent = golden_events[i]
        collect_tfevent = collect_tfevents[i]
        assert os.path.basename(cur_gold_tfevent) == os.path.basename(collect_tfevent)
