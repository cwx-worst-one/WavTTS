import ipdb
import tqdm
import yaml

from samantha.dataio.batching import SimpleBatcher, setup_batcher_fn
from samantha.dataio.bigmusic.bigmusic_compose import *
from samantha.dataio.parquet import ParquetDataset


def init_transform(params):
    cls_name = params["type"]
    kwargs = {k: v for k, v in params.items() if k != "type"}
    cls = globals()[cls_name]
    return cls(**kwargs)


def load_yaml(fp):
    with open(fp) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def compose_item_transform(parsed_yaml, stage="train", exclude=None):
    if exclude is None:
        exclude = []

    transforms = [
        init_transform(params)
        for params in parsed_yaml["data"][f"{stage}_item_transform"]
        if params["type"] not in exclude
    ]

    def compose(item):
        for transform in transforms:
            if isinstance(item, dict):
                item = transform(item)
                if isinstance(item, (tuple, list)):
                    assert transform.maybe_return_list_output
                if item is None:
                    return None
            else:  # list
                n_items = len(item)
                item = [transform(i) for i in item]
                item = [i for i in item if i is not None]
                print(f"filtered out {len(item)}/{n_items}")
                if not item:
                    return None
        return item

    return compose, parsed_yaml["data"][f"{stage}_dataset_ids"]


def compose_batch_transform(parsed_yaml, stage="train", exclude=None):
    if exclude is None:
        exclude = []

    transforms = [
        init_transform(params)
        for params in parsed_yaml["data"][f"{stage}_batch_transform"]
        if params["type"] not in exclude
    ]

    def compose(items):
        batch = {}
        for transform in transforms:
            transform(items, batch)
        return batch

    return compose


def _get_data_distribution(data):
    """Get distribution of data values by counting occurrences."""
    from collections import Counter

    import numpy as np

    try:
        # Handle different data types
        if isinstance(data, (list, tuple)):
            if len(data) == 0:
                return "empty"
            # For nested structures, flatten first level
            if isinstance(data[0], (list, tuple, np.ndarray)):
                flat_data = []
                for item in data:
                    if hasattr(item, "flatten"):
                        flat_data.extend(item.flatten().tolist())
                    else:
                        flat_data.extend(
                            item if isinstance(item, (list, tuple)) else [item]
                        )
                counter = Counter(flat_data)
            else:
                counter = Counter(data)
        elif hasattr(data, "flatten"):  # numpy arrays, tensors
            counter = Counter(data.flatten().tolist())
        elif hasattr(data, "shape"):  # tensors/arrays with shape info
            return f"shape={data.shape}, dtype={getattr(data, 'dtype', type(data).__name__)}"
        else:
            return f"type={type(data).__name__}, value={str(data)[:50]}..."

        # Limit output for readability
        if len(counter) > 10:
            most_common = counter.most_common(5)
            total_unique = len(counter)
            return f"top_5={dict(most_common)}, unique_values={total_unique}"
        else:
            return dict(counter)

    except Exception as e:
        return f"error_analyzing: {str(e)[:50]}"


def test_item(yaml_file, test_number=100, inspect_keys=["genre_primary_tag"]):

    parsed_yaml = load_yaml(yaml_file)  # Fixed variable name
    item_transform, data_ids = compose_item_transform(parsed_yaml, stage="train")
    batch_transform = compose_batch_transform(parsed_yaml, stage="train")

    test = parsed_yaml["data"]["train_batcher"]
    batcher = setup_batcher_fn(test)

    try:
        batcher = setup_batcher_fn(parsed_yaml["data"]["train_batcher"])
    except Exception as e:
        print(f"setup a simplebatcer {e}")
        batcher = SimpleBatcher()

    data_id = data_ids[0]  # <- pick one
    dataset = ParquetDataset(data_id=data_id)
    dataset_iter = iter(dataset)

    input_items = []
    output_items = []

    batch_count = 0

    # Process items until we have enough batches
    with tqdm.tqdm(total=test_number, desc="Collecting batches") as pbar:
        while batch_count < test_number:
            try:
                item_i = next(dataset_iter)
                item_o = item_transform(item_i)

                if item_o is None:
                    continue

                input_items.append(item_i)
                output_items.append(item_o)

                # Use streaming batcher
                batch = batcher.collate_batch(item_o)
                if batch is not None:
                    batch_processed = batch_transform(batch)
                    print(
                        f"Batch {batch_count} (size: {len(batch)}): {list(batch_processed.keys())}"
                    )

                    for key in inspect_keys:
                        print(f" {key}: {_get_data_distribution(batch_processed[key])}")

                    batch_count += 1
                    pbar.update(1)

            except StopIteration:
                print("Dataset exhausted before reaching target batch count")
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test item processing with YAML configuration"
    )
    parser.add_argument(
        "--input", required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        help="Number of test items to process (default: 100)",
    )

    args = parser.parse_args()

    test_item(yaml_file=args.input, test_number=args.num)
