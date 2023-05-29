import collections
import json

import pandas as pd
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "The input file to be converted.")

flags.DEFINE_string("output_prefix", None, "The prefix of output file.")


def batching(iterable, n=500000):
    buffer = []
    for e in iterable:
        if len(buffer) >= n:
            yield buffer
            del buffer
            buffer = []
        buffer.append(e)

    if len(buffer) > 0:
        yield buffer


def main(_):
    with open(FLAGS.input_file) as f:
        for idx, batch in enumerate(batching(f)):
            sessions = collections.defaultdict(list)
            print(len(batch))
            for line in batch:
                pairs = json.loads(line)
                if not isinstance(pairs, list):
                    pairs = [pairs]
                pairs = [
                    {"prompt": pair["prompt"], "response": pair["response"]}
                    for pair in pairs
                ]
                sessions["session"].append(pairs)
            df = pd.DataFrame(sessions)
            df.to_parquet(FLAGS.output_prefix + "_" + str(idx) + ".parquet")


if __name__ == "__main__":
    app.run(main)
