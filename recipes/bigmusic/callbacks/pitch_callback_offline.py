import argparse
from recipes.bigmusic.callbacks.mir_metrics import MIRPitchMetricsCallback

def mir_pitch_transition_offline(output_dir: str) -> None:

    callback = MIRPitchMetricsCallback()
    callback.on_predict_end(None, None, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/mlx_devbox/users/wenyi.hsiao/playground/testcases")
    args = parser.parse_args()
    mir_pitch_transition_offline(args.output_dir)
