import argparse
from recipes.bigmusic.callbacks.mir_metrics import MIRSectionTransitionCallback
from recipes.bigmusic.callbacks.common_callbacks import ForceAlignCallback, UploadToEasyCycleCallback, ASRCallback

def mir_section_transition_offline(output_dir: str) -> None:

    callback = UploadToEasyCycleCallback()
    callback.on_predict_end(None, None, output_dir)

    callback = ASRCallback()
    callback.on_predict_end(None, None, output_dir)

    callback = ForceAlignCallback()
    callback.on_predict_end(None, None, output_dir)

    callback = MIRSectionTransitionCallback()
    callback.on_predict_end(None, None, output_dir=output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    mir_section_transition_offline(args.output_dir)
