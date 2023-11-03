import mir_eval
import numpy as np
from mir_tools.key_detection.constants import KEY_MAJMIN_NAME
from scipy import stats


def post_process(keysig_probs):
    keysig = np.argmax(np.sum(keysig_probs, 0))
    keysig_pred_labels = np.argmax(keysig_probs, 1)
    output = {}
    output["keymode_name"] = KEY_MAJMIN_NAME[keysig]
    output["keymode_id"] = keysig
    output["keymode_pred"] = np.array([k for k in keysig_pred_labels.astype(int)])
    return output


def get_key_score(processed_preds, key_map):
    out = {}
    keymode_label = []

    # breakpoint()
    post_processed_preds = post_process(processed_preds["keysig_pred_probs"])
    for frame in processed_preds["keysig_truth_labels"]:
        if frame.sum() == 0:
            keymode_label.append(0)
        else:
            keymode_label.append(np.argmax(frame))
    keymode_label = np.array(keymode_label)
    keymode_song_label = stats.mode(keymode_label)[0]
    out["KMACC"] = mir_eval.key.weighted_score(
        KEY_MAJMIN_NAME[post_processed_preds["keymode_id"]]
        .replace(":", " ")
        .replace("Maj", "major")
        .replace("Min", "minor"),
        KEY_MAJMIN_NAME[keymode_song_label]
        .replace(":", " ")
        .replace("Maj", "major")
        .replace("Min", "minor"),
    )

    # out["KMACC"] = float(
    #    post_processed_preds["keymode_id"] in key_map[keymode_song_label]
    # )

    length = len(keymode_label[keymode_label != 0])
    if length == 0:
        length = 1
    out["KMACC0"] = (
        sum(
            [
                x in key_map[y]
                for x, y in zip(
                    post_processed_preds["keymode_pred"][keymode_label != 0],
                    keymode_label[keymode_label != 0],
                )
            ]
        )
        / length
    )

    return out, keymode_song_label, keymode_label
