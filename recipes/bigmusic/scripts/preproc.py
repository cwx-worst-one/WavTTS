import os
import sys

from recipes.bigmusic.utils.preproc_utils import (
    symbolic_preproc,
    symbolic_preproc_mp,
)
from recipes.bigmusic.datasets.symbolic_music.base_codec import IndexerConfig
from recipes.bigmusic.datasets.symbolic_music.trans5stems_codec import Trans5StemsCodec


def pre_20240319_bm():
    # from symbolic_gen.dataloader.msd_zip_loader import get_mir_dfs_dict_iter
    # from symbolic_gen.dataloader.remi_preproc_mp import preproc_mp_iter_input
    parquet_dataset_ids = [1837, 1881, 1885]
    data_id = sys.argv[2]

    ## Default value, singletrack
    skip_ratio = 0.0
    valid_perc = 0.01
    seed = 42
    preproc_func = symbolic_preproc
    output_dir_root = os.path.join(
        os.environ["AI_MUSIC_DIR"],
        f"../../dump/ai_music/20240312.symbolic.dump/data"
    )

    if data_id == "20240325.licensed.part":
        note_subsets = ["vocal", "guitar", "piano", "bass", "drums"]

        valid_perc = 0.05

        dfs_dict_iter = Trans5StemsCodec.get_dfs_dict_iter(
            # num_samples=3,
            data_ids=parquet_dataset_ids,
            skip_ratio=skip_ratio,
            note_subsets=note_subsets,
            seed=seed,
        )

        config = IndexerConfig(
            include_phoneme=False,
            include_utterance_phoneme_tokens=False,
            include_bar_event = True,
            include_stem_indicator = True,
            include_drum_events = True,
            include_section_indicator_each_bar = True,
            include_bpm_levels = True,
            include_prompt = True,
        )
        codec = Trans5StemsCodec(config)
        output_dir = os.path.join(output_dir_root, data_id)
        os.makedirs(output_dir, exist_ok=True)
        preproc_func = symbolic_preproc_mp
    else:
        raise ValueError(f"{data_id} not implemented")

    preproc_func(
        output_dir,
        dfs_dict_iter,
        codec,
        valid_perc=valid_perc,
    )

    from IPython import embed; embed(using=False); os._exit(0)  # fmt: skip


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        locals()[sys.argv[1]]()
    else:
        [print(k) for k, v in locals().items() if callable(v)]
