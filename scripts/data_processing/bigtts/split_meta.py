import json
import os

metas = json.load(
    open(
        "/mnt/bn/jdy-lq-6/xmly/mos3.2_sim0.525_snr6_rms-13_asr0.85_internal_1/metas/xmly/wds2meta.json",  # noqa
        "r",
        encoding="utf8",
    )
)

meta = {}
idx = 0

for count, (k, v) in enumerate(metas.items()):
    if (count + 1) % 50 == 0:
        output_dir = f"split/part={idx:05d}/"
        os.makedirs(output_dir, exist_ok=True)
        json.dump(
            meta,
            open(f"{output_dir}/wds2meta.json", "w", encoding="utf8"),
            indent=2,
            ensure_ascii=True,
        )
        meta = {}
        idx += 1
    meta[k] = v

if meta:
    output_dir = f"split/part={idx:05d}/"
    os.makedirs(output_dir, exist_ok=True)
    json.dump(
        meta,
        open(f"{output_dir}/wds2meta.json", "w", encoding="utf8"),
        indent=2,
        ensure_ascii=True,
    )
