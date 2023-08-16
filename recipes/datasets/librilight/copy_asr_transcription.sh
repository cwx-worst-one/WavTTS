OUT_DIR=txt_files
for d in /mnt/bn/nas-jeff-02/data/valle/mos3.8_sim0.0_snr7_rms-13_asr0.85/check/librilight_230711/*; do
    txt_path=$d/text.txt
    dirname=$(basename $(dirname $txt_path))
    out_fp=${dirname}_text.txt
    echo $txt_path $out_fp
    cp $txt_path $OUT_DIR/$out_fp
done;