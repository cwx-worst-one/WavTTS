#exp=PrefixLDM4_300M_48A100_token30000_setting2_40hzMel_UMMv02
#ckpt="epoch=00-step=320000-loss=0.13.ckpt"
#step=320000

exp=PrefixLDM4_300M_48A100_setting2_40hzWVAE_UMMv02_textDrop0.25_Norm2
ckpt="epoch=00-step=150000-loss=0.52.ckpt"
step=150000

out_name=${exp}_${step}

final_outdir=/mnt/bn/jdy-lq-2/bigtts-nar/demo_output/${out_name}
mkdir -p $final_outdir

out_dir=/mnt/bn/jdy-lq-2/bigtts-nar/output

lang=en
src_outdir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/ddim_10steps_TextCFG4
demo_outdir=${final_outdir}/${lang}
mkdir -p $demo_outdir
ls $src_outdir/*.wav | head -n100 | xargs -i cp {} $demo_outdir

lang=zh
src_outdir=$out_dir/$exp/icl_testset_2.0_${lang}/$step/ddim_10steps_TextCFG4
demo_outdir=${final_outdir}/${lang}
mkdir -p $demo_outdir
ls $src_outdir/*.wav | head -n100 | xargs -i cp {} $demo_outdir

cd $final_outdir
cd ..
tar -cf ${out_name}.tar ${out_name}

