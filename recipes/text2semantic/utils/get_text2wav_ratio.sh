# set -x
meta_file=$1
utt2ratio=$2

rm -f $utt2ratio

meta_dir=`dirname $utt2ratio`
utt2ratio_name=`basename $utt2ratio`

timestamp=$(date +%s)
thread_dir=/tmp/thread_metas_$timestamp
sudo mkdir $thread_dir
num_job=64
num=`wc -l $meta_file | awk -F' ' '{print $1}'`
num_per_thread=`expr $num / $num_job + 1`
sudo split -l $num_per_thread --additional-suffix=.lst -d $meta_file $thread_dir/thread-

cd /mnt/bn/huangzhiying-nas-speech2speech-volume1/code/samantha/recipes/valle/utils/
for x in {00..63};do
    python3 get_text2wav_ratio.py $thread_dir/thread-$x.lst ${utt2ratio}.$x &
done
wait
cd -

for x in {00..63};do
    cat ${utt2ratio}.$x >> $utt2ratio
done

rm -f ${utt2ratio}.*