export S3A_VERSION=1.0.0.$1

if [ ! -f lab_audio.seed.s3a_$S3A_VERSION.tar.gz ]; then
    wget http://luban-source.byted.org/repository/scm/lab_audio.seed.s3a_$S3A_VERSION.tar.gz
fi
sudo pip3 uninstall s3a -y \
    && rm -fr tmp.s3a \
    && mkdir tmp.s3a \
    && tar -xvf lab_audio.seed.s3a_$S3A_VERSION.tar.gz -C tmp.s3a \
    && sudo pip3 install --no-cache-dir tmp.s3a/s3a-$S3A_VERSION-cp39-cp39-linux_x86_64.whl \
    && rm -fr tmp.s3a lab_audio.seed.s3a_$S3A_VERSION.tar.gz
