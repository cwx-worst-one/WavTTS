S3A_VERSION=1.0.0.$1

current_s3a_version=$(pip show s3a | grep "Version" | grep -oE "[^: ]+$")
if [ "${current_s3a_version}" != "${S3A_VERSION}" ]; then
    echo "[WARNING] current_s3a_version '${current_s3a_version}'!='${S3A_VERSION}', will download and install"
    sudo pip3 uninstall s3a -y;
    if [ ! -f lab_audio.seed.s3a_$S3A_VERSION.tar.gz ]; then
        wget http://luban-source.byted.org/repository/scm/lab_audio.seed.s3a_$S3A_VERSION.tar.gz;
    else
        echo "[WARNING] 'lab_audio.seed.s3a_$S3A_VERSION.tar.gz' is already exists, skip donwload"
    fi
    rm -rf tmp.s3a;
    mkdir tmp.s3a;
    tar -xvf lab_audio.seed.s3a_$S3A_VERSION.tar.gz -C tmp.s3a;
    sudo pip3 install --no-cache-dir tmp.s3a/s3a-$S3A_VERSION-cp39-cp39-linux_x86_64.whl;
    rm -fr tmp.s3a lab_audio.seed.s3a_$S3A_VERSION.tar.gz;
else
     echo "[INFO] current_s3a_version '${current_s3a_version}' has already installed"
fi
