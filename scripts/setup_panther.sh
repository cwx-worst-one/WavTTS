export MARIANA_PANTHER_VERSION=1.7.9.72
wget http://luban-source.byted.org/repository/scm/lab.speech.panther_arnold_${MARIANA_PANTHER_VERSION}.tar.gz
tar -zxf lab.speech.panther_arnold_${MARIANA_PANTHER_VERSION}.tar.gz
sudo pip3 uninstall -y panther-gpu
pip3 uninstall -y panther-gpu
pip3 install *torch*/panther_gpu-*.whl