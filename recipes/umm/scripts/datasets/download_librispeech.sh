#!/bin/bash
sudo apt-get install axel -y
axel https://us.openslr.org/resources/12/train-clean-360.tar.gz
tar -xvf train-clean-360.tar.gz