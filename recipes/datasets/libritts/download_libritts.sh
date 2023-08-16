#!/bin/bash
sudo apt-get install axel -y
axel https://us.openslr.org/resources/60/train-clean-360.tar.gz
tar -xf train-clean-360.tar.gz


axel https://us.openslr.org/resources/60/test-clean.tar.gz
tar -xf test-clean.tar.gz