#!/bin/bash -ex
sudo mkdir /home/code
sudo chown -R tiger /home/code

wget "https://security.feishu.cn/link/safety?target=https%3A%2F%2Fleafboat.byted.org%2Fstatic%2Flauncher%3Fos%3Dlinux%26arch%3Damd64&scene=ccm&logParams=%7B%22location%22%3A%22ccm_default%22%7D&lang=en-US" -O launcher

chmod +x launcher

./launcher \
    -token eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MjUxLCJyb2xlIjoiYWdlbnQiLCJzY29wZXMiOm51bGwsImlzcyI6ImxlYWZib2F0Iiwic3ViIjoic2YtaFhLZVNuVVUwVyJ9.HGoejJdQxgDi2sfi-t9McMJ_Y4riRWdNjlY4gYinKKVCnJ4OceNlyn1qcfR14LxaeVNwZd7b2EgATOnNOJ6EzQ \
    -engine shell \
    -concurrency 1