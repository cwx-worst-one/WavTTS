#!/bin/bash

byted_wandb=$(ls -d /usr/local/lib/python3.11/site-packages/byted_wandb-*);
wandb=$(echo ${byted_wandb} | sed 's=byted_wandb=wandb=');
cp -r $byted_wandb $wandb;
sed -i 's=Name: byted-wandb=Name: wandb=;/Requires-Dist:/d' ${wandb}/METADATA;
