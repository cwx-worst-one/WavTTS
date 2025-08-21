#!/bin/bash

byted_wandb=$(ls -d /usr/local/lib/python3.12/site-packages/byted_wandb-*);
wandb=$(echo ${byted_wandb} | sed 's=byted_wandb=wandb=');
cp -r $byted_wandb $wandb;
sed -i 's=Name: byted-wandb=Name: wandb=;/Requires-Dist:/d' ${wandb}/METADATA;

sed -i 's|from pathtools.patterns import match_any_paths|def match_any_paths(*args, **kwargs): return False|g' \
    /usr/local/lib/python3.12/site-packages/wandb/vendor/watchdog_0_9_0/wandb_watchdog/events.py
sed -i 's|from pathtools.path import absolute_path|def absolute_path(path): return path|g' \
    /usr/local/lib/python3.12/site-packages/wandb/vendor/watchdog_0_9_0/wandb_watchdog/observers/kqueue.py
