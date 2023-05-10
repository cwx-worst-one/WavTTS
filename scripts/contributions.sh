#!/bin/bash
# since date, such as 2021-11-01
SINCE=$1

git log --since=${SINCE} --format='%aN' | sort -u | while read name; do echo -en "$name\t"; git log --author="$name" --since=${SINCE} --pretty=tformat: --numstat | awk '{ add += $1; subs += $2; loc += $1 + $2 } END { printf "added %s\tremoved %s\ttotal %s\n", add, subs, loc  }' -; done | sort -nk7 -r
