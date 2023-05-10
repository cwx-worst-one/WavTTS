#! /bin/sh

exec 1>&2

EMPTY_TREE=$(git hash-object -t tree /dev/null)
# or: EMPTY_TREE=4b825dc642cb6eb9a060e54bf8d69288fbee4904
if git diff --cached --numstat $EMPTY_TREE | grep -e '^-' >/dev/null; then
    echo Error: commit would add binary files:
    git diff --cached --numstat $EMPTY_TREE | grep -e '^-' | cut -f3-
    exit 1
fi
