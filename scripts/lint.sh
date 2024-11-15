#!/bin/bash -e
pip3 install \
    isort==5.12.0 \
    black==24.2.0 \
    flake8==7.0.0 \
    yamllint==1.26.3

function print_failed_info() {
    echo "lint failed"
    exit 1
}

set -e
git ls-files samantha | grep -E "\.py$" | xargs isort --check-only
[ ${?} -eq 0 ] || print_failed_info
git ls-files samantha | grep -E "\.py$" | xargs black --check --diff --color
[ ${?} -eq 0 ] || print_failed_info
git ls-files samantha | grep -E "\.py$" | xargs flake8 --max-line-length 120 --max-doc-length 200 --ignore=E701,W503,E999,F405,F401,E712,F403,E203,F821,F811 --count --statistics
[ ${?} -eq 0 ] || print_failed_info
yamllint samantha recipes --no-warnings
[ ${?} -eq 0 ] || print_failed_info
echo "lint success"