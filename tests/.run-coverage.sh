#!/bin/bash
git ls-files tests | grep -e "\.py$" | xargs python3 -m pytest -m "not disable" --cov-report=xml:coverage.xml --cov=samantha --junit-xml=report.xml
