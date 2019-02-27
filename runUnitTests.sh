#!/usr/bin/env bash
# runUnitTests.sh
# export COVERAGE_DEBUG=process,config
python -m  pytest --doctest-modules --cov-report=term --mypy --cov-config=.coveragerc --pylint \
mlyoucanuse/ --cov=mlyoucanuse/
find . -name '.coverage' -type f -delete
find . -name '.coverage.*' -type f -delete
codecov --token=6e8e2361-345c-41c9-9401-a6911bdb8a40
