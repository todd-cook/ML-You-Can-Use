#!/usr/bin/env bash
# runUnitTests.sh
# export COVERAGE_DEBUG=process,config
black mlyoucanuse
python -m  pytest --doctest-modules --cov-report=term --mypy --cov-config=.coveragerc --pylint \
mlyoucanuse/ --cov=mlyoucanuse/
find . -name '.coverage' -type f -delete
find . -name '.coverage.*' -type f -delete
