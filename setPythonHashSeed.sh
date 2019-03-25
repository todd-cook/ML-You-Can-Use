#!/usr/bin/env bash
# echo "Please source this script into your environment to set the Python library hash seed to zero, for reproducibility across runs."
# echo "Same string hashed in separate python calls without the seed."
# python -c "print (hash('the default hash function is not cryptographically secure'))"
# python -c "print (hash('the default hash function is not cryptographically secure'))"
PYTHONHASHSEED=0
export PYTHONHASHSEED
# echo "Same string hashed in separate python calls with the seed set."
# python -c "print (hash('the default hash function is not cryptographically secure'))"
# python -c "print (hash('the default hash function is not cryptographically secure'))"
