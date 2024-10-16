#!/bin/bash

export PYTHONPATH=`pwd`
python3 ./tms/test_utils/test_agv/test_agv.py "$@"
