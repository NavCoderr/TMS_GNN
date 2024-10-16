#!/bin/bash

export PYTHONPATH=`pwd`
python3 ./tms/test_utils/test_agv_controller/test_agv_controller.py "$@"
