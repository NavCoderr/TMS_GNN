#!/bin/bash

export PYTHONPATH=`pwd`
python3 ./tms/tms_cli.py "$@" tms/test_utils/testGraph.json tms/test_utils/mesTasksMapping.json
