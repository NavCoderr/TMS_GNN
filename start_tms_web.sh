#!/bin/bash

PYTHONPATH=`pwd`:PYTHONPATH
export PYTHONPATH
PATH=`pwd`:$PATH
export PATH

cd ./tms || exit
python -m flask --app tms_web run