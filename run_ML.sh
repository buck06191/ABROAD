#!/bin/bash
# A simple script to run the ML python code and write output to a date-named DIR
DATE=`date +%d-%m-%YT%H:%M:%S`
mkdir logs/$DATE
python ML.py | tee logs/$DATE/stdout
