#!/bin/bash

function handler() {
    kill -s SIGTERM $PID
}
trap handler SIGTERM

addr=$(host $(hostname) | awk '{print $NF}')
python3 lclstream/psana_pull.py -l tcp://$addr:2020 &
PID=$!
echo "Started puller pid $PID"

ssh psana /sdf/home/r/rogersdd/venvs/run_psana_push 8 \
    -e xpptut15 -r 580 -d jungfrau4M -m image \
    -a tcp://$addr:2020 -c smd \
    --img_per_file 20

kill $PID
