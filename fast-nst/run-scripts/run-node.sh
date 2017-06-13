#!/bin/bash
if [ -z "$STY" ]; then exec screen -dm -S nst /bin/bash "$0"; fi
python3 /mnt/fnst/fast-nst/fast-style-transfer.py -c /mnt/fnst/run-conf/common.conf |& tee /mnt/fnst/console-log/$(uname -n).log;
exit 0;
