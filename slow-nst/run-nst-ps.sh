#!/bin/bash
if [ -z "$STY" ]; then exec screen -dm -S nst /bin/bash "$0"; fi
python3 /mnt/disks/nst-disk/neural-style-transfer.py --vgg_19_checkpoint /mnt/disks/nst-disk/vgg_19.ckpt ignored ignored ignored ps-1:2222,ps-2:2222 worker:2222 ps 1
