#!/bin/bash
if [ -z "$STY"]; then exec screen -dm -S nst-worker /bin/bash "$0"; fi
tensorboard --logdir /mnt/disks/nst-disk/logs --port 80 &
python3 /mnt/disks/nst-disk/neural-style-transfer.py --logdir /mnt/disks/nst-disk/logs --interim_file_template /mnt/disks/nst-disk/intermediate-imgs/run1_ --vgg_19_checkpoint /mnt/disks/nst-disk/vgg_19.ckpt /mnt/disks/nst-disk/input/content.jpg /mnt/disks/nst-disk/input/style.jpg /mnt/disks/nst-disk/output/run1.jpg ps-1:2222,ps-2:2222 worker:2222 worker 0
