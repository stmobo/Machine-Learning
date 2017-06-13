#!/bin/bash
nohup tensorboard --logdir=/mnt/fnst/log --port 80 > `uname -n`-tensorboard.log &
