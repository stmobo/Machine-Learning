#!/bin/bash
apt-get update
apt-get install -y python3-pip nfs-common
pip3 install --force-reinstall --ignore-installed pip && pip3 install tensorflow scipy pillow
mkdir -p /mnt/disks/nst-disk
chmod a+w /mnt/disks/nst-disk
echo 'worker:/nst-disk /mnt/disks/nst-disk nfs auto,nfsvers=4,proto=tcp,noatime 0 0' >> /etc/fstab
