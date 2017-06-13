# Requires: pybooru library
# Performs 'stage 3' input retrieval: image fetching and packing

import argparse
import csv

from pybooru import Danbooru

import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('mapfile')
parser.add_argument('postfile')
parser.add_argument('outfile')

args = parser.parse_args()

tag_mapping = {}
with open(args.mapfile, newline='') as mapfile:
    reader = csv.reader(mapfile)
    tag_mapping = dict(reader)

n_posts_processed = 0
with open(args.postfile, newline='') as postfile, open(args.outfile, mode='w', newline='') as outfile:
    reader = csv.reader(postfile)
    writer = csv.writer(outfile)

    client = Danbooru('danbooru')
    for row in reader:
        post_id = int(row[0])

        # Get an image url:
        normal_url = row[1]
        large_url = row[2]
        preview_url = row[3]

        preferred_url = normal_url
        if preferred_url.lstrip() == '':
            preferred_url = large_url

        if preferred_url.lstrip() == '':
            post_details = client.post_show(post_id)
            if 'source' in post_details:
                print("Got source for post {}".format(post_id))
                preferred_url = 'http://danbooru.donmai.us'+post_details['source']
            else:
                continue # skip this image

        # Convert tags to tag indexes:
        tag_idxs = []
        tags = row[4:-1]
        for tag in tags:
            if tag in tag_mapping:
                tag_idxs.append(tag_mapping[tag])

        outrow = [post_id, preferred_url]
        outrow.extend(tag_idxs)
        writer.writerow(outrow)

        if n_posts_processed % 20 == 0:
            print("Processed {} posts...".format(n_posts_processed))

        n_posts_processed += 1
