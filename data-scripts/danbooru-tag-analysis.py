# Requires: pybooru library
# Performs 'stage 2' input processing: tag counting

import argparse
import csv

from pybooru import Danbooru

import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('infile')
parser.add_argument('tagfile')
parser.add_argument('charfile')
parser.add_argument('mapfile')

args = parser.parse_args()

# Row format: ID, file url, large file url, preview file URL, general tags (separated) -->
tag_occurences = {}
character_occurences = {}
with open(args.infile, newline='') as infile:
    reader = csv.reader(infile)
    for row in reader:
        tags = row[4:-1]
        character = row[-1]

        character_occurences[character] = character_occurences.get(character, 0) + 1
        for tag in tags:
            tag_occurences[tag] = tag_occurences.get(tag, 0) + 1

tag_counts = list(tag_occurences.items())
character_counts = list(character_occurences.items())

tag_counts.sort(key=lambda i: i[1], reverse=True)
character_counts.sort(key=lambda i: i[1], reverse=True)

# Now map tags to indices

excluded_tags = [
    '1girl',
    'solo',
    'md5_mismatch',
    'translated',
    '',
    ' ',
    'game_cg',
    'highres',
    'lowres',
    'absurdres',
    'web_address',
    'official_art',
    'spoilers',
    'signature',
    'comic',
    'jpeg_artifacts',
    'watermark',

]

def is_excluded_tag(tag):
    return tag.endswith('_id') or tag.endswith('_request') or tag.endswith('_username') or tag.endswith('_name') or tag.endswith('_filesize') or tag.startswith('animated') or (tag in excluded_tags)

# Row format: ID, file url, large file url, preview file URL, general tags (separated) -->
map_items = []

for tag in tag_counts:
    if len(map_items) < 1000:
        if not is_excluded_tag(tag[0]):
            map_items.append( (tag[0], len(map_items)) )
    else:
        break

with open(args.mapfile, mode='w', newline='') as mapfile:
    writer = csv.writer(mapfile)
    for map_pair in map_items:
        writer.writerow(map_pair)

with open(args.tagfile, mode='w', newline='') as tagfile:
    writer = csv.writer(tagfile)
    for tag_pair in tag_counts:
        writer.writerow(tag_pair)

with open(args.charfile, mode='w', newline='') as charfile:
    writer = csv.writer(charfile)
    for character_pair in character_counts:
        writer.writerow(character_pair)
