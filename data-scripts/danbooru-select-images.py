# Requires: pybooru library
# Performs 'stage 1' input retrieval: url retrieval and tag fetching

import argparse
import csv

from pybooru import Danbooru

import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('outfile')
parser.add_argument('num_posts', type=int)
parser.add_argument('tags', nargs=2)

args = parser.parse_args()

client = Danbooru('danbooru')

search_tag_str = args.tags[0] + ' ' + args.tags[1]
danbooru_url = 'http://danbooru.donmai.us'

n_rows_written = 0

# Row format: ID, file url, large file url, preview file URL, general tags (separated) -->
with open(args.outfile, mode='a', newline='') as outfile:
    writer = csv.writer(outfile)
    while n_rows_written < args.num_posts:
        posts = client.post_list(tags=search_tag_str, random=True)
        diagstr = ''
        for post in posts:
            tags = post['tag_string_general'].split(' ')

            main_url = ''
            large_url = ''
            preview_url = ''

            if ('large_file_url' in post) and post['has_large']:
                large_url = danbooru_url+post['large_file_url']
            else:
                pass
                #print("[Warning] Post {} missing large file URL.".format(post['id']))

            if 'file_url' in post:
                main_url = danbooru_url+post['file_url']
            else:
                print("[Warning] Post {} missing main file URL.".format(post['id']))

            if 'preview_file_url' in post:
                preview_url = danbooru_url+post['preview_file_url']
            else:
                pass
                #print("[Warning] Post {} missing preview file URL.".format(post['id']))

            row = [
                post['id'],
                main_url,
                large_url,
                preview_url,
            ]

            diagstr += post['tag_string_character'] + ' '

            row.extend(tags)
            row.append(post['tag_string_character'])

            writer.writerow(row)
            n_rows_written += 1
        print("Characters: {}".format(diagstr))
        print("Retrieved {} rows of post metadata so far...".format(n_rows_written))
