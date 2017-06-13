# Requires: pybooru library
# Performs 'stage 3' input retrieval: image fetching and packing

import argparse
import csv

import requests
import tensorflow as tf
from PIL import Image
from io import BytesIO

import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('infile')
parser.add_argument('outdir')
parser.add_argument('basename')

args = parser.parse_args()

def get_out_file(args, fn_no):
    cur_no = fn_no
    fname = '{}-{:d}.tfrecords'.format(args.basename, cur_no)
    cur_path = os.path.join(args.outdir, fname)

    while os.path.exists(cur_path):
        cur_no += 1
        fname = '{}-{:d}.tfrecords'.format(args.basename, cur_no)
        cur_path = os.path.join(args.outdir, fname)

    return cur_path, cur_no

def skip(reader, n=1):
    for i in range(n*20):
        next(reader)

with open(args.infile, newline='') as infile:
    reader = csv.reader(infile)

    file_number = 0
    post_number = 0

    fname, file_number = get_out_file(args, file_number)

    skip(reader, file_number)

    writer = tf.python_io.TFRecordWriter(fname)

    sys.stdout.write('{}  |'.format(fname))
    sys.stdout.flush()

    for row in reader:
        post_id = int(row[0])
        img_url = row[1]

        if len(row) <= 2:
            sys.stderr.write('Rejecting image {} ({}): no valid tags\n'.format(post_id, img_url))
            sys.stderr.flush()
            continue

        tag_idxs = row[2:]

        # Unmangle certain urls:
        if img_url.startswith('http://danbooru.donmai.ushttp'):
            actual_url_idx = img_url.find('http://', 1)
            img_url = img_url[actual_url_idx:]

        tag_vector = [0] * 1000
        for tag_idx in tag_idxs:
            i = int(tag_idx)
            if i >= 1000:
                continue

            tag_vector[i] = 1

        tag_vector = bytes(tag_vector)

        tag_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tag_vector]))

        try:
            r = requests.get(img_url)
            r.raise_for_status()

            content_type = r.headers['content-type']
            if not content_type.startswith('image'):
                sys.stderr.write('Rejecting image {} ({}): invalid Content-Type {}\n'.format(post_id, img_url, content_type))
                sys.stderr.flush()
                continue

            filedata = BytesIO()
            img = Image.open(BytesIO(r.content))
            img.save(filedata, format='PNG')
        except:
            exc_type = sys.exc_info()[0]
            exc_val = sys.exc_info()[1]
            exc_trace = sys.exc_info()[2]

            sys.stderr.write('Exception while processing image: {} ({})\n{}, {} \n {}\n'.format(post_id, img_url, exc_type, exc_val, exc_trace))
            sys.stderr.flush()

            continue

        img_bytes = filedata.getvalue()
        img_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))

        example = tf.train.Example(features=tf.train.Features(feature={
            'tags': tag_feat,
            'image': img_feat
        }))

        writer.write(example.SerializeToString())

        post_number += 1
        if post_number % 20 == 0:
            sys.stdout.write('=|\n')

            new_fname, next_file_number = get_out_file(args, file_number+1)

            n_files_skipped = (next_file_number - file_number) - 1
            if n_files_skipped > 0:
                skip(reader, n_files_skipped)
            file_number = next_file_number

            sys.stdout.write('{}  |'.format(new_fname))

            writer.close()
            writer = tf.python_io.TFRecordWriter(new_fname)
        else:
            sys.stdout.write('=')
        sys.stdout.flush()

    sys.stdout.write('|')
    sys.stdout.flush()
    writer.close() # close the writer if necessary
