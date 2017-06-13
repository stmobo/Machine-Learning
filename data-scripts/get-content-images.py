import argparse
import csv

import aiohttp
import asyncio
import async_timeout

from PIL import Image

import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('image_csv', help='Path to OpenImages images.csv file.')
parser.add_argument('output_dir', default='input/content/training', help='Base path to put downloaded images in.')
parser.add_argument('--num-images', type=int, default=1000, help='Number of images to download.')
parser.add_argument('--final-img-count', type=int, default=100000, help='Final number of images to have total, across runs of this script.')
parser.add_argument('--num-concurrent', type=int, default=50, help='Number of concurrent requests to make at once')
parser.add_argument('--img-format', help='File type to save output images as')

async def get_image(args, img_id, img_url, session):
    print("Requesting: Image {} from {}".format(img_id, img_url))
    try:
        async with session.get(img_url, timeout=15) as resp:
            if resp.status >= 400: # not ok status
                return img_id, -1, 'Got status {}: {}'.format(resp.status, resp.reason), None

            img_fmt = ''
            if resp.content_type == 'image/png':
                img_fmt = 'PNG'
            elif resp.content_type == 'image/jpeg' or resp.content_type == 'image/jpg':
                img_fmt = 'JPEG'
            else:
                return img_id, -1, 'Invalid MIME type: ' + resp.content_type, None

            if args.img_format is None:
                if img_fmt == 'PNG':
                    out_path = os.path.join(args.output_dir, img_id+'.png')
                elif img_fmt == 'JPEG':
                    out_path = os.path.join(args.output_dir, img_id+'.jpg')
            else:
                out_path = os.path.join(args.output_dir, img_id+'.'+args.img_format.lower())

            if args.img_format is None:
                n_bytes = 0
                with open(out_path, 'wb') as fd:
                    while True:
                        f_chunk = await resp.content.readany()
                        n_bytes += len(f_chunk)
                        if not f_chunk:
                            break
                        fd.write(f_chunk)
            else:
                buf = bytearray()
                while True:
                    f_chunk = await resp.content.readany()
                    if not f_chunk:
                        break
                    buf.extend(f_chunk)
                with Image.open(buf) as im:
                    im.save(out_path)

            return img_id, n_bytes, 'Success', out_path
    except aiohttp.ClientError as exc:
        return img_id, -1, 'HTTP client exception: {}'.format(str(exc)), None
    except:
        return img_id, -1, '{} exception: {}'.format(sys.exc_info()[0], sys.exc_info()[1]), None

def get_url_batches(reader):
    n_images_downloaded = 0
    while n_images_downloaded < args.final_img_count:
        current_batch = []
        while len(current_batch) < args.num_concurrent and n_images_downloaded < args.final_img_count:
            cur_row = next(reader)
            img_id = cur_row[0]

            n_images_downloaded += 1

            # don't refetch
            if os.path.exists(os.path.join(args.output_dir, img_id+'.png')) or os.path.exists(os.path.join(args.output_dir, img_id+'.jpg')):
                print("Skipping {} -- already fetched".format(img_id))
                continue

            current_batch.append(cur_row)
        yield current_batch

async def get_images(args, loop):
    async with aiohttp.ClientSession(loop=loop) as sess:
        n_images_retrieved = 0
        n_requests_made = 0

        def image_req_done(ft):
            nonlocal n_images_retrieved
            img_id, n_bytes, stat, out_path = ft.result()
            if stat == 'Success':
                print("Retrieved image {} ({} bytes) --> {}".format(img_id, n_bytes, out_path))
                n_images_retrieved += 1
            else:
                print("Error retrieving image {}: {}".format(img_id, stat))

        with open(args.image_csv, 'r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            csvheader = next(csvreader)

            print('CSV header: ' + str(csvheader))

            for row_batch in get_url_batches(csvreader):
                print("Requesting images {} - {} [{:%} complete]".format(n_images_retrieved, n_images_retrieved+args.num_concurrent, n_images_retrieved/args.num_images))

                current_requests = []
                for csv_row in row_batch:
                    if n_images_retrieved+len(current_requests) >= args.num_images:
                        break

                    img_url = csv_row[3]      # Default to OriginalURL
                    if len(csv_row) >= 11 and len(csv_row[10]) > 0:
                        img_url = csv_row[10] # Get 300k thumbnail if it's there

                    ft = asyncio.ensure_future(get_image(args, csv_row[0], img_url, sess))
                    ft.add_done_callback(image_req_done)
                    current_requests.append(ft)
                    n_requests_made += 1

                await asyncio.gather(*current_requests)

                if n_images_retrieved > args.num_images:
                    break
    return n_requests_made, n_images_retrieved

args = parser.parse_args()
loop = asyncio.get_event_loop()
n_requests_made, n_images_retrieved = loop.run_until_complete(get_images(args, asyncio.get_event_loop()))

print("Image download complete: got {} images with {} requests".format(n_images_retrieved, n_requests_made))
