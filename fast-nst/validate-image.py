import glob
from PIL import Image
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('paths', action='append', help='Pattern / paths to images to validate.')
parser.add_argument('--reject-path', help='Directory to move invalid files to.')
parser.add_argument('--convert', help='Extension / format to convert images to.')

args = parser.parse_args()

def rejectFile(fn):
    basename = os.path.basename(fn)
    newPath = os.path.join(args.reject_path, basename)
    os.rename(fn, newPath)

n_images_validated = 0
for pattern in args.paths:
    for fn in glob.iglob(pattern):
        if not os.path.exists(fn):
            continue

        n_images_validated += 1
        if n_images_validated % 100 == 0:
            print("Processed {} images so far...".format(n_images_validated))

        stem, ext = os.path.splitext(fn)
        basename = os.path.basename(fn)

        if args.convert is None:
            ext_fmt = ''
            if ext == '.jpg' or ext == '.jpeg':
                ext_fmt = 'JPEG'
            elif ext == '.png':
                ext_fmt = 'PNG'
            else:
                print("Invalid image: {} (extension not recognized)".format(fn))
                rejectFile(fn)
                continue

        try:
            with Image.open(fn) as im:
                actual_fmt = im.format
                if args.convert is None:
                    if actual_fmt != 'JPEG' and actual_fmt != 'PNG':
                        print("Invalid image: {} (format {} not valid)".format(fn, actual_fmt))
                        rejectFile(fn)
                        continue

                    if actual_fmt != ext_fmt:
                        newPath = stem
                        if actual_fmt == 'JPEG':
                            newPath += '.jpg'
                        elif actual_fmt == 'PNG':
                            newPath += '.png'
                        else:
                            print("Invalid image: {} (format {} not valid)".format(fn, actual_fmt))
                            rejectFile(fn)
                            continue

                        print("Corrected image extension: {} --> {}".format(fn, newPath))
                        os.rename(fn, newPath)
                else:
                    im.save(stem+'.'+args.convert)
                    os.remove(fn)
        except IOError as err:
            if err.errno == os.errno.EBUSY or err.errno == os.errno.ENOENT:
                pass
            else:
                print("Invalid image: {} (IOError on read: {})".format(fn, str(err)))
                rejectFile(fn)
