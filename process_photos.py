#!/usr/bin/env python3
from io import BytesIO
from PIL import Image
import os
import argparse
import logging
import hashlib

logger = logging.getLogger(__name__)

def main():
    opts = get_options()
    logging.basicConfig(level=logging.DEBUG if opts.verbose else logging.INFO)
    file_list = load_file_list(opts.file_list)
    file_list = set(file_list)
    logger.debug(f"{opts=}")
    extensions = [f".{e}" for e in opts.extensions]
    logger.debug(f"{extensions=}")
    torch_thumbs_dir = os.path.join(opts.output, "torch_thumbs")
    os.makedirs(torch_thumbs_dir, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(opts.base_dir, topdown=True):
        dirnames_filtered = [d for d in dirnames if not d.startswith('.')]
        dirnames.clear()
        dirnames.extend(dirnames_filtered)

        for filename in filenames:
            if filename.startswith('.'):
                continue
            if not any(filename.endswith(e) for e in extensions):
                continue
            filepath = os.path.join(dirpath, filename)
            if filepath in file_list:
                logger.info(f"Already processed {filepath}")
                continue
            logger.info(f"Process {filepath}")
            try:
                small_image = transform_image(filepath, thumb_wh=(opts.thumb_size, opts.thumb_size))
            except Exception as ex:
                logger.error(f"Error for {filename}: {ex}")
                continue
            hexdigest = hashlib.md5(small_image).hexdigest()
            outfile = os.path.join(torch_thumbs_dir,f"{hexdigest}.jpg")
            print(f"{hexdigest}\t{filepath}")
            with open(outfile, 'wb') as h:
                h.write(small_image)


def get_options() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process photos")
    p.add_argument("-d", "--dir", dest="base_dir", default=".", help="start crawl here")
    p.add_argument("-l", "--file-list", default="file_list.meta")
    p.add_argument("-x", "--extensions", default="jpg,png", help="Comma-separated extensions")
    p.add_argument("-o", "--output", default="output", help="Output directory")
    p.add_argument("-s", "--thumb-size", type=int, default=224)
    p.add_argument("-v", "--verbose", action="store_true", default=False)
    args = p.parse_args()
    args.extensions = [x.strip() for x in args.extensions.lower().split(",")]
    return args


def load_file_list(path: str) -> list[str]:
    meta_list = []
    with open(path) as h:
        for line in h:
            line = line.strip()
            if line.startswith('#'):
                continue
            parts = line.split("\t", 1)
            file_md5_hex, original_file_path = parts
            #m = (bytearray.fromhex(file_md5_hex), original_file_path)
            meta_list.append(original_file_path)
    return meta_list

def transform_image(filepath: str, thumb_wh: tuple[int,int]) -> bytes:
    im = Image.open(filepath)
    im.thumbnail(thumb_wh)
    with BytesIO() as output:
        im.save(output, 'JPEG', optimize=False)
        return output.getvalue()
   


if __name__=="__main__":
    main()