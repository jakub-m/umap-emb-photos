#!/usr/bin/env python3

from PIL import Image
from torchvision import transforms
from typing import Callable
import argparse
import fastai.vision.models as vision_models
import logging
import numpy as np
import os
import sys
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


def main():
    opts = get_options()
    logging.basicConfig(level=logging.DEBUG if opts.verbose else logging.INFO)
    logger.debug(f"{opts=}")
    logger.debug(f"Load model {opts.model}")
    os.makedirs(opts.out_dir, exist_ok=True)
    model_to_load = {
        'resnet18': vision_models.resnet18,
        # resnet18, resnet34, resnet50, resnet101, resnet152
        'resnet34': vision_models.resnet34,
    }[opts.model]

    if opts.model == "resnet18":
        model_to_load = vision_models.resnet18
        # More on normalization:
        # https://pytorch.org/hub/pytorch_vision_resnet/
        # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
        preprocess = transforms.Compose([
            transforms.Resize(256), # TODO should rescale if the actual images are already 224?
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif opts.model == "resnet34":
        model_to_load = vision_models.resnet34
        preprocess = transforms.Compose([
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"bad model: {opts.model}")

    model = model_to_load(progress=True)

    logger.debug(f"Model:\n{model}")
    # Remove the last linear layer and extract 512 (for resnet18) features. In resnet 18 the last layer is
    #   (fc): Linear(in_features=512, out_features=1000, bias=True)
    model = strip_last_layer(model)
    files_so_far = 0
    for files_batch in iter_chunked(opts.files, opts.batch):
        prc_so_far = files_so_far / len(opts.files) * 100
        logger.info(f"Progress: {prc_so_far:.1f}% ({files_so_far}/{len(opts.files)})")
        logger.debug(f"Process batch of {len(files_batch)} files: {files_batch}")
        feats = get_features_for_files(files_batch, model, preprocess)
        for in_filename, feat in feats:
            logger.debug(f"{in_filename=}, {feat.shape=}")
            out_filename = f"{os.path.basename(in_filename)}.npy"
            np.save(os.path.join(opts.out_dir, out_filename), feat)
        files_so_far += len(files_batch)

def get_options() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="extract visual features (embeddings) from images")
    p.add_argument('files', nargs='+', help="Image paths. If there is a single parameter \"-\" then use stdin.")
    p.add_argument('-v', '--verbose', action="store_true", default=False)
    p.add_argument('-b', '--batch', type=int, default=128)
    p.add_argument('-o', '--output', dest="out_dir", default=None, help="Directory where to store the outputs, one output file per input file, same file name, diffrent extension.")
    p.add_argument('-m', '--model', default="resnet18")
    args = p.parse_args()
    if args.files == ['-']:
        args.files = [p.strip() for p in sys.stdin]
    if not args.out_dir:
        args.out_dir = f"output_features_{args.model}"
    return args

def get_features_for_files(filenames: list[str], stripped_model: nn.Module, preprocess: Callable) -> list[tuple[str, np.ndarray]]:
    # Open and concatenate all the input images into a single batch.
    images = [Image.open(f) for f in filenames]
    images = [im.convert('RGB') for im in images] # In case the image is CMYK
    images = [preprocess(im) for im in images]
    images = [im.unsqueeze(0) for im in images]
    input_batch = torch.concat(images, dim=0)

    with torch.no_grad():
        output = stripped_model(input_batch).squeeze(dim=2).squeeze(dim=2)
        # The output should be a tensor of shape(n_images, n_features)
    features = output.numpy() # type: np.ndarray
    logger.info(f"{input_batch.shape=} {features.shape=}")
    # Zip features with filenames
    return [(filenames[i], features[i,:]) for i in range(features.shape[0])]

def strip_last_layer(m: nn.Module) -> nn.Module:
    children = [l for l in m.children()]
    return nn.Sequential(*children[:-1])

def iter_chunked(input_list: list, chunk_size:int):
    return (input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size))


if __name__ == "__main__":
    main()