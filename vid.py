#!/usr/bin/env python3

"""Write a comparison video."""

__author__ = 'Connor Sanchez'

import os
import numpy as np
import skimage.io as io

IN_DIR = os.path.join('preds', 'in')
OUT_DIR = os.path.join('preds', 'out')
NEW_DIR = os.path.join('preds', 'new')

INPUTS = list(map(lambda row: os.path.join(IN_DIR, row), sorted(os.listdir(IN_DIR))))
OUTPUTS = list(map(lambda row: os.path.join(OUT_DIR, row), sorted(os.listdir(OUT_DIR))))

CMAP = np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255],
                   [0, 0, 0], [255, 0, 255], [255, 255, 0], [0, 255, 255]], dtype=np.uint8)


def cmap(pixel):
    return CMAP[pixel - 33]


print(len(INPUTS))
nn = 0

for input, output in zip(INPUTS, OUTPUTS):
    nn += 1
    print(nn)

    in_img = io.imread(input)
    out_img = io.imread(output)
    out_cmap = np.asarray(list(map(cmap, out_img.reshape(-1))),
                          dtype=np.uint8).reshape(in_img.shape)
    new_img = np.concatenate([in_img, out_cmap], axis=1)

    io.imsave(os.path.join(NEW_DIR, '{:03d}{}'.format(nn, '.png')), new_img)
