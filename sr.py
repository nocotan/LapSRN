# -*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
from PIL import Image

import chainer
from chainer import cuda
from chainer import serializers
from lapsrn.models import LapSRN


def clip_img(x):
    return np.uint8(0 if x < 0 else (255 if x > 255 else x))


def img2variable(img, xp):
    return chainer.Variable(xp.array([img.transpose(2, 0, 1)], dtype=xp.float32))


def variable2img(x):
    x.to_cpu()
    print(x.data.max())
    print(x.data.min())
    img = (np.vectorize(clip_img)(x.data[0, :, :, :])).transpose(1, 2, 0)
    return img


def main():
    parser = argparse.ArgumentParser(description="LapSRN Super-Resolution")
    parser.add_argument("--model")
    parser.add_argument("--image")
    parser.add_argument("--scale", default=4)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# model: {}'.format(args.model))
    print('# image: {}'.format(args.image))
    print('# scale: {}'.format(args.scale))
    print('')

    print("loading model...")
    model = LapSRN()
    serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        xp = cuda.cupy
    else:
        xp = np

    img = Image.open(args.image)
    img = xp.asarray(img, dtype=xp.float32)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    print("inferring...")
    img_variable = img2variable(img, xp)
    img_variable_sr = model(img_variable)
    img_sr = variable2img(img_variable_sr)

    print("saving HR image...")
    cv2.imwrite("./result.png", cv2.cvtColor(img_sr, cv2.COLOR_RGB2BGR))

    print("Done")


if __name__ == "__main__":
    main()
