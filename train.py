# -*- coding: utf-8 -*-
import argparse
import glob
import os
import math
import numpy as np

import chainer
from chainer import cuda
from chainer import serializers
from chainer.optimizers import Adam
from chainer.iterators import MultiprocessIterator

from lapsrn.models import LapSRN, l1_charbonnier
from lapsrn.dataset import ImageDataset


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]

    pred = pred[shave_border:height - shave_border,
                shave_border:width - shave_border]

    gt = gt[shave_border:height - shave_border,
            shave_border:width - shave_border]

    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))

    if rmse == 0:
        return 100
    else:
        return 20 * math.log10(255.0 / rmse)


def main():
    parser = argparse.ArgumentParser(description="LapSRN")
    parser.add_argument("--dataset", type=str, default="\"data/*\"")
    parser.add_argument("--outdirname", type=str, default="./models")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=128)
    parser.add_argument("--model", default=None)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# dataset: {}'.format(args.dataset))
    print('# outdirname: {}'.format(args.outdirname))
    print('# scale: {}'.format(args.scale))
    print('# batchsize: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# steps_per_epoch: {}'.format(args.steps_per_epoch))
    print('# model: {}'.format(args.model))
    print('')

    OUTPUT_DIRECTORY = args.outdirname
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    model = LapSRN()
    if args.model is not None:
        print("Loading model...")
        serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        xp = cuda.cupy
    else:
        xp = np

    optimizer = Adam()
    optimizer.setup(model)

    print("loading dataset...")
    paths = glob.glob(args.dataset)
    train_dataset = ImageDataset(scale=args.scale,
                                 paths=paths,
                                 dtype=xp.float32,
                                 cropsize=96)

    iterator = MultiprocessIterator(train_dataset,
                                    batch_size=args.batchsize,
                                    repeat=True,
                                    shuffle=True)

    step = 0
    epoch = 0
    loss = 0
    print("training...")
    for zipped_batch in iterator:
        lr = chainer.Variable(xp.array([zipped[0] for zipped in zipped_batch]))
        hr = chainer.Variable(xp.array([zipped[1] for zipped in zipped_batch]))

        sr = model(lr)
        loss += l1_charbonnier(sr, hr, model).data
        optimizer.update(l1_charbonnier, sr, hr, model)

        if step % args.steps_per_epoch == 0:
            loss /= args.steps_per_epoch
            print("Epoch: {}, Loss: {}, PSNR: {}".format(epoch,
                                                         loss,
                                                         PSNR(sr.data[0],
                                                              hr.data[0])))
            chainer.serializers.save_npz(
                os.path.join(OUTPUT_DIRECTORY, "model_{}.npz".format(epoch)),
                model)
            epoch += 1
            loss = 0
        step += 1

        if epoch > args.epoch:
            break

    print("Done")


if __name__ == "__main__":
    main()
