# -*- coding: utf-8 -*-
import chainer.links as L
import chainer.functions as F
from chainer import Chain, initializers


class ConvBlock(Chain):
    def __init__(self):
        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.c2 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.c3 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.c4 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.c5 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.c6 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.c7 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.c8 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.c9 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.c10 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1)
            self.dc1 = L.Deconvolution2D(None, 64, ksize=4, stride=2, pad=1)

    def __call__(self, x):
        h = F.leaky_relu(self.c1(x))
        h = F.leaky_relu(self.c2(x))
        h = F.leaky_relu(self.c3(x))
        h = F.leaky_relu(self.c4(x))
        h = F.leaky_relu(self.c5(x))
        h = F.leaky_relu(self.c6(x))
        h = F.leaky_relu(self.c7(x))
        h = F.leaky_relu(self.c8(x))
        h = F.leaky_relu(self.c9(x))
        h = F.leaky_relu(self.c10(x))
        h = F.leaky_relu(self.dc1(x))

        return h


class LapSRN(Chain):
    def __init__(self, scale=4):
        initializer = initializers.HeNormal()
        super(LapSRN, self).__init__()
        with self.init_scope():
            self.scale = scale
            self.c1 = L.Convolution2D(None, 64, ksize=3, stride=1, pad=1,
                                      initialW=initializer)

            self.i = L.Deconvolution2D(None, 3, ksize=4, stride=2, pad=1)
            self.r = L.Convolution2D(None, 3, ksize=3, stride=1, pad=1,
                                     initialW=initializer)
            self.f = ConvBlock()

    def __call__(self, x):
        h = F.relu(self.c1(x))
        f = self.f(h)
        r = self.r(f)
        i = self.i(x)

        hr = i + r
        for j in range(int((self.scale/2) - 1)):
            f = self.f(f)
            r = self.r(f)
            i = self.i(hr)

            hr = i + r

        return hr


def l1_charbonnier(x, y, model, eps=1e-6):
    t = x  - y
    loss = F.sum(F.sqrt(t*t+eps))
    return loss
