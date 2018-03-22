#!/usr/bin/python
#Improved version from Daniel Maturana

import os
import gzip
import struct
import subprocess
import sys


YANN_BASE = "http://yann.lecun.com/exdb/mnist/"


def die(msg):
    print(msg)
    sys.exit(1)


def convert(fname, fname_out):
    print('converting %s' % fname)
    with gzip.open(fname, 'r') as fin, open(fname_out, 'w') as fout:
        # important to read metadata in case Mr. Yann decides to change MNIST
        magic = struct.unpack('>i', fin.read(4))[0]
        if magic == 2049:
            items = struct.unpack('>i', fin.read(4))[0]
            for _ in range(items):
                fout.write("%d\n" % ord(fin.read(1)))
        elif magic == 2051:
            images, rows, cols = struct.unpack('>3i', fin.read(12))
            imgsize = rows*cols
            for _ in range(images):
                image = struct.unpack('%dB' % imgsize, fin.read(imgsize))
                fout.write(" ".join(str(px-128) for px in image)+"\n")
        else:
            die('%s is not MNIST!!!' % fname)
    print('saving converted file to %s' % fname_out)


def dl(fn):
    print('downloading %s' % fn)
    res = subprocess.call(['wget', '%s/%s' % (YANN_BASE, fn)])
    if res != 0:
        die("Can't download %s" % fn)


if __name__ == "__main__":
    files = ["train-images-idx3-ubyte.gz",
             "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz",
             "t10k-labels-idx1-ubyte.gz"]

    outfiles = ["mnistX.txt",
                "mnistYn.txt",
                "mnistXe.txt",
                "mnistYen.txt"]

    for ifn, ofn in zip(files, outfiles):
        if not os.path.exists(ifn):
            dl(ifn)
        if not os.path.exists(ofn):
            convert(ifn, ofn)
