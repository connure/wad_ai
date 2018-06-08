#!/usr/bin/env python3

"""Count lines of code."""

__author__ = 'Connor Sanchez'

import os
import sys


def loc(path):
    ps = os.listdir(path=path)
    n = 0
    for fn in ps:
        fp = os.path.join(path, fn)
        with open(fp) as f:
            n += sum(1 for row in f)
    return n


if __name__ == '__main__':
    print(loc(sys.argv[1]))
