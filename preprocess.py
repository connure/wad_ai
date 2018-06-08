#!/usr/bin/env python3

"""Preprocess data files."""

__author__ = 'Connor Sanchez'

import os

TOKEN_SEP = '\t'
PATH_SEP = '\\'
DATA_DIR = 'data'

LIST_DIR = os.path.join(DATA_DIR, 'train_video_list')
X_DIR = os.path.join(DATA_DIR, 'train_color')
Y_DIR = os.path.join(DATA_DIR, 'train_label')
XYT_PATH = os.path.join(DATA_DIR, 'xyt.txt')


def main():
    xyt = []
    lists = os.listdir(path=LIST_DIR)
    for list_i in lists:
        list_path = os.path.join(LIST_DIR, list_i)
        with open(list_path) as lst:
            data = [line.rstrip() for line in lst]
        xy = [row.split(TOKEN_SEP) for row in data]
        xpp = [row[0].split(PATH_SEP)[-1] for row in xy]
        ypp = [row[1].split(PATH_SEP)[-1] for row in xy]
        n = len(xpp)
        for ii in range(n - 1):
            xyt.append('{}, {}, {}, {}{}'.format(os.path.join(X_DIR, xpp[ii]),
                                                 os.path.join(X_DIR, xpp[ii + 1]),
                                                 os.path.join(Y_DIR, ypp[ii]),
                                                 os.path.join(Y_DIR, ypp[ii + 1]),
                                                 os.linesep))
    with open(XYT_PATH, mode='w') as xyt_txt:
        xyt_txt.write(''.join(xyt))


if __name__ == '__main__':
    main()
