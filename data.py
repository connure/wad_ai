"""Get data from file and setup dataset graph."""

__author__ = 'Connor Sanchez'

import os
import numpy as np
import tensorflow as tf

NUM_TIME = 2
TEST_DIR = os.path.join('data', 'test')
DATA_FILE = os.path.join('data', 'xyt.txt')
CLASSES_FILE = os.path.join('data', 'classes.txt')
TOKEN_SEP = ', '

NUM_EPOCHS = 5
BATCH_SIZE = 10

NUM_PARALLEL_CALLS = 60
BUFFER_SIZE = 1

SHAPE = (2710, 3384)
SHAPE_RESIZE = (512, 640)

DTYPE_X = tf.uint8
DTYPE_Y = tf.uint16

DTYPE_IN = tf.float32
DTYPE_OUT = tf.int32

NUM_XCH = 3
NUM_YCH = 1

NUM_CLASSES = 8

NUM_DATA = 39180
FOLD = 5

NUM_VAL = NUM_DATA // (FOLD * BATCH_SIZE) * BATCH_SIZE
NUM_TRAIN = NUM_VAL * (FOLD - 1)

NUM_VAL_STEPS = NUM_VAL // BATCH_SIZE
NUM_TRAIN_STEPS = NUM_TRAIN // BATCH_SIZE


def shuffle(array):
    num_arr = len(array)
    perm = np.random.permutation(num_arr)
    shuffled_array = array[perm]
    return shuffled_array


def read_xyt():
    with open(DATA_FILE) as df:
        data = [line.rstrip() for line in df]
    xyt = np.asarray([d.split(TOKEN_SEP) for d in data])
    shuffled_xyt = shuffle(xyt)
    xt = np.asarray([xy[:NUM_TIME] for xy in shuffled_xyt])
    yt = np.asarray([xy[-NUM_TIME:] for xy in shuffled_xyt])
    return (xt[:NUM_TRAIN], yt[:NUM_TRAIN]), (xt[-NUM_VAL:], yt[-NUM_VAL:])


def load_x(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=NUM_XCH)
    image_decoded.set_shape([*SHAPE, NUM_XCH])
    image_resized = tf.image.resize_images(image_decoded, size=SHAPE_RESIZE,
                                           method=tf.image.ResizeMethod.AREA, align_corners=True)
    image_cast = tf.cast(image_resized, dtype=DTYPE_IN) / 255.
    return image_cast


def load_y(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=NUM_YCH, dtype=DTYPE_Y)
    image_decoded.set_shape([*SHAPE, NUM_YCH])
    image_resized = tf.image.resize_images(image_decoded, size=SHAPE_RESIZE,
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    image_squeezed = tf.squeeze(image_resized, axis=-1)
    image_label = tf.cast(image_squeezed // 1000, dtype=tf.int16)
    label = tf.cast(image_label - 33, dtype=tf.uint8)
    not_other = tf.not_equal(label, 4)
    in_range = tf.less_equal(label, 7)
    mask = tf.logical_and(not_other, in_range)
    new = 4 * tf.ones(SHAPE_RESIZE, dtype=tf.uint8)
    new_label = tf.where(mask, x=label, y=new)
    new_cast = tf.cast(new_label, dtype=DTYPE_OUT)
    return new_cast


def load_xy(xi, yi):
    return load_x(xi), load_y(yi)


def load_xx(xi, xj):
    return load_x(xi), load_x(xj)


def next_batch(xt, yt, map_fn):
    xyt = (xt, yt)
    num_shuffle = len(xt)
    dataset = tf.data.Dataset.from_tensor_slices(xyt).shuffle(num_shuffle).repeat(count=NUM_EPOCHS)
    batched_dataset = dataset.map(map_fn, num_parallel_calls=NUM_PARALLEL_CALLS).batch(BATCH_SIZE)
    # .prefetch(BUFFER_SIZE)
    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element


def next_train(xt_train, yt_train):
    class_train = next_batch(xt_train[:, 0], yt_train[:, 0], load_xy)
    coh_train = next_batch(xt_train[:, 0], xt_train[:, 1], load_xx)
    ncoh_train = next_batch(xt_train[:, 0], shuffle(xt_train[:, 1]), load_xx)
    return class_train, coh_train, ncoh_train


def next_val(xt_val, yt_val):
    class_val = next_batch(xt_val[:, 0], yt_val[:, 0], load_xy)
    return class_val


def read_xt_test():
    files = os.listdir(TEST_DIR)
    file_paths = [os.path.join(TEST_DIR, name) for name in files]
    return file_paths


def next_test(xt_test):
    dataset = tf.data.Dataset.from_tensor_slices(xt_test)
    batched_dataset = dataset.map(load_x, num_parallel_calls=NUM_PARALLEL_CALLS).batch(BATCH_SIZE)
    # .prefetch(BUFFER_SIZE)
    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element


def classes():
    with open(CLASSES_FILE) as cf:
        class_str = [line.rstrip() for line in cf]
    keys_vals = [cs.split(TOKEN_SEP) for cs in class_str]
    class_dict = {kv[0]: kv[1] for kv in keys_vals}
    return class_dict


if __name__ == '__main__':
    print('data')
