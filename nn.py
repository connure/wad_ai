"""Setup the network and train, save, and validate."""

__author__ = 'Connor Sanchez'

import numpy as np
import tensorflow as tf

import data
import util

# set env var to only use listed gpu(s)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# model constants
SAVE_DIR = 'models/'
SAVE_PATH = 'models/model.ckpt'

# data constants
NUM_EPOCHS = data.NUM_EPOCHS
BATCH_SIZE = data.BATCH_SIZE

SHAPE = data.SHAPE_RESIZE

NUM_XCH = data.NUM_XCH
NUM_YCH = data.NUM_YCH
NUM_CLASSES = data.NUM_CLASSES

DTYPE_IN = data.DTYPE_IN
DTYPE_OUT = data.DTYPE_OUT

SHAPE_IN = [BATCH_SIZE, *SHAPE, NUM_XCH]
SHAPE_OUT = [BATCH_SIZE, *SHAPE]
SHAPE_PRED = [BATCH_SIZE, *SHAPE, NUM_CLASSES]

NUM_TRAIN_STEPS = data.NUM_TRAIN_STEPS
NUM_VAL_STEPS = data.NUM_VAL_STEPS

# hyper-parameters
GAMMA = float(NUM_CLASSES)  # we already reduce mean so no need to here, we don't want to over channel dim
DELTA = 1.  # center of [0, 2]
ALPHA = .001 * .5 / np.e  # scale coh loss to range of cross ent loss
BETA = .001 * .5 / np.e  # scale ncoh loss to range of cross ent loss
EPSILON = .001


# define the u-net
def unet():
    inputs = tf.placeholder(dtype=DTYPE_IN, shape=SHAPE_IN)
    tf.add_to_collection('inputs', inputs)
    enc0 = inputs
    enc1 = util.conv3x3_x2_relu(enc0, SHAPE_IN[-1], 16)
    enc2 = util.down_and_conv(enc1, 16, 32)
    enc3 = util.down_and_conv(enc2, 32, 64)
    enc4 = util.down_and_conv(enc3, 64, 128)
    enc5 = util.down_and_conv(enc4, 128, 256)
    dec5 = enc5
    dec4 = util.up_and_concat_and_conv(dec5, enc4, 256, 128)
    dec3 = util.up_and_concat_and_conv(dec4, enc3, 128, 64)
    dec2 = util.up_and_concat_and_conv(dec3, enc2, 64, 32)
    dec1 = util.up_and_concat_and_conv(dec2, enc1, 32, 16)
    dec0 = util.conv1x1_relu(dec1, 16, NUM_CLASSES)
    outputs = dec0
    tf.add_to_collection('outputs', outputs)
    return (inputs), (outputs)


# net helpers
def predict(logits):
    pred = tf.argmax(logits, axis=-1, output_type=DTYPE_OUT)
    return pred


def class_loss(labels, logits):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, reduction=tf.losses.Reduction.MEAN)
    return loss


def l1_loss(logits_m, logits_n):
    pred_m = tf.nn.softmax(logits_m)
    pred_n = tf.nn.softmax(logits_n)
    norm_l1 = tf.losses.absolute_difference(pred_n, pred_m, reduction=tf.losses.Reduction.MEAN)
    return norm_l1


def class_acc(labels, logits):
    labels_pred = predict(logits)
    correct_preds = tf.equal(labels, labels_pred)
    correct_cast = tf.cast(correct_preds, dtype=DTYPE_IN)
    acc = tf.reduce_mean(correct_cast)
    return acc


# training nets
def train_loss_and_acc(logits):
    labels = tf.placeholder(dtype=DTYPE_OUT, shape=SHAPE_OUT)
    loss = class_loss(labels, logits)
    acc = class_acc(labels, logits)
    return (labels), (loss, acc)


def coh_loss(logits_cur):
    logits_next = tf.placeholder(dtype=DTYPE_IN, shape=SHAPE_PRED)
    norm_l1 = l1_loss(logits_cur, logits_next)
    loss = ALPHA * GAMMA * norm_l1
    return (logits_next), (loss)


def ncoh_loss(logits_i):
    logits_j = tf.placeholder(dtype=DTYPE_IN, shape=SHAPE_PRED)
    norm_l1 = l1_loss(logits_i, logits_j)
    loss = BETA * tf.maximum(0., DELTA - GAMMA * norm_l1)
    return (logits_j), (loss)


# validation nets
def val_loss_and_acc(logits):
    labels = tf.placeholder(dtype=DTYPE_OUT, shape=SHAPE_OUT)
    loss = class_loss(labels, logits)
    acc = class_acc(labels, logits)
    return (labels), (loss, acc)


# main
def main(_):
    # read classes
    class_dict = data.classes()
    print(class_dict)

    # read data
    (xt_train, yt_train), (xt_val, yt_val) = data.read_xyt()

    # graphs
    graph = tf.Graph()

    # default graph context
    with graph.as_default():
        # datasets
        class_train, coh_train, ncoh_train = data.next_train(xt_train, yt_train)
        class_val = data.next_val(xt_val, yt_val)

        # model nets
        (inputs), (outputs) = unet()

        # training nets
        (labels_train), (loss_train, acc_train) = train_loss_and_acc(outputs)
        (logits_coh), (loss_coh) = coh_loss(outputs)
        (logits_ncoh), (loss_ncoh) = ncoh_loss(outputs)

        # training merged summaries
        merged_train = tf.summary.merge([tf.summary.scalar('loss_train', loss_train),
                                         tf.summary.scalar('acc_train', acc_train)])
        merged_coh = tf.summary.scalar('loss_coh', loss_coh)
        merged_ncoh = tf.summary.scalar('loss_ncoh', loss_ncoh)

        # optimizers
        optim_train = tf.train.GradientDescentOptimizer(EPSILON).minimize(loss_train)
        optim_coh = tf.train.GradientDescentOptimizer(EPSILON).minimize(loss_coh)
        optim_ncoh = tf.train.GradientDescentOptimizer(EPSILON).minimize(loss_ncoh)

        # validation nets
        (labels_val), (loss_val, acc_val) = val_loss_and_acc(outputs)

        # validation merged summaries
        merged_val = tf.summary.merge([tf.summary.scalar('loss_val', loss_val),
                                       tf.summary.scalar('acc_val', acc_val)])

        # summary writer
        writer = tf.summary.FileWriter(SAVE_DIR, graph=graph)

        # global variables initializer
        init = tf.initializers.global_variables()

        # saver
        saver = tf.train.Saver()

        # configure session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

    # session context
    with tf.Session(graph=graph, config=config) as sess:
        # initialize global variables
        sess.run(init)

        # epoch loop
        for ei in range(NUM_EPOCHS):
            # training loop
            print('Training...')
            for ti in range(NUM_TRAIN_STEPS):
                print('Epoch: {}, Training step: {}'.format(ei, ti))

                # get next training data batch
                (x_train, y_train), (x_cur, x_next), (x_i, x_j) = sess.run([class_train, coh_train, ncoh_train])

                # training step: classes
                summary_train, _ = sess.run([merged_train, optim_train],
                                            feed_dict={inputs: x_train, labels_train: y_train})

                # training step: coherence
                logits_next = sess.run(outputs, feed_dict={inputs: x_next})
                summary_coh, _ = sess.run([merged_coh, optim_coh], feed_dict={inputs: x_cur, logits_coh: logits_next})

                # training step: non-coherence
                logits_j = sess.run(outputs, feed_dict={inputs: x_j})
                summary_ncoh, _ = sess.run([merged_ncoh, optim_ncoh], feed_dict={inputs: x_i, logits_ncoh: logits_j})

                # add training summaries
                global_step_train = ei * NUM_TRAIN_STEPS + ti
                writer.add_summary(summary_train, global_step=global_step_train)
                writer.add_summary(summary_coh, global_step=global_step_train)
                writer.add_summary(summary_ncoh, global_step=global_step_train)

            # save
            print('Saving...')
            global_step_save = ei * NUM_TRAIN_STEPS + NUM_TRAIN_STEPS
            save_path = saver.save(sess, SAVE_PATH, global_step=global_step_save)
            print('Epoch: {}, Saved in path: {}'.format(ei, save_path))

            # validation loop
            print('Validating...')
            for vi in range(NUM_VAL_STEPS):
                print('Epoch: {}, Validation step: {}'.format(ei, vi))

                # get next validation data batch
                (x_val, y_val) = sess.run(class_val)

                # validation step: classes
                summary_val = sess.run(merged_val, feed_dict={inputs: x_val, labels_val: y_val})

                # add validation summaries
                global_step_val = ei * NUM_VAL_STEPS + vi
                writer.add_summary(summary_val, global_step=global_step_val)


if __name__ == '__main__':
    tf.app.run()
