"""Import the trained network and test it."""

__author__ = 'Connor Sanchez'

import os
import skimage.io as io
import numpy as np
import tensorflow as tf

import data

VIDEO = 'road02_cam_5_video_5_image_list_test'
SAVE_PATH = os.path.join('models', 'model.ckpt-6264')
IN_DIR = os.path.join('preds', 'in')
OUT_DIR = os.path.join('preds', 'out')

META_PATH = SAVE_PATH + '.meta'
X_DIR = os.path.join('data', 'test')
LIST_PATH = os.path.join('data', 'test_video_list_and_name_mapping', 'list_test', VIDEO + '.txt')
MAPPING_PATH = os.path.join('data', 'test_video_list_and_name_mapping', 'list_test_mapping',
                            'md5_mapping_' + VIDEO + '.txt')


# main
def main(_):
    # get testing data
    with open(LIST_PATH) as f:
        paths = [path.strip() for path in f]
    with open(MAPPING_PATH) as f:
        mapping = {line.split('\t')[1].strip(): line.split('\t')[0].strip() for line in f}
    md5s = list(map(lambda path: mapping[path], paths))
    xt_test = list(map(lambda md5: os.path.join(X_DIR, ''.join([md5, '.jpg'])), md5s))

    names = list(map(lambda path: path.split('\\')[-1].split('.')[0], paths))
    in_test = list(map(lambda name: os.path.join(IN_DIR, ''.join([name, '.jpg'])), names))
    out_test = list(map(lambda name: os.path.join(OUT_DIR, ''.join([name, '.png'])), names))

    # graphs
    graph = tf.Graph()

    # default graph context
    with graph.as_default():
        # testing data network
        data_test = data.next_test(xt_test)

        # configure session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

    # session context
    with tf.Session(graph=graph, config=config) as sess:
        # import the meta-graph and restore the variables
        saver = tf.train.import_meta_graph(META_PATH)
        saver.restore(sess, SAVE_PATH)

        # get the network io
        inputs = graph.get_collection('inputs')[0]
        outputs = graph.get_collection('outputs')[0]

        # prediction
        pred = tf.cast(33 + tf.argmax(outputs, axis=-1, output_type=tf.int32), dtype=tf.uint8)

        for bi in range(len(xt_test) // data.BATCH_SIZE):
            # get next testing data
            x_test = sess.run(data_test)

            # get outputs
            labels = sess.run(pred, feed_dict={inputs: x_test})

            # save images
            for ii in range(len(labels)):
                ti = bi * data.BATCH_SIZE + ii
                io.imsave(in_test[ti], np.uint8(255 * x_test[ii]))
                io.imsave(out_test[ti], labels[ii])


if __name__ == '__main__':
    tf.app.run()
