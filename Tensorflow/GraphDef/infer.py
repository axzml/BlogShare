#_*_ coding:utf-8 _*_
## infer.py
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import dtypes
from tensorflow.python.tools import optimize_for_inference_lib
import os
from os.path import join, exists
import numpy as np

emb_dim = 4
checkpoint_dir = 'checkpoint_dir'
meta_name = '0'
saver_dir = join(checkpoint_dir, meta_name)
meta_file = saver_dir + '.meta'
model_file = tf.train.latest_checkpoint(checkpoint_dir)

np.random.seed(123)
test_data = np.random.randn(4, emb_dim)

def read_pb_meta(meta_file):
    meta_graph_def = meta_graph.read_meta_graph_file(meta_file)
    return meta_graph_def


def update_node(graph, src_node_name, tar_node):
    """
    @params:
        graph : tensorflow Graph object
        src_node_name : source node name to be modified
        tar_node : target node
    """
    input = graph.get_tensor_by_name('{}:0'.format(src_node_name))
    for op in input.consumers():
        idx_list = []
        for idx, item in enumerate(op.inputs):
            if src_node_name in item.name:
                idx_list.append(idx)
        for idx in idx_list:
            op._update_input(idx, tar_node)



def eval_graph():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, model_file)
        output = sess.run(['output:0'], feed_dict={
            'IteratorGetNext:0': test_data
        })
        print('eval_graph:\n{}'.format(output))


def modify_graph():
    meta_graph_def = read_pb_meta(meta_file)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(meta_graph_def.graph_def, name="")
        input_ph = tf.placeholder(tf.float64, [None, emb_dim], name='input')
        update_node(graph, 'IteratorGetNext', input_ph)

    with tf.Session(graph=graph) as sess:
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, model_file)
        output = sess.run(['output:0'], feed_dict={
            'input:0': test_data
        })
        print('modify_graph:\n{}'.format(output))


if __name__ == '__main__':
    eval_graph()
    modify_graph()
