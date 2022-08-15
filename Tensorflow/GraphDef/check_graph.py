#_*_ coding:utf-8 _*_
## check_graph.py
import tensorflow as tf
from tensorflow.python.framework import meta_graph
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from google.protobuf import text_format

import os
from os.path import join, exists
import numpy as np

checkpoint_dir = 'checkpoint_dir'
meta_name = '0'
saver_dir = join(checkpoint_dir, meta_name)
meta_file = saver_dir + '.meta'
model_file = tf.train.latest_checkpoint(checkpoint_dir)


def read_pb_meta(meta_file):
    meta_graph_def = meta_graph.read_meta_graph_file(meta_file)
    return meta_graph_def

def read_txt_meta(txt_meta_file):
    meta_graph = MetaGraphDef()
    with open(txt_meta_file, 'rb') as f:
        text_format.Merge(f.read(), meta_graph)
    return meta_graph

def read_pb_graph(graph_file):
    try:
        with tf.gfile.GFile(graph_file, 'rb') as pb:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(pb.read())
    except IOError as e:
        raise Exception("Parse '{}' Failed!".format(graph_file))
    return graph_def


def check_graph_def(graph_def):
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            name=""
        )
        print('===> {}'.format(type(graph)))
        for op in graph.get_operations():
            print(op.name, op.values())  ## 打印网络结构

def check_graph(graph_file):
    graph_def = read_pb_graph(graph_file)
    check_graph_def(graph_def)
    

if __name__ == '__main__':
    check_graph_def(read_pb_meta(meta_file).graph_def)