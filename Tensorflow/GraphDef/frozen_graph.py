#_*_ coding:utf-8 _*_
## frozen_graph.py
import tensorflow as tf
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

def check_graph_def(graph_def):
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            name=""
        )
        print('===> {}'.format(type(graph)))
        for op in graph.get_operations():
            print(op.name, op.values())  ## 打印网络结构

def write_frozen_graph():
    meta_graph_def = read_pb_meta(meta_file)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(meta_graph_def.graph_def, name="")
        input_ph = tf.placeholder(tf.float64, [None, emb_dim], name='input')
        update_node(graph, 'IteratorGetNext', input_ph)

    with tf.Session(graph=graph) as sess:
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, model_file)

        input_node_names = ['input']
        ##placeholder_type_enum = [dtypes.float64.as_datatype_enum]
        placeholder_type_enum = [input_ph.dtype.as_datatype_enum]
        output_node_names = ['output']
        ## 对 graph 进行优化, 把和 inference 无关的节点给删除, 比如 Saver 有关的节点
        graph_def = optimize_for_inference_lib.optimize_for_inference(
            graph.as_graph_def(), input_node_names, output_node_names, placeholder_type_enum
        )
        check_graph_def(graph_def)
        ## 将 ckpt 转换为 frozen_graph, 网络权重和结构写入统一 pb 文件中, 参数以 Const 的形式保存
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, 
            graph_def, output_node_names)
        out_graph_path = os.path.join('.', "frozen_model.pb")
        with tf.gfile.GFile(out_graph_path, "wb") as f:
            f.write(frozen_graph.SerializeToString())


def read_frozen_graph():

    with tf.Graph().as_default() as graph:
        graph_def = tf.GraphDef()
        with open("frozen_model.pb", 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        
        # print(graph_def)
    
    with tf.Session(graph=graph) as sess:
        output = sess.run(['output:0'], feed_dict={
            'input:0': test_data
        })
        print('frozen_graph:\n{}'.format(output))   


if __name__ == '__main__':
    write_frozen_graph()
    read_frozen_graph()
