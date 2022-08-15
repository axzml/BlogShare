#_*_ coding:utf-8 _*_
import tensorflow as tf
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
test_data = np.random.randn(4, emb_dim) ## 生成测试数据

def eval_graph():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, model_file)
        for op in sess.graph.get_operations():
            print(op.name, op.values())
        output = sess.run(['output:0'], feed_dict={
            'IteratorGetNext:0': test_data
        })
        print('eval_graph:\n{}'.format(output))

if __name__ == '__main__':
    eval_graph()