import tensorflow as tf
import os
import numpy as np
from os.path import join, exists

batch_size = 2
steps = 10
epochs = 1
emb_dim = 4
sample_num = epochs * steps * batch_size

checkpoint_dir = 'checkpoint_dir'
meta_name = '0'
saver_dir = join(checkpoint_dir, meta_name)

def data_generator():
    dataset = tf.data.Dataset.from_tensor_slices((np.random.randn(sample_num, emb_dim),\
                        np.random.randn(sample_num)))
    dataset = dataset.repeat(epochs).batch(batch_size)
    iterator = tf.data.make_one_shot_iterator(dataset)
    feature, label = iterator.get_next()
    return feature, label

def model(feature, params=[10, 5, 1]):
    fc1 = tf.layers.dense(feature, params[0], activation=tf.nn.relu, name='fc1')
    fc2 = tf.layers.dense(fc1, params[1], activation=tf.nn.relu, name='fc2')
    fc3 = tf.layers.dense(fc2, params[2], activation=tf.nn.sigmoid, name='fc3')
    out = tf.identity(fc3, name='output')
    return out

def train():
    feature, label = data_generator()
    output = model(feature)
    loss = tf.reduce_mean(tf.square(output - label))
    train_op = tf.train.AdamOptimizer(learning_rate=0.1, name='Adam').minimize(loss)
    saver = tf.train.Saver()


    if exists(checkpoint_dir):
        os.system('rm -rf {}'.format(checkpoint_dir))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            local_step = 0
            save_freq = 2
            while True:
                local_step += 1
                _, loss_val = sess.run([train_op, loss])
                if local_step % save_freq == 0:
                    saver.save(sess, saver_dir)
                print('loss: {:.4f}'.format(loss_val))
        except tf.errors.OutOfRangeError:
            print("train end!")


if __name__ == '__main__':
    train()
