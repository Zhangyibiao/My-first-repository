# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data#从官网下载并且读取数据

import tensorflow as tf#导入TensorFlow

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])#声明变量x。x表示所有的手写体图片。None表示任意维度的向量
  W = tf.Variable(tf.zeros([784, 10]))#某个像素在某个分类下的证据（w用一个784维度的向量表示像素值，用10维度的向量表示分类，而2个向量进行乘法运算（或者说2个向量的笛卡尔集）就表示“某个像素在某个分类下的证据”。）
  b = tf.Variable(tf.zeros([10]))#10个分类的偏移值（在计算的过程中，变量是样本训练的基础，通过不断调整变量来实现一个收敛过程找到变量的最佳值）
  y = tf.matmul(x, W) + b#得到每个分类的偏移量

  # Define loss and optimizer#定义损失和优化器
  y_ = tf.placeholder(tf.float32, [None, 10])#增加一个占位符来输入真实分布值

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))#实现交叉熵功能
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)#梯度递减算法为训练的优化器

  sess = tf.InteractiveSession()#在 InteractiveSession 中启用该模型
  tf.global_variables_initializer().run()#初始化所有的变量
  # Train
  for _ in range(1000):#随机梯度递减训练
    batch_xs, batch_ys = mnist.train.next_batch(100)#从训练集随机抽取一百条数据
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})#执行train_step将占位数据替换成从测试图片库mnist.train中获取的参数。

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))# tf.argmax 是一个非常有用的方法，它能够找到张量中某个列表最高数值的条目索引。
  # 例如 tf.argmax(y,1) 是找到张量y第二个向量的最大值
  # （图标标签是0~9,softmax计算完成后会得到一个分布概率，argmax方法就是找到每一个图片对应的最高概率）
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)