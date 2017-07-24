#-*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

#변수 선언
x_data = [[1.,2.,1.,1.], [2.,1.,3.,2.],[3.,1.,3.,4.],[4.,1.,5.,5.],[1.,7.,5.,5.],[1.,2.,5.,6.],[1.,6.,6.,6.],[1.,7.,7.,7.]]
y_data = [[0.,0.,1.], [0.,0.,1.],[0.,0.,1.],[0.,1.,0.],[0.,1.,0.],[1.,0.,0.],[1.,0.,0.],[1.,0.,0.]]

X = tf.placeholder(tf.float32 , shape = [None,4])
Y = tf.placeholder(tf.float32, shape = [None,3])

nb_classes = 3;

W =tf.Variable(tf.random_normal([4,nb_classes]), name = 'weight')
b =tf.Variable(tf.random_normal([nb_classes]),name = 'bias')

#hypothesis 선언
#softmax = exp(logits) / reduce_sum(exp(logits),dim)
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

#cost함수 선언
cost  = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis = 1))

#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

#train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer , feed_dict = {X : x_data, Y : y_data})
        if step % 100 == 0 :
            print(step , sess.run(cost, feed_dict = {X : x_data, Y : y_data}))
