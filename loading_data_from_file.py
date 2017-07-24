#-*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

xy = np.loadtxt("mlr06.csv",delimiter = ',', dtype = np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape , x_data, len(x_data))
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32 , shape = [None,4])
Y = tf.placeholder(tf.float32, shape = [None,1])

W = tf.Variable(tf.random_normal([4,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')


#hopothesis 함수 선언
hypothesis = tf.matmul(X,W) + b

#cost 함수 선언
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#session 만들고 train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val , _ = sess.run([cost,hypothesis,train], feed_dict = {X : x_data , Y : y_data})
    if step %100 == 0:
        print(step, cost_val, hy_val)
