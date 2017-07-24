#-*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

#H(x1,x2,x3) = x1*w1 + x2*w2 + x2*w2

#그래프 먼저 그린다
#1. 변수 선언
#x1_data = [73.,93.,89.,96.,73.]
#x2_data = [80.,88.,91.,98.,66.]
#x3_data = [75.,93.,90.,100.,70.]
#y_data = [152.,185.,180.,196.,142.]

x_data = [ [73.,80.,75.] , [93.,88.,93.] , [89.,91.,90.], [96.,98.,100.], [73.,66.,70.]]
y_data = [[152.], [185.], [180.] , [196.], [142.]]

#x1 = tf.placeholder(tf.float32)
#x2 = tf.placeholder(tf.float32)
#x3 = tf.placeholder(tf.float32)
#Y = tf.placeholder(tf.float32)

print x_data
print y_data

"""
x data:
총 데이터가 몇개가 들어올지 모르니 None
그러나 변수가 3개인것은 정해졌으니 3로 fix
"""
X = tf.placeholder(tf.float32 , shape = [None,3])
Y = tf.placeholder(tf.float32, shape = [None,1])

#w1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
#w2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
#w3 = tf.Variable(tf.random_normal([1]), name = 'weight3')

W = tf.Variable(tf.random_normal([3,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')


#hopothesis 함수 선언
#hypothesis = x1*w1 + x2*w2 + x3*w3
hypothesis = tf.matmul(X,W) + b

#cost 함수 선언
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

#session 만들고 train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val , _ = sess.run([cost,hypothesis,train], feed_dict = {X : x_data , Y : y_data})
    if step %10 == 0:
        print(step, cost_val,hy_val)
