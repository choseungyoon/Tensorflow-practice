#-*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

# Y = Wx + b model
"""
입력데이터 x_data를 이용해 출력 데이터 y_data를 만들수 있는 최적의파라메타 w와 b를 찾도록
텐서플로우 코드를 만드는 것이 목적

y_data = W * x_data + b 와 같은 직선이 나옴

"""
# Variable 메소드를 호출하면 텐서플로우 내부의 그래프 데이터 구조에 만들어질 하나의 변수를 정의했다고 이해하면 됨
# tensorflow가 사용하는 variable /  tensorflow가 자체적으로 변경하는 variable

#x_train = [1,2,3]
#y_train = [1,2,3]
X = tf.placeholder(tf.float32,shape = [None])
Y = tf.placeholder(tf.float32,shape = [None])

#random_normal([1]) 여기서 1은 shape =1 을 표현함
W = tf.Variable(tf.random_normal([1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')

#hypothesis = x_train *W + b
hypothesis = X *W + b

#reduce_mean 평균
#cost = tf.reduce_mean(tf.square(hypothesis - y_train))
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

###### 그래프 구현 완료
#session 생성
sess = tf.Session()

#실행하기 전에 먼저 global_variables_initializer 해서 변수 초기화 해줘야함
#Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(2001):
    #sess.run(train)
    cost_val , W_val, b_val, _ = sess.run([cost, W,b,train], feed_dict= {X : [1,2,3] , Y : [1,2,3]})
    if step % 20 ==0:
        #print(step, sess.run(cost), sess.run(W), sess.run(b))
        print(step,cost_val,W_val,b_val)
