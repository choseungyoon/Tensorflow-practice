#-*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

xy = np.loadtxt('data-01-test-score.csv',delimiter = ',', dtype = np.float32)
#마지막 한개 빼고 가져옴 [0:-1]
x_data = xy[:, 0 : -1]
y_data = xy[:, [-1]]
