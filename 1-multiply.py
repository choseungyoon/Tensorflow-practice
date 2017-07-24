#-*- coding: utf-8 -*-

import tensorflow as tf
#placeholder : 프로그램 실행 중에 값을 변경 할 수 있는 symbolic variable

a = tf.placeholder("float")
b = tf.placeholder("float")

"""
tensorflow에서 제공하는 math함수
관련함수 참조 (https://www.tensorflow.org/api_guides/python/math_ops)

행렬 연산에 관한 함수도 제공
관련함수 참조 (https://www.tensorflow.org/api_guides/python/math_ops#Matrix_Math_Functions)
"""
y = tf.mul(a,b)


"""
Session
심볼릭 표현으로 된 수식을 계산하기 위해 세션 생성

Session()함수를 통해 세션을 생성함으로써 프로그램이 텐서플로우 라이브러리와 상호작용하게 됨
즉, 세션을 생성하여 run()메소드를 호출할 때 비로소 심볼릭 코드가 실행됨

"""
sess = tf.Session()

print sess.run(y, feed_dict = {a:3 , b:3})

"""
텐서플로우 프로그램의 일반적 구조
: 알고리즘을 먼저 기술하고 세션을 생성하여 연산을 실행
: 연산과 데이터에 대한 모든 정보는 그래프 구조안에 저장됨
: 그래프 구조는 수학 계산을 표현
: 노드는 수학 연산을 나타내고 데이터 입력과 출력의 위치를 나타내거나 저장된 변수를 읽거나 쓴다.
: 엣지는 입력 값과 출력 값으로 연결된 노드 사이의 관계를 표현하고
  그와 동시에 텐서플로우의 기본 데이터 구조인 텐서를 운반
"""


"""

"""
