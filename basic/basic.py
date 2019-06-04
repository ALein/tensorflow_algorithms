# D:\anaconda\envs\tensorflow\python
# _*_ coding:utf-8 _*_
"""
# Time: 2019/6/4  17:08
# Author: AL_Lein
"""
import tensorflow as tf
import numpy as np
#tensorflow中大部分数据是float32

#create real data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###

#定义变量
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

#如何计算预测值
y = Weights * x_data + biases

# loss function
loss = tf.reduce_mean(tf.square(y-y_data))

#梯度下降优化器，定义learning rate
optimizer = tf.train.GradientDescentOptimizer(0.5)

#训练目标是loss最小化
train = optimizer.minimize(loss)

#初始化变量，即初始化 Weights 和 biases
init = tf.global_variables_initializer()

#创建session，进行参数初始化
sess = tf.Session()
sess.run(init)

#开始训练200步，每隔20步输出一下两个参数
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))
### create tensorflow structure end ###



"""Tensorflow的Session,对话控制模块，可以用sesison.run来运行框架中的某一个
点的功能"""

import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1,matrix2)

sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()


with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

"""TF variable"""

import tensorflow as tf

#定义变量，给定初始值和name
state = tf.Variable(0,name="counter")
#counter:0
print(state.name)

one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value)

#这里只是定义，必须用session.run来执行
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


"""TF placeholder

placeholder 是 Tensorflow 中的占位符，暂时储存变量.
Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(), 然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).
"""

import tensorflow as tf

input1 = tf.placeholder(dtype=tf.float32)
input2 = tf.placeholder(dtype=tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[3.],input2:[5]}))

"""激励函数

激励函数运行时激活神经网络中某一部分神经元，将激活信息向后传入下一层的神经系统。激励函数的实质是非线性方程。
 Tensorflow 的神经网络 里面处理较为复杂的问题时都会需要运用激励函数 activation function 
 """