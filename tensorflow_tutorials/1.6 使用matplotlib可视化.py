"""
构造一个3层的网络
输入层一个结点，隐层3个结点，输出层一个结点

输入层的维度是[n,1]
隐层的维度是  [1,10]
输出层的维度是[10,1]

so,
权值矩阵的维度是：
weight1=[1,10]
bais1=[10,1]

weight2=[10,1]
bais2=[1,1]

网络的结构和1.5的内容是一样的，只不过是这一次把每次的训练结构可视化show出来了
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#图像模块

#添加层函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 创建 【in_size * out_size】的矩阵，服从正态分部
    # 我们给这一层加的权重是随机生成的
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    #这一步设置了内部函数式如何转化的，这里是 利用矩阵乘法 +  矩阵加法
    #也就是可以理解线性函数
    #
    #tf.matmul是矩阵乘法
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
#产生-1到1之间均匀分布的300个数
noise = np.random.normal(0, 0.05, x_data.shape)
#噪声

y_data = np.square(x_data) - 0.5
#y =x**2-0.5 +noise
plt.scatter(x_data, y_data)
plt.show()
#
# define placeholder for inputs to network
#设置占位符
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
#x_data相当于输入层
# add hidden layer
#添加隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
#输出预测层
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
#损失函数,reduction_indices参数，表示函数的处理维度。
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#训练函数，梯度下降，学习速度为0.1
# important step
sess = tf.Session()
#初始
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)
#生成所有变量
#print(x_data)
#print("\n")
#print(y_data)
# plot the real data
plt.scatter(x_data,y_data) #绘制散点图
prediction_value=None  #预测矩阵
for i in range(1000):
    # training
    #训练
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 500 == 0:
        global prediction_value
        prediction_value= sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        plt.plot(x_data, prediction_value)

plt.show()
print(type(prediction_value))
print(prediction_value.shape)
for i in range(300):
    print(y_data[i,:],"  ",prediction_value[i,:])