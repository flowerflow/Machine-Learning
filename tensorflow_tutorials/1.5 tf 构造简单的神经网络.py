"""
构造一个3层的网络
输入层一个结点，隐层10个结点，输出层一个结点

输入层的维度是[n,1]
隐层的维度是  [1,10]
输出层的维度是[10,1]

so,
权值矩阵的维度是：
weight1=[1,10]
bais1=[10,1]

weight2=[10,1]
bais2=[1,1]
"""


import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    #我们给这一层加的权重是随机生成的
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #创建 【in_size * out_size】的矩阵，服从正态分部

    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    #创建偏移函数  【1*outsize】 +0.1
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    #这一步设置了内部函数式如何转化的，这里是 利用矩阵乘法 +  矩阵加法
    #也就是可以理解线性函数
    #
    #tf.matmul是矩阵乘法

    # input的矩阵【N * input__size】  *  【in_size * out_size】
    # 返回 N  *  outsize
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 构造一个数据集
x_data = np.linspace(-1,1,300)[:, np.newaxis]
#指定的间隔内返回均匀间隔的数字。即1 到 -1 间，300个数
#np.newaxis在【，】前面时，变为列扩展的二维数组 1*n
#np.newaxis在【，】后面时，变为行扩展的二维数组 n*1
#所以是 【300*1】
noise = np.random.normal(0, 0.05, x_data.shape)
#np.random.normal为高斯正态分布

#y_data = np.square(x_data) - 0.5 + noise
#y_data为 x**2 -0.5+ 噪声 noise
#模拟y=x**2
y_data = np.square(x_data)

# placeholder 占个位
xs = tf.placeholder(tf.float32, [None, 1]) #  N  * 1类型矩阵 ，N未知
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer 添加隐藏层（中间层

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 矩阵乘法  【N*1】  * 【1 * 10】 =N* 10


# add output layer，输出层
# 上一层的输出是这一层的输入
prediction = add_layer(l1, 10, 1, activation_function=None)
# N* 10 10*1 --》N*1

# the error between prediction and real data

#loss函数和使用梯度下降的方式来求解
#创建损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
#创建训练步骤
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12

if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
#初始化
sess = tf.Session()
sess.run(init)
#开始运行
for i in range(1000):
    # trainings
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        # 在带有placeholder的变量里面，每一次sess.run 都需要给一个feed_dict，这个不能省略啊！
        print("loss : ",sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
#现在开始进行模拟
xx_data = np.linspace(-1,1,10)[:, np.newaxis]
yy_data = np.square(xx_data)
print(xx_data)
print("XX : ",sess.run(prediction, feed_dict={xs: xx_data, ys: yy_data}))