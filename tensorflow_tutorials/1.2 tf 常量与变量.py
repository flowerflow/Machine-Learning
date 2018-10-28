import tensorflow as tf 

## 定义
#定义一个变量，可以理解为 var是在代码的命名，而myvar是tensorflow变量空间里定义的名字
#即  你是李狗蛋，而你的外国名是jack
var =tf.Variable(0,name="myvar")

#定义一个常量 =1
con_var=tf.constant(1)

#定义一个加法关系，new_var=var+con_var
new_var=tf.add(var,con_var)

## 开始计算

#初始化，在初始化之前是变量是没有值的（其实是有概念值，但是没有被实例化，就好比我说要你一个苹果，但是我没种出来
init =tf.global_variables_initializer()

#这里变量还是没有被激活，需要再在 sess 里, sess.run(init) , 激活 init 这一步.，
sess=tf.Session()

#计算
sess.run(init)

#输出
print ('var : ',sess.run(var))
print ('con_var : ',sess.run(con_var))
print ('new_var : ',sess.run(new_var))

# 关闭会话
sess.close()
"""
# 另一种写法
with tf.Session() as sess:
    sess.run(init)
    print ('var : ',sess.run(var))
    print ('con_var : ',sess.run(con_var))
    print ('new_var : ',sess.run(new_var))
"""