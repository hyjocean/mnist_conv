# mnist_conv
# 1.sess
 张量需要用sess.run(Tensor)来得到具体的值
 需要加sess = tf.InteractiveSession()
 使用sess.session()的时候采用
 with sess.session() as sess 可以防止会话资源泄露

# 2.tf.matmul是矩阵的乘法
  tf.multiply是矩阵的点乘

# 3.tf.nn.softmax()
  tf.nn  nn 是神经网络的缩写。
  softmax是其中一种操作，计算softmax的激活
  类似的还有tf.nn.conv2d/tf.nn.relu 应用relu函数等等

# 4.tf.truncated_normal
  tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定。这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。

# 5.tf.Variabl
   在TensorFlow的世界里，变量的定义和初始化是分开的，所有关于图变量的赋值和计算都要通过tf.Session的run来进行。想要将所有图变量进行集体初始化时应该用tf.global_variables_initializer

# 6.tf.reshape
  将tensor转换为参数shape的形式
  形状发生变化的原则时数组元素的个数是不能发生改变的，否则出错
  -1的应用：在不知道填什么数字的情况下，填写-1，由python自己计算出这个值，但只能出现一个-1  
  >>>d = a.reshape((2,4))
  >>>d
  array([[1, 2, 3, 4],
       [5, 6, 7, 8]])

  >>>f = a.reshape((2,2,2))
  >>>f
     array([[[1, 2],
             [3, 4]],
            [[5, 6],
             [7, 8]]])

# 7.tf.argmax
  tensorflow调用np中的np.argmax
  作用是，返回最大值所在下标。有时需注意轴的axis的问题
  tf.argmax(y_,1)是在x轴上进行的比较

# 8.tf.equal
  tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
>>>import tensorflow as tf
>>>import numpy as np
    A = [[1,3,4,5,6]]
    B = [[1,3,4,3,2]]
    with tf.Session() as sess:
    print(sess.run(tf.equal(A, B)))
output: [[ True  True  True False False]]

# 9.accuracy.eval
  等同于sess.run()
