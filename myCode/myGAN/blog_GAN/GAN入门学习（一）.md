



#### 定义真实/噪音样本生成函数，并且生成图片保存

![image-20191226112644002](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191226112644002.png)

#### 定义生成器discriminator

![image-20191226124100784](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191226124100784.png)

这里一下输入函数这里，有点复杂

在keras中，数据是以张量的形式表示的，不考虑动态特性，仅考虑shape的时候，可以把张量用类似矩阵的方式来理解。 

例如:

  [[1],[2],[3]] 这个张量的shape为（3,1）
  [[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]]这个张量的shape为（3,2,2）,
  [1,2,3,4]这个张量的shape为（4，）

括号里面的数字，从左到右分别是从该矩阵外到内的元素个数表示。例如第二个例子的（3,2,2），就是最外层3维数组时有3个元素，往内深入看2维数组时，一个2维数组有2个元素，再深入看1维数组时，里面有2个元素，所以为（3,2,2）.

而当为1维数组时，shape因为时元组，所以不能（4），要在后面加逗号，故（4，）

通过input_length和input_dim这两个参数，可以直接确定张量的shape。

常见的一种用法：只提供了input_dim=32，说明输入是一个32维的向量，相当于一个一阶、拥有32个元素的张量，它的shape就是（32，）。因此，input_shape=(32, )

而 Reshape 就是用来将输入shape转换为特定的shape，可以理解为重新定义张量形状。



#### 定义判别器generator

![image-20191226125537850](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191226125537850.png)



#### 连接两个模型

![image-20191226130720362](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191226130720362.png)

把trainable设置为手工更新，为了降低损失值，使得 generator每次训练被迫适应discriminator，也是这样的形式得到discriminator得反馈。



#### 初始化模型

![image-20191226161954338](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191226161954338.png)

#### 训练数据

![image-20191226162043287](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191226162043287.png)

将阶段性成果输出

![image-20191226162110706](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191226162110706.png)

#### 结果

原本服从均匀分布的噪音数据被生成了服从正态分布数据。

但是loss值梯度爆炸，与理论值相反增长。目前还不懂所以然







 参考连接：

https://blog.csdn.net/u010159842/article/details/78983841 

https://blog.csdn.net/yangdashi888/article/details/80452156 

https://blog.csdn.net/pmj110119/article/details/94739765 