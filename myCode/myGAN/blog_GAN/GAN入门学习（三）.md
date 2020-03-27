上一篇blog实验了基于MNIST数据集生成数据，虽然肉眼可以分辨不出来了，但是d_loss与g_loss不是很理想。这次尝试对其进行优化

#### 加载数据

![image-20191230233840998](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230233840998.png)

#### 生成器函数

![image-20191230233932887](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230233932887.png)

这里的与上一篇的生成函数中的激活函数不同，这次用激活函数用relu，输出函数用tanh

#### 判别器函数

![image-20191230234324465](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230234324465.png)

 判别器中使用了LeakyRelu激活函数，并且使用步长为2的卷积代替了池化。 

#### 函数初始化

![image-20191230234456756](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230234456756.png)

对判别器和生成器的函数初始化，并且建立对抗模型combined

#### 训练

![image-20191230234854867](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230234854867.png)

这里同样是分成两步，第一步训练判别器d，第二步训练生成器g。

与上一次训练不同的是，训练判别器d时，之前是真假标签一起送进去训练

这次是分别真样本训练一次，假样本训练一次

这里的一个epoch实际上是一个batch，而不是遍历整个数据集的意思，也没有采用shuffle数据集的方式，而是通过每个epoch随机选取训练数据的方式来增加随机性。 

![image-20191230235356551](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230235356551.png)

将d_loss与g_loss的结果描绘成图

#### 结果

第一轮训练结果

![image-20191230235518931](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230235518931.png)

第1000轮训练结果

![image-20191230235614851](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230235614851.png)

第3800轮训练结果

![image-20191230235641374](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230235641374.png)

清晰度和上一篇差不多，但是数字的多样化要比上一篇的代码好

再看统计图

![image-20191230235801475](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230235801475.png)

在250轮左右已经 d_loss和g_loss就比较稳定了



参考连接： https://blog.csdn.net/yiqisetian/article/details/99678881 