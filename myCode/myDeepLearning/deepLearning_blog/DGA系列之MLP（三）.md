继上一篇介绍完MLP的原理之后，这一篇进入代码实战部分。使用tensoflow基于mnist识别验证码，体现MLP思想。

## 代码实战
我们将通过以下步骤：
	
	1.直接使用softmax函数，看看只使用线性函数处理效果如何
	2.加一层隐藏层，看处理效果如何
	3.多个隐藏层，看处理效果如何


#### 直接使用softmax函数
MSIST是一个入门级的计算机视觉数据集，它包含各种手写数字图片（0-9），也包含了每一张图片对应的标签（0-9），训练数据(50000, 784)，测试数据(10000, 784)，

首先加载数据，并将他们分为训练，校验，测试数据集
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207155226243.png)
使用one-hot处理标签集，除了某一位的数字是 1 以外其余各维度数字都是 0。所以在此教程中，数字 n 将表示成一个只有在第 n 维度（从 0 开始）数字为 1 的 10 维向量，训练集和测试集都要处理
![在这里插入图片描述](https://img-blog.csdnimg.cn/201912071613417.png)
如果对one-hot方法和源码中784或者10这些数字很迷惑的，可以看这篇博客https://blog.csdn.net/weixin_41847115/article/details/84890654

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207161554448.png)
placeholder给每个参数设置占位符

```python
y = tf.nn.softmax(tf.matmul(x,W) + b)
```
定义整个系统的操作函数

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207161910592.png)
定义衰减函数，这里衰减函数用交叉熵来衡量，通过梯度下降算法以0.01的学习率最小化交叉熵
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207162043479.png)
初始化tensorflow
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207163815199.png)
每次训练的数据是batch_size=100

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207164811855.png)
验证结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207164949433.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
命中率90%左右

#### 加一层隐藏层

前面的提取数据，划分数据和上面的差不多
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207171257998.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207171441115.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
重点在个模块，中间加个维度为300的隐藏层，并且通过relu函数将其激活，dropout是为了防止数据过拟合的正则项

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207171730781.png)
定义衰减函数，这次的学习率为0.3

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207171853770.png)
训练数据，将keep_prob设置为0.75

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207171938177.png)
最后验证结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207172017462.png)
准确率达到95.55%，比第一个方法高了
#### 多层隐藏层
和第一层隐藏层没什么不同，加了几个隐藏层而已
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207172259374.png)
对应的权重系数w，偏重截距b
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207172335104.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
算出来对应的隐藏层
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207172400798.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
最后打印的出来的准确率为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207172522941.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
效果比之前的两个还要高，达到比较优的命中率97%

分别对应代码在github：
[myMlp1.py](https://github.com/Xandra-chan/Deep-learning-of-DGA/blob/master/code/algorithm_principle/myXgboost1.py)
[myMlp2.py](https://github.com/Xandra-chan/Deep-learning-of-DGA/blob/master/code/algorithm_principle/myXgboost2.py)
[myMlp3.py](https://github.com/Xandra-chan/Deep-learning-of-DGA/blob/master/code/algorithm_principle/myXGBoost3.py)

参考连接：https://blog.csdn.net/weixin_41819529/article/details/82973862