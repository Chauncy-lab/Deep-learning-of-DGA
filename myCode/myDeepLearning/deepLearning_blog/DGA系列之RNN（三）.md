继上一篇介绍完RNN算法之后，我们现在来进行代码实战

## 代码实战
这次同样是验证码识别，不同的是上次DNN是tensorflow实现的，而这次是用tflearn框架实验的。

##### 实验步骤
	1.DNN的tflearn实现mnist验证码识别
	2.RNN的tflearn实现mnist验证码识别

##### 加载数据
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208225008103.png)
##### DNN
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208225529564.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
DNN算法，这个维度维度是784，这是将所有数据放在一个维度里。
最后softmax分类作为输出层。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208231206149.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
结果输出99%，效果相当不错

##### RNN
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208230718615.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
将784分成28x28，化为连续28个维度为28得特征向量序列

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208231732856.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
结果输出95%左右

代码上传到个人github：[myRnn.py](https://github.com/Xandra-chan/Deep-learning-of-DGA/blob/master/code/machine_learning/myRnn.py)