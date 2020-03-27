这篇博客是对书中《web安全深度学习实战》DGA域名检测RNN的代码进行复现与解释。

## 实验步骤
1. 获取样本数据
2. 提取特征
3. 将样本划分为训练集和测试集
4. 使用RNN算法在训练集上训练，获得模型数据
5. 使用模型数据在测试集上进行测试
6. 验证RNN算法的结果

## 获取样本数据

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208191644306.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
老规矩，加载正常样本和黑样本。

## 提取特征与划分数据集
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208191720837.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
## 训练、预测和验证RNN
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208191915970.png)
将提取出来的特征数据二值化，并定义长度为64，不够长度就补0

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019120819201828.png)
建立网络模型，先定义传入数据形状，64维，然后通过embedding再嵌入，变成64x64，并且也将张量传入lstm。最后连接全层网络，定义损失函数等。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208192309796.png)
训练数据

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208192327534.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
验证数据，最后输出报告
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208192426239.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
吐槽一句，这里我用我的8核心cpu跑这个代码，直接爆炸......
后来感谢有个大佬借了台32核的服务器跑了一个钟才跑完>-<，结果不易啊

代码上传到个人github：[rnn_dga.py](https://github.com/Xandra-chan/Deep-learning-of-DGA/blob/master/code/deep_learning/rnn_dga.py)