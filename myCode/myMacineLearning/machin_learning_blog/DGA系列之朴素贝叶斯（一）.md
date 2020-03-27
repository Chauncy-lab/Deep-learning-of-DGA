

# DGA系列之朴素贝叶斯（一）

​	这篇博客是对书中《web安全机器学习入门》DGA域名检测朴素贝叶斯的代码进行复现与解释。

​	**实验步骤如下**

	1. 获取样本数据
	2. 提取特征
	3. 将样本划分为训练集和测试集
	4. 使用朴素贝叶斯算法在训练集上训练，获得模型数据
	5. 使用模型数据在测试集上进行测试
	6. 验证朴素贝叶斯算法的结果
#### 获取样本数据

![image-20191201181413410](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191201181413410.png)

​	加载正常样本和dga黑样本

#### 提取特征与划分数据集

​	（一）提取统计特征

![image-20191201181728196](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191201181728196.png)

​	以每个域名元音字母个数，不重复字符个数，数字个数三个特征来标记一个域名，将其向量化

![image-20191201181851183](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191201181851183.png)

​	将其特征化之后，分别将x（样本集）,y（标签集）切割划分为训练集和测试集

​	（二）提取2-Gram

![image-20191201183452131](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191201183452131.png)

#### 训练、预测和验证朴素贝叶斯

![image-20191201182303529](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191201182303529.png)

对结果进行输出，上面的是基于统计特征的，下面的2-Gram也

![image-20191202094152720](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191202094152720.png)

#### 验证结果

​	（一）统计特征的朴素贝叶斯

![image-20191201183829557](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191201183829557.png)

​	（二）2-Gram特征的朴素贝叶斯

![image-20191201183703519](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191201183703519.png)



​	源码已上传到GitHub中，[nb_dga.py](https://github.com/Xandra-chan/Deep-learning-of-DGA/blob/master/code/deep_learning/nb_dga.py) 

