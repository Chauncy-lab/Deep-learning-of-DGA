

# DGA系列之朴素贝叶斯（一）

	这篇博客是对书中《web安全机器学习入门》DGA域名检测朴素贝叶斯的代码进行复现与解释。
	**实验步骤如下**：
	
		1. 获取样本数据
	    2. 提取特征
	    3. 将样本划分为训练集和测试集
	    4. 使用朴素贝叶斯算法在训练集上训练，获得模型数据
	    5. 使用模型数据在测试集上进行测试
	    6. 验证朴素贝叶斯算法的结果

#### 获取样本数据
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202095433705.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
	加载正常样本和dga黑样本

#### 提取特征与划分数据集

	（一）提取统计特征
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202095523391.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)


	以每个域名元音字母个数，不重复字符个数，数字个数三个特征来标记一个域名，将其向量化
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202095549563.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
	将其特征化之后，分别将x（样本集）,y（标签集）切割划分为训练集和测试集

	（二）提取2-Gram
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202095803484.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
#### 训练、预测和验证朴素贝叶斯

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202095607182.png)


#### 验证结果

	（一）统计特征的朴素贝叶斯
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191203141927343.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202095944724.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)


	（二）2-Gram特征的朴素贝叶斯

对结果进行输出，上面的是基于统计特征的，下面的2-Gram也是同理
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202095930276.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202100003941.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
	源码已上传到GitHub中，[nb_dga.py](https://github.com/Xandra-chan/Deep-learning-of-DGA/blob/master/code/deep_learning/nb_dga.py) 