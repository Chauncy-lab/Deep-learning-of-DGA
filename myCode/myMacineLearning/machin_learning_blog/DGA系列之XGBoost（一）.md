这篇博客是对书中《web安全深度学习入门》DGA域名检测XGBoost的代码进行复现与解释。代码大部分与之前我发布的朴素贝叶斯算法的代码差不多。

**实验步骤如下**：

		1. 获取样本数据
	    2. 提取特征
	    3. 将样本划分为训练集和测试集
	    4. 使用XGBoost算法在训练集上训练，获得模型数据
	    5. 使用模型数据在测试集上进行测试
	    6. 验证XGBoost算法的结果

#### 获取样本数据
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191203135057392.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
老规矩，加载正常样本和黑样本。

#### 提取特征与划分数据集
（一）提取统计特征
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191203140137212.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
以每个域名元音字母个数，不重复字符个数，数字个数三个特征来标记一个域名，将其向量化
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191203140323944.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
这个代码和叶贝斯的一模一样，看不懂的可以看我的博客[DGA之叶贝斯（一）](https://blog.csdn.net/qq_39936434/article/details/103343767)

（二）提取2-Gram
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019120314121264.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
#### 训练、预测和验证XGBoost
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191203141305305.png)
#### 验证结果
(一) 统计特征的XGBoost
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191203141451139.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191203141704601.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)


（二）2-Gram特征的XGBoost
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191203141731653.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191203141606524.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
源代码在GitHub上：[xb_dga.py](https://github.com/Xandra-chan/Deep-learning-of-DGA/blob/master/code/deep_learning/xb_dga.py)