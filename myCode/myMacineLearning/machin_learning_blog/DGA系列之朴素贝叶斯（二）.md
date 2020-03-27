这篇博客是《web安全机器学习入门》DGA域名检测朴素贝叶斯的代码进行复现与解释。

## 实验步骤如下
1.数据搜集和数据清洗
2.特征化
3.训练样本
4.效果验证


#### 数据搜集和数据清洗
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202105152923.png)
	返回如下结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202105226298.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
 load_dga
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202105304783.png)

#### 特征化、训练与验证
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202105732916.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
使用三折交叉验证法，输出结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202112005312.png)
命中率还不错率还不错

想深入了解三折交叉验证法得话，看我另一篇[Blog](https://blog.csdn.net/qq_39936434/article/details/103335072)



其中，对某些代码与函数解释

初始化变量y1,y2,y3 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202105331882.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019120210535346.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202105408863.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
concatenate 转换成数组
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202105432441.png)
##### 核心处理特征函数
countVectorizer 是用来处理N-Gram特征的函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202110847568.png)
countVectorizer参数介绍：每2个切割，单词读取错误忽略，正则匹配所有字符，频数起码出现1次
然后，fit_transform训练数据

我们用简单的数据测试下，看返回结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019120211101641.png)
用countVectorizer切割
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202110847568.png)
分割出来的词典
print(cv.get_feature_names())
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202111136232.png)
无序词典，并且带有下标
print(cv.vocabulary_)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019120211123168.png)
输出训练后的稀疏矩阵，print(x)
参数为 ：data列表下标，无序词典下标，该词在data出现的频数
eg：“为了”在无序词典的下标为0，而且属于data列表的0下标，在data列表里出现了两次，所以为（0，0）2，所以定位了一个词的位置和频数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191202111325597.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
转为array，print(x.toarray())
x的每个定位都可以在坐标中找到，例如：“为了”，
他的是（0，0） 2，则对应矩阵中第一行第一列的值2，
其他依此类推
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019120211140226.png)

本博客对应代码已上传至GitHub：[nb_dga.py](https://github.com/Xandra-chan/Deep-learning-of-DGA/blob/master/code/machine_learning/nb_dga.py)