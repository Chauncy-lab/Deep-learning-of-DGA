继上一篇博客，简单使用了XGBoost对DGA进行检测之后，这一篇博客试图去更深一层，在数学原理和代码层面了解XGBoost算法
## 基本概念
XGBoost全名叫（eXtreme Gradient Boosting）极端梯度提升，经常被用在一些比赛中，其效果显著。它是大规模并行boosted tree的工具，它是目前最快最好的开源boosted tree工具包。XGBoost 所应用的算法就是 GBDT（gradient boosting decision tree）的改进，既可以用于分类也可以用于回归问题中。

## 集成思想
一开始人们用决策树做分类的时候，算法往往是基于一棵树的决策，而得到一个得分，而这个得分不够精确。这也是叫做弱分类器
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205101049763.png)
后来把这些弱分类器集成，将他们每个人得到分数全部累加起来，就是集成思想
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205101325860.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
## XGBoost基本原理
当我们在计算每个叶子节点的权值（预测值）的时候：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205101736333.png)
w是其叶子的得分，x是叶子节点的下标

目标函数：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205101909829.png)
用来预测真实值与预测值的均方差，该值越小越好

再对比上面一节的图片，当我们在基于一颗决策树（分类器）上再加入一颗决策树的时候，我们希望是我们需要的叶子节点的得分是通过累加后越来越接近我们的期望值，所以说XGBoost是一个提升的过程。


因为t棵树的值，等于t-1得出的值与t得出的值累加，所以有以下推导
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205103143603.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
其中，每当插入一棵树的时候，为了避免过拟合，还需要加入惩罚函数。计算方式如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205103514647.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
最终我们的整体表达式为
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205103931173.png)
每次往模型中加入一棵树，其损失函数便会发生变化。另外在加入第t棵树时，则前面第t-1棵树已经训练完成，此时前面t-1棵树的正则项和训练误差都成已知常数项

如果损失函数采用均方误差时，其目标损失函数变为
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205105002772.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
残差部分（圈出来的）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205105155878.png)
每次加入数的时候，都需要是残差更小，效果就越好
最终化成这样：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019120510553122.png)

## 目标函数的求解
需要用到泰勒展开式：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019120510560555.png)
这里我们用泰勒展开式来近似原来的目标函数，将看作。则原目标函数可以写成：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205105625317.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019120510572510.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205105737619.png)
同时对于第t棵树时
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205105809229.png)为常数。同时去除所有常数项。故目标损失函数可以写成：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205105916143.png)
而复杂度等于
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205110043750.png)
同时我们将目标函数全部转换成在第t棵树叶子节点的形式
所以对
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205110137815.png)
可以看做是每个样本在第t棵树的叶子节点得分值相关函数的结果之和，所以我们也能从第t棵树的叶子节点上来表示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205110154236.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
其中为第t棵树中总叶子节点的个数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205110315372.png)
表示在第个叶子节点上的样本，Wj 第个 j 叶子节点的得分值。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205110416288.png)
则：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205110634995.png)
希望目标函数最小，所以令其偏导等于0，再带回目标函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205110900968.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
## 实例讲解
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205111004554.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
G1，H1都是可以根据上面的公式求出来，其他同理。

那我们应该怎么应用呢？
其实我们上班得出来的公式是一个得分标准的评分，我们还有一部分就是基于某个标准来判断左右树的切割点，每一个切割点都要遍历一下选取最优的切割点，而判断的公式如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191205111743452.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
分出来的左右数评分值（使用得出来的目标函数计算出来的）-分出来的左右数评分值（使用得出来的目标函数计算出来的）-每次加入树的代价

这里注意，因为目标函数前面带有负号，所以实际计算出来的式子（如上图所示）是（分出来的左右数评分值-分出来的左右数评分值）。

得出了的Gain（增益值），哪个增益最大，就选哪种方法切割

到此，XGBoost的算法完成了自己的使命。

下一篇会将XGBoost的代码部分。

参考连接：
https://www.cnblogs.com/zongfa/p/9324684.html
https://blog.csdn.net/huibeng7187/article/details/77588001
bilibi视频：https://www.bilibili.com/video/av26088803?p=4