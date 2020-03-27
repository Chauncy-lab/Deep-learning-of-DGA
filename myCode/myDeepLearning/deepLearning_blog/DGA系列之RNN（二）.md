继上一篇DGA检测代码涉及到的RNN，我们来挖一挖RNN（循环神经网络）

## 发展史
在了解这个算法之前，先了解它为何出现。说下个人理解，神经网络是基于感知机的扩展，而DNN可以理解为有很多隐藏层的神经网络。多层神经网络和深度神经网络DNN其实也是指的一个东西，DNN存在一些局限性，例如参数数量膨胀，局部最优，梯度消失和无法对时间序列上的变化进行建模。而RNN就是为了解决无法对时间序列上的变化进行建模而诞生的，在普通的全连接网络或者CNN中，每层神经元的信号只能向上一层传播，样本的处理在各个时刻独立（这种就是前馈神经网络），而在RNN中，神经元的输出可以在下一个时间戳直接作用到自身。（t+1）时刻网络的最终结果O(t+1)是该时刻输入和所有历史共同作用的结果，这就达到了对时间序列建模的目的。我们可以看做CNN是时间上的前馈网络（DNN）

## 原理
学过DNN的我们都知道，前馈神经网络的网络模型是这样的
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207142044607.png)
包括输入层，隐藏层，输出层。添加隐藏层也是在同一个维度上面添加的，而不同的DNN网络结构是独立的，也就是我们一开始所假设的，每个时刻都是独立的，互不关联的。

而RNN的网络结构则是，在横向隐藏层与隐藏层有链接，这个横向的我们可以理解为一种时间分布{0,1,2,3....t}，而且假设第t时刻发生的结果只与t-1时刻的结果有关系，所以就有如下结构图

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019120819533922.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
h是隐藏层，x是输入，y是输出，而隐藏层和隐藏层的联系有权重系数U，x和隐藏层的权重系数是v。
我们将其对比成计算图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208200338976.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
x是输入，h是隐层单元，o为输出，L为损失函数，y为训练集的标签。这些元素右上角带的t代表t时刻的状态，其中需要注意的是，因策单元h在t时刻的表现不仅由此刻的输入决定，还受t时刻之前时刻的影响。V、W、U是权值，同一类型的权连接权值相同
对于t时刻：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208200413931.png)
其中ϕ()为激活函数，一般来说会选择tanh函数，b为偏置。t时刻的隐藏层是等于（t-1时刻的隐藏层 *  权重比W）+（t时刻的输入的x * 权重比U) +偏执量B
t时刻的输出就更为简单：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208200740653.png)
最终模型的预测输出为：![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208200803526.png)
其中σ为激活函数，通常RNN用于分类，故这里一般用softmax函数。

上面是标准的RNN流程输出，但是如果实际上相关的信息和预测的词位置之间的间隔是非常小的情况下才有很好的效果，如果信息与预测点相隔很长那就效果很差，例如这个图的情况
![在这里插入图片描述](https://img-blog.csdnimg.cn/201912082011185.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
在这个间隔不断增大时，RNN 会丧失学习到连接如此远的信息的能力。
所以引入了LSTM，他就是为了解决这个问题而生的。

LSTM(Long Short Term )网络,是一种 RNN 特殊的类型，可以学习长期依赖信息，标准的RNN结构，中间会有个tanh 层
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208201333173.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
LSTM 同样是这样的结构，但是重复的模块拥有一个不同的结构。不同于 单一神经网络层，整体上除了h在随时间流动，细胞状态c也在随时间流动，细胞状态c就代表着长期记忆。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208201409972.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5OTM2NDM0,size_16,color_FFFFFF,t_70)
最终可以使模型达到学习长期依赖信息的目的。

这个原理太过于复杂，网上已经有很多优秀的文章。如果想要了解更多就看他们的就好
https://www.jianshu.com/p/9dc9f41f0b29
https://blog.csdn.net/zhaojc1995/article/details/80572098