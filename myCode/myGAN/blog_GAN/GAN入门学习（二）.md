自上一次用Gan简单生成相似的分布数据之后，这次使用深度卷积GAN（DCGAN）实现图片的识别与生成。本次实验是基于MNIST数据集生成数据

#### 加载MNIST数据集

![image-20191230213907736](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230213907736.png)

#### 生成器函数

![image-20191230213607599](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230213607599.png)

#### 判别器函数

![image-20191230213706701](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230213706701.png)

#### 将生成器和判别器连接（对抗模型）

![image-20191230213832829](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230213832829.png)



#### 训练数据

![image-20191230222923712](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230222923712.png)

加载数据，并且切割好数据

![image-20191230223008917](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230223008917.png)

初始化生成器和判别器，定义优化算法与损失等参数

训练与上一篇博客生成相同分布的步骤一样，大致分两步：

![image-20191230223838255](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230223838255.png)

第一步（训练判别器d）：

生成均匀分布噪音样本noise，并且从X_train中抽取真实样本，

生成器g以原本的定义好的函数（规则）去从noise噪音样本predict（筛选）出假样本

每到100轮就输出成品（按照生成器规则参数生成的样本）

合并真实与假的样本，并创建对应的标签组

训练判别器d，并且输出损失值d_loss

![image-20191230225009433](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230225009433.png)

第二步（训练生成器g）:

又生成均匀分布的样本

这次将判别器d固定参数不变

直接从这些样本中训练生成器g（整个对抗模型中d不变，g为了降低loss不断变化参数）

训练完，开启d参数可变

![image-20191230225515860](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230225515860.png)

保存本次训练的权重参数，方便下次直接生成

![image-20191230225602314](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230225602314.png)

单独生成函数

![image-20191230225703647](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230225703647.png)

最后将记录d_loss与g_loss的走向，描绘成图

整个训练过程过程中，d和g的博弈中成长究竟是如体现的呢？我说下我的理解。

第一次训练中，第一步将真假数据放入d训练，训练了一定精度使得d有基础的识别能力，g一开始也有自己的生成能力。然后到第二步，为了体现模型的泛化能力，又生成一堆均匀分布的数据，这次送进对抗模型中（d的参数已经锁定，其实就是训练g），为了降低整个模型的loss值，g会将自己选出来最真的假样本并且标记为1去欺骗第一步训练出来的d，但是由于第一轮g没经验，生成的样本与d参数觉得为真的（只有认为是真的，loss值才会降下来）相差甚远，所以d不同意这是真的，第一轮的整体loss值（g_loss）无法降下来很多，只降下来一丢丢，g只是向d认为真的标准（参数）靠近了一点点，g在博弈中成长了一丢丢

上面是第一轮的情况，然后第二轮g带着这增长一点点的骗术，重复着上一轮的步骤。生成了g所认为真的假样本，与真实样本合并并且打上标签，拿去训练d，d见识到了更加真的假样本与真样本，从而d的识别能力得到提高，在博弈中d也成长了一丢丢。

如此循环着，经过一定次数的训练，理论上g的骗术越来越强，生成的样本越来越真，g_loss越来越低，d越来越难识别，d的loss越来越高。最理想的状态就是达到一个平衡值，g生成的就是真样本，d根本就无法判别该样本是真还是假。

#### 结果

这是第1轮g生成的图片

![image-20191230231921475](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230231921475.png)

第10轮已经模模糊糊看处数字

![image-20191230232107448](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230232107448.png)

第30轮

![image-20191230232133644](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230232133644.png)

第150轮

![image-20191230232250686](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230232250686.png)



看看统计图

![image-20191230232327516](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20191230232327516.png)



横坐标是训练次数，我这里定义的是训练150轮，每轮训练468次

看看走势图，训练次数过多，会导致梯度爆炸。大概在15000次左右g_loss的值走向最低，而d_loss上升到最高。

15000/468=32次左右，得到的效果最好

参考连接： https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650731540&idx=1&sn=193457603fe11b89f3d298ac1799b9fd&chksm=871b306ab06cb97c502af9552657b8e73f1f5286bc4cc71b021f64604fd53dae3f026bc9ac69&mpshare=1&scene=23&srcid=12158jtIOgd23oyZM7eTgUrZ#rd 

