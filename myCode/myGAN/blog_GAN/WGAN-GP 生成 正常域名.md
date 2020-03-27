 # WGAN-GP 生成 正常域名



## 前言

之前一直想通过GAN来生成高相似度的正常域名，无奈多次尝试失败。

目前国内网上公开资源较少，而且是一个失败的案例。

国内唯一公开的demo：[使用生成对抗网络GAN生成DGA](http://webber.tech/posts/使用生成对抗网络(GAN)生成DGA/ )

国外也有人研究相关课题，倒是生成比较不错的效果，不过用的是python 2.7： [py_2.7_adversarial_DGA]( https://github.com/aaleotti-unimore/adversarial_DGA )

由于python 2.7 2020年开始已经停止维护，我用python 3.6 复刻该项目，结果没有达到预期

在此只能改变方向，发现WGAN-GP 能很好的实现我的需求。

## 原理

网上有很多优秀的阐述WGAN-GP的讲解，以下列出一些优秀的文章 :

#### WGAN 

[WGAN原理]( https://www.cnblogs.com/Allen-rg/p/10305125.html )

[令人拍案叫绝的WGAN](https://zhuanlan.zhihu.com/p/44169714)  

[WGAN的来龙去脉]( https://www.jianshu.com/p/f1462c489a63 )

[WGAN代码解读及实验总结](https://blog.csdn.net/CLOUD_J/article/details/94392474)

[WGAN的改进点和实操]( https://blog.csdn.net/li123128/article/details/97373899 )

[WGAN TensorFlow 代码]( https://zhuanlan.zhihu.com/p/25563732 )

上面的文章介绍了GAN的缺陷，和向WGAN的进化的历程。

WGAN 相对GAN的变化主要有以下几点：

- 判别器最后一层去掉sigmoid
- 生成器和判别器的loss不取log
- 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
- 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行

而且WGAN 引进了 Wasserstein距离 ， 由于它相对KL散度与JS散度具有优越的平滑特性，理论上可以解决梯度消失问题。接着通过数学变换将Wasserstein距离写成可求解的形式，利用一个参数数值范围受限的判别器神经网络来最大化这个形式，就可以近似Wasserstein距离。在此近似最优判别器下优化生成器使得Wasserstein距离缩小，就能有效拉近生成分布与真实分布。WGAN既解决了训练不稳定的问题，也提供了一个可靠的训练进程指标，而且该指标确实与生成样本的质量高度相关 。

#### WGAN-GP

[WGAN-GP-更加先进的Lipschitz限制手法](https://www.zhihu.com/question/52602529/answer/158727900)

[从GAN到WGAN再到WGAN-GP]( https://www.jianshu.com/p/e901908a1d93 )

 WGAN-GP 说到底就是对参数截断Lipschitz的改进。

#### 其他优秀的文章汇总：

[用各种GAN生成MNIST数字]( https://blog.csdn.net/songbinxu/article/details/85930769 )

[用各种GAN生成图片](  https://blog.csdn.net/Geoffrey_MT/article/details/81198504  )

#### 关于指标

关于指标的问题非常重要，当我们接触一门新的技术时，首先就是要入手相关demo进行尝试有个直观的了解，而人工智能方面的成功与否出了看产出的东西是否符合，更需要了解学术性或者更为准确性的公认指标来衡量该程序是否成功。

而令我苦恼很久的是，刚开始接触的WGAN时，虽然理论提到WGAN相对GAN有了明确可指导进程的指标，但是很多文章关于“该指标是什么”，还是“指标越大还是越小好”，都只是略略提过，本文将从公式原理和引用他人博客两方面经过佐证衡量WGAN-GP的成功指标取向。

先说结论：

WGAN-GP运行之后会产生一个判别器D的损失函数d_loss，d_loss为衡量WGAN-GP好坏的指标，越小越好，不看符号只看数值。

由WGAN的最终损失函数如下：

![image-20200205202853465](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205202853465.png)

而WGAN-GP 只是加了个惩罚项，公式如下

![image-20200205203039253](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205203039253.png)

 

自从WGAN/WGAN-GP添加了新指标 Wasserstein （经过一系列的演算用L表示真实分布与生成分布之间的Wasserstein距离）。观察公式发现，忽略掉正则项。d_loss(L(D))就是表示两个分布的距离，无论是正负，其绝对值越小越好。

博客方面：

[WGAN原理有提及d_loss与L的关系（相反数）](https://www.cnblogs.com/Allen-rg/p/10305125.html )

[代码最后分析了结果：WGAN的d_loss与g_loss走向]( https://posts.careerengine.us/p/58a2a00f732a570a31371cec )

[WGAN与WGAN-GP流程部分从公式入手分析d_loss与g_loss的走向]( https://zhuanlan.zhihu.com/p/66489938 )

[评论部分有人对loss走向发表见解]( https://blog.csdn.net/qq_20943513/article/details/73129308 )

## 代码实战部分

讲了一大通原理，终于到了实战部分

实现环境： gpu: Python3.5, Tensorflow-gpu:1.10.0

![image-20200205210430556](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205210430556.png)

加载Alexa正常域名数据，和各种参数定义

![image-20200205210830227](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205210830227.png)

定义D和G模型

![image-20200205210929343](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205210929343.png)

定义真实样本与噪音样本，注意真实样本这里只是用placeholder占位符先占位，还没有实质的张量tensor

[关于placeholder与 feed_dict]( https://zhuanlan.zhihu.com/p/25307881 )

![image-20200205211153525](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205211153525.png)

原WGAN的损失函数

![image-20200205211230729](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205211230729.png)

加入正则项，最终形成WGAN-GP公式

![image-20200205211323706](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205211323706.png)

优化参数，这里没有按照理论的来，用了Adam优化

![image-20200205211435859](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205211435859.png)

创建session开始训练，每训练一次生成器就训练10次判别器，

而且生成器和判别器各自训练时，另一个参数固定保持不变

一些想法：代码如何具体体现D和G对抗思想，博弈中提升的呢？核心在于上面两个session.run里面，他们迭代执行d_loss与g_loss的优化，拼命的使其降到最低，新一轮中D判别得越准，而G也生成的东西越逼真，为了降低loss，优化器又进行了新一轮的优化....以此类推，达到博弈中提升。

## 结果

经历30000轮的训练，d_loss越来越小，接近收敛

![image-20200205212558685](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205212558685.png)

生成了起码肉眼看起来类似人类起名习惯的域名

![image-20200205212841266](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205212841266.png)

github地址：

