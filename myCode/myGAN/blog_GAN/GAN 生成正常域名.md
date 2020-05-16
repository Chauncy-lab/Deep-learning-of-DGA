# GAN 生成正常域名

### 前言

最近接触gan对抗网络，想用GAN生成正常域名

目前国内网上公开资源较少，而且是一个失败的案例。

国内唯一公开的demo：[使用生成对抗网络GAN生成DGA](http://webber.tech/posts/使用生成对抗网络(GAN)生成DGA/ )

国外也有人研究相关课题，倒是生成比较不错的效果，不过用的是python 2.7： [py_2.7_adversarial_DGA]( https://github.com/aaleotti-unimore/adversarial_DGA )

由于python 2.7 2020年开始已经停止维护，我用python 3.6 复刻该项目，结果没有达到预期

但是生成的效果比上面的demo好很多，还是具有一定参考价值

### 代码部分

篇幅有限只贴重要部分代码

![image-20200205235346079](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205235346079.png)

定义 生成器

![image-20200205235407969](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205235407969.png)

定义判别器

![image-20200205235438244](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205235438244.png)

连接成对抗模型

接下来开始train

![image-20200205235647586](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205235647586.png)

先加载数据

![image-20200205235716554](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200205235716554.png)

用函数分别对D和G进行优化

![image-20200206002618332](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200206002618332.png)

生成正常样本

![image-20200206002638803](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200206002638803.png)

产生对应的假样本

![image-20200206002727995](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200206002727995.png)

真假样本交替训练

![image-20200206002750048](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200206002750048.png)

放进判别器中训练

![image-20200206002807789](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200206002807789.png)

训练完判别器，就固定判别器参数

![image-20200206002839417](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200206002839417.png)

制造噪音样本，并且标记为1，企图欺骗对抗模型

判别器D不变，只有生成器G为了降loss不断适应，从而达到提升G的目的（生成越来越接近D的样本）

### 结果

训练了200轮，生成的域名如图（二级域名）

![image-20200206004008498](C:\Users\xingchi\AppData\Roaming\Typora\typora-user-images\image-20200206004008498.png)

可看到，有些域名还算正常，另外一些存在大量重复，与人类起名习惯有一定差异。

但相对国内开源的那个demo更加接近人类的起域名习惯。

源码在github上：