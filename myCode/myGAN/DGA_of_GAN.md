# DGA of  GAN

### 一、关于GAN的基础学习

对GAN的基础原理进行了解，并用GAN做一些很基础的事：

实验生成简单的高斯分布、Mnist手写图片等(目前收录在reinforcement_learning_GAN中)

相关讲解的blog（目前收录在blog_GAN中）

### 二、关于GAN在dga的应用

目前探究的应用方向有两个：

1. 使用GAN生成更加相似的正常域名Alexa，企图欺骗模型
2. 使用GAN生成更加相似的dga域名，预测DGA变种样本

### 二、目前dga的GAN研究的进度

#### （1）生成正常域名

###### Adversarially-Tuned Domain Generation and Detection(目前收录在dga2_GAN中)

[最著名的论文:Adversarially-Tuned Domain Generation and Detection]( https://arxiv.org/abs/1610.01969 )

[Adversarially-Tuned Domain Generation and Detection 的阅读笔记]( https://blog.csdn.net/zko1021/article/details/85269554 )

[Adversarially-Tuned Domain Generation and Detection 的github复现代码]( https://github.com/roreagan/DeepDGA )

###### 国内唯一公开的demo(目前收录在dga_GAN中)

[使用生成对抗网络GAN生成DGA](http://webber.tech/posts/使用生成对抗网络(GAN)生成DGA/ )

[对应的github地址]( https://github.com/bts-webber/GAN_for_DGA )

###### 国外的实验

[py_2.7_adversarial_DGA]( https://github.com/aaleotti-unimore/adversarial_DGA )(目前收录在dga4_GAN中，原项目为py2.7，本文用3.6复现，讲解blog放在blog_GAN中)

[同上的项目研究_detect_DGA]( https://github.com/aaleotti-unimore/detect_DGA ) （部分数据来源于detect_DGA）

[WGAN生成DGA]( https://github.com/Jared-Lee/GANDGA ) (目前收录在dga3_GAN中，讲解blog放在blog_GAN中）

#### （2）生成相似的DGA域名

[基于对抗模型的恶意域名检测方法的研究与实现_袁辰](D:\document\school\毕业论文\参考文献\文献库\未读文献\DGA)（包括部分参数源码，和伪代码，复现有待实践）

#### （3）其他涉及知识

###### 各种变种dga的生成样本

[ Domain Generation Algorithms (DGAs) ]( https://github.com/baderj/domain_generation_algorithms )

### 三、下一步研究方向

1. 目前用到最多的噪声为服从正态分布或均匀分布，接下来研究更多不同类型噪声分布对生成样本质量的影响。

2. 将生成的对抗样本放进不同的算法模型进行样本攻击，研究其对模型检测能力的影响，反之，可提高模型的自生防御能力。

     





