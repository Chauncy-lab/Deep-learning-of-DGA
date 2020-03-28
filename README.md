# Research on automatic detection technology of DGA

基于深度学习对dga恶意域名检测研究

blog：https://blog.csdn.net/qq_39936434/article/details/103310765

## 前言

​	因为毕设而接触了该课题，发现很有趣。该仓库专注于DGA域名检测技术的研究，内容包括：传统机器学习、深度学习与生成对抗网络GAN。

## 历史演变

​	先看以下一张图了解为何会出现DGA域名检测技术

![image-20200215102947312](https://raw.githubusercontent.com/Xandra-chan/Deep-learning-of-DGA/master/image/image-20200215102947312.png )



​	由此，安全人员与攻击者开始在人工智能领域中展开漫长的博弈~

## 本仓库所做的研究

​	本仓库研究以传统机器学习为起点，按照历史存在的问题的演变顺序一步步推进。

### 传统机器学习

​	传统机器学习解决的问题：

​	（1）实现自动化检测技术，效率比传统的黑名单检验、逆向等方法高出一大截

​	本仓库的贡献：

​	（1）对兜哥的《Web安全之机器学习入门》关于DGA部分的代码基于多种特征提取进行复现

​	传统机器学习所存在的问题：

​	（1）基于手工提取的监督学习算法存在输入样本被攻击的问题

​	（2）缺乏即时的训练数据，往往攻击发生过后才能获取训练样本

### 深度学习

​	深度学习解决的问题：

​	（1）无监督算法无需人工特征提取，输入一定量的数据模型可以自己训练，一定程度解决了训练缺乏的问题

​	（2）自动进行特征提取，有时能以人类没想到的角度提取特征，提高检测效率	

​	本仓库的贡献：

​	（1）同样复现了兜哥的 《Web安全之深度学习实战》 关于DGA检测部分代码

​	（2）并且在此基础上使用深度学习方法与传统机器学习的检测方法进行对比，通过AUC，准确率等多种指标印证

​	（3）对多个算法比较中最优的模型——MLP，进行超参数调参，提高模型检测能力

​	深度学习所存在的问题：

​	（1）虽然模型可以自己学习并训练数据，但是其初始化调参时仍然需要输入样本进行基本的训练，所以任然存在输入样本被攻击的问题

### 对抗生成网络GAN

​	对抗网络GAN解决的问题：

​	（1）分为鉴别网络和生成网络，相互博弈训练。既能训练高检测的模型，也可生成高价值的对抗样本。增强了模型的防御能力和扩充了有效训练数据集。

​	本仓库的贡献：

​	（1）复现国内公开的[demo]( https://github.com/bts-webber/GAN_for_DGA )，但是该demo是一个失败的实验

​	（2）使用GAN技术进行对抗性域名生成，效果比（1）的好，但是总体仍然不能达到符合人类语言习惯的域名

​	（3）借鉴国外的成功demo，使用更加稳定的WGAN-GP技术成功生成符合需求的对抗性域名

​	

## 数据集

[360实验室]( http://data.netlab.360.com/ )

## 其他

[AI-Security-Learning]( https://github.com/0xMJ/AI-Security-Learning )

[AI-for-Security-Learning]( https://github.com/404notf0und/AI-for-Security-Learning )

[Domain Generation Algorithms]( https://github.com/baderj/domain_generation_algorithms )