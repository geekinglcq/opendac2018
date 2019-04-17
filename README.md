# opendac2018
2018开放学术数据挖掘大赛 第四名解题方案  No.4 solution of Open Academic Data Challenge 2018

Authors： 萌新们  

December 2018

## 简介  

我们队伍这次参加2018开放学术数据挖掘大赛取得了第四名的成绩, 这次参赛的契机在于我们队在去年的学术精准画像比赛上拿到了No.2的成绩, 代码开源在[Github](https://github.com/geekinglcq/aca)上, 陆陆续续也有不少人来咨询相关的情况,因此想着作为一个优良传统就继续参加了今年的比赛.    

该程序的主入口在`main.ipynb`中， 请用jupyter notebook运行。  
本程序使用python3, 需要使用的包在`requirements.txt`.  代码中有用到来自论文 Name Disambiguation in AMiner: Clustering, Maintenance, and Human in the Loop的[示例代码](https://github.com/zhysora/BUAALAB_IN_WDQ), 半监督聚类的代码改动自[这里](https://github.com/datamole-ai/active-semi-supervised-clustering).  

总的介绍一下我们的解题思路, 大体上受启发于KDD论文Name Disambiguation in AMiner: Clustering, Maintenance, and Human in the Loop. 首先进行的是全局度量学习, 这一阶段过后我们可以对每篇论文学到一个向量表征, 利用该向量表征,我们可以对每个名字下的论文按照一定的规则构建一个子图(图中点即为每篇paper, 连着的边表示我们认为这两篇文章是同一个作者写的).   
接下来我们探索了两种处理方案:    

1) 对于每个子图, 首先利用一些强规则来增加图中的边, 然后利用并查集来简单打通图中的连通分量进行聚类. 然后可以直接输出结果.  

2) 将每个子图传入一个两层的图卷积神经网络, 精调paper的向量表征.接着:   

    2a) 利用XMeans算法(规避Kmeans中K的选择)来做无监督聚类.  
 
    2b) 利用一些强规则生成Pair-wise的限制,并由此利用PCKMeans进行半监督聚类.  
    
最终提交结果来看, 并查集方案由于其泛化性良好,取得了我们提交中的最高分0.69, 2b的半监督聚类方案分数低一些0.64, 2a的无监督聚类0.58.  

## 全局度量学习  

首先，我们用文档中词向量的加权平均$x_i = \sum_{w_i\in D_i} \alpha_i w_i$表示文档$D_i$的特征, 其中权重$\alpha_i$为词$w_i$的逆文档频率（inverse document frequency, idf）值。
	
在这一基础上，进一步使用三元损失进行微调，其目的是在特征空间里，使得同属一簇的样本距离尽可能近，而不同簇的尽可能远。 换言之，在微调后的特征空间中，一篇文档$D_i$，它与和它同属一簇的另一篇文档$D_{i+}$的距离，要比和它不属一簇的一篇文档$D_{i-}$要近。那么，我们可以把这一目标形式化为：  

![](http://latex.codecogs.com/gif.latex?L=\sum_{(D_i,D_{i+},D_{i-})\in%20S}\max(0,%20\delta(y_i,%20y_{i+})%20-%20\delta(y_i,%20y_{i-})%20+%20m%20))
  
其中$S$是所有可能的三元组的集合，由于该集合可能非常大，实际我们可以从中采样一些三元组；$y_i$是微调后的特征空间；$\delta(x, y)$是距离函数，实际使用欧氏距离。
	
训练时，该微调模型每次接受三元组作为参数，使用两层共享权重的全连接层，归一化后，接上述三元损失。
	
推断时，该模型只接受一个输入$x_i$， 输出微调后的特征$y_i$。	  

## 并查集    

并查集非常简单, 它是一种树型的数据结构，用于处理一些不交集（Disjoint Sets）的合并及查询问题。我们把所有论文先作为单独元素的集合输入, 然后对每对连边的论文做Union操作. 最终我们可以得到一个天然的划分来当做聚类.  

## 聚类     

聚类的探索上我们尝试了两种方案,一种是无监督聚类, 主要使用了XMeans算法,另一种是半监督聚类,主要使用了PCKMeans算法.  

主要介绍一下半监督聚类. 这聚类中,我们通过一些强规则(例如合作者完全一致)生成了一些所谓Must-link的pairwise constraints. 然后我们将这些限制加入到了优化目标中, 这样我们的目标函数就是:  

![](http://latex.codecogs.com/gif.latex?J_{pckm}=\sum_{x_i\in%20X}(||x_i-\mu_{l_i}||^2)%20+%20\sum_{(x_i,x_j)\in%20M}w_{ij}f_M(x_i,x_j)\mathbbm{1}[l_i\neq%20l_j])

同时, 利用这些限制我们也可以优化聚类的初始化,使结果更加鲁棒. 当这些限制导出的连通分量大于聚类数时,我们利用farthest-first遍历来选取聚类, 而当连通分量数小于聚类数,我们可以将所有的连通分量的中心作为聚类中心再随机安排剩下的聚类中心.  

