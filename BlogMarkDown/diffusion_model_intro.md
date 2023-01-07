# 扩散模型 (Diffusion Model) 简要介绍与源码分析


@[toc]

## 前言
近期同事分享了 Diffusion Model, 这才发现生成模型的发展已经到了如此惊人的地步, OpenAI 推出的 [Dall-E 2](https://openai.com/dall-e-2/) 可以根据文本描述生成极为逼真的图像, 质量之高直让人惊呼哇塞. 今早公众号给我推送了一篇关于 Stability AI 公司的报道, 他们推出的 AI 文生图扩散模型 Stable Diffusion 已开源, 能够在消费级显卡上实现 Dall-E 2 级别的图像生成, 效率提升了 30 倍. 

于是找到他们的开源产品体验了一把, 在线体验地址在 [https://huggingface.co/spaces/stabilityai/stable-diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion) (开源代码在 Github 上: [https://github.com/CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)), 在搜索框中输入 "A dog flying in the sky" (一只狗在天空飞翔), 生成效果如下:

<div align=center><img src="https://img-blog.csdnimg.cn/c706d79056fe4f13b232f5f9bc146cd5.png" width="50%"/></div>

**Amazing**! 当然, 不是每一张图片都符合预期, 但好在可以生成无数张图片, 其中总有效果好的. 在震惊之余, 不免对 Diffusion Model (扩散模型) 背后的原理感兴趣, 就想看看是怎么实现的. 

当时同事分享时, PPT 上那一堆堆公式扑面而来, 把我给整懵圈了, 但还是得撑起下巴, 表现出似有所悟、深以为然的样子, 在讲到关键处不由暗暗点头以表示理解和赞许. 后面花了个周末专门学习了一下, 公式推导+代码分析, 感觉终于了解了基本概念, 于是记录下来形成此文, 不敢说自己完全懂了, 毕竟我不做这个方向, 但回过头去看 PPT 上的公式就不再发怵了.


## 广而告之

可以在微信中搜索 “珍妮的算法之路” 或者 “world4458” 关注我的微信公众号, 可以及时获取最新原创技术文章更新.

另外可以看看知乎专栏 [PoorMemory-机器学习](https://www.zhihu.com/column/c_1323592860575502336), 以后文章也会发在知乎专栏中. 

## 总览

本文对 Diffusion Model 扩散模型的原理进行简要介绍, 然后对源码进行分析. 扩散模型的实现有多种形式, 本文关注的是 DDPM (denoising diffusion probabilistic models). 在介绍完基本原理后, 对作者释放的 Tensorflow 源码进行分析, 加深对各种公式的理解.


### 参考文章
在理解扩散模型的路上, 受到下面这些文章的启发, 强烈推荐阅读:

+ Lilian 的博客, 内容非常非常详实, 干货十足, 而且每篇文章都极其用心, 向大佬学习: [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
+ ewrfcas 的知乎, 公式推导补充了更多的细节: [由浅入深了解Diffusion Model](https://zhuanlan.zhihu.com/p/525106459)
+ Lilian 的博客, 介绍变分自动编码器 VAE: [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/#vae-variational-autoencoder), Diffusion Model 需要从分布中随机采样样本, 该过程无法求导, 需要使用到 VAE 中介绍的重参数技巧.
+ [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf) 论文, 
	+ 其 TF 源码位于: [https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion), **源码介绍以该版本为主**
	+ PyTorch 的开源实现: [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), 核心逻辑和上面 Tensorflow 版本是一致的, Stable Diffusion 参考的是 pytorch 版本的代码.


## 扩散模型介绍

### 基本原理

Diffusion Model (扩散模型) 是一类生成模型, 和 VAE (Variational Autoencoder, 变分自动编码器), GAN (Generative Adversarial Network, 生成对抗网络) 等生成网络不同的是,  扩散模型在前向阶段对图像逐步施加噪声, 直至图像被破坏变成完全的高斯噪声, 然后在逆向阶段学习从高斯噪声还原为原始图像的过程.

具体来说, 前向阶段在原始图像 $\mathbf{x}_0$ 上逐步增加噪声, 每一步得到的图像 $\mathbf{x}_t$ 只和上一步的结果 $\mathbf{x}_{t - 1}$ 相关, 直至第 $T$ 步的图像 $\mathbf{x}_T$ 变为纯高斯噪声. 前向阶段图示如下:

<div align=center><img src="https://img-blog.csdnimg.cn/d7a2d5872c0f446dab9f71f4222b57b6.png" width="80%"/></div>

而逆向阶段则是不断去除噪声的过程, 首先给定高斯噪声 $\mathbf{x}_T$, 通过逐步去噪, 直至最终将原图像 $\mathbf{x}_0$ 给恢复出来, 逆向阶段图示如下:

<div align=center><img src="https://img-blog.csdnimg.cn/48e916d641a443dd8313f854d77bbe27.png" width="80%"/></div>

模型训练完成后, 只要给定高斯随机噪声, 就可以生成一张从未见过的图像. 下面分别介绍前向阶段和逆向阶段, 只列出重要公式, 

### 前向阶段
由于前向过程中图像 $\mathbf{x}_t$ 只和上一时刻的 $\mathbf{x}_{t - 1}$ 有关, 该过程可以视为马尔科夫过程, 满足:

$$
\begin{align}
q\left(x_{1: T} \mid x_0\right) &=\prod_{t=1}^T q\left(x_t \mid x_{t-1}\right)  \\
q\left(x_t \mid x_{t-1}\right) &=\mathcal{N}\left(x_t ; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I}\right), 
\end{align}
$$

其中 $\beta_t\in(0, 1)$ 为高斯分布的方差超参, 并满足 $\beta_1 < \beta_2 < \ldots < \beta_T$. 另外公式 (2) 中为何均值 $x_{t-1}$ 前乘上系数 $\sqrt{1-\beta_t} x_{t-1}$ 的原因将在后面的推导介绍. 上述过程的一个美妙性质是我们可以在任意 time step 下通过 [重参数技巧](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick) 采样得到 $x_t$. 

 [重参数技巧 (reparameterization trick)](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick) 是为了解决随机采样样本这一过程无法求导的问题. 比如要从高斯分布 $z \sim \mathcal{N}(z; \mu, \sigma^2\mathbf{I})$ 中采样样本 $z$, 可以通过引入随机变量 $\epsilon\sim\mathcal{N}(0, \mathbf{I})$, 使得 $z = \mu + \sigma\odot\epsilon$, 此时 $z$ 依旧具有随机性, 且服从高斯分布 $\mathcal{N}(\mu, \sigma^2\mathbf{I})$, 同时 $\mu$ 与 $\sigma$ (通常由网络生成) 可导. 

简要了解了重参数技巧后, 再回到上面通过公式 (2) 采样 $x_t$ 的方法, 即生成随机变量 $\epsilon_t\sim\mathcal{N}(0, \mathbf{I})$, 
然后令 $\alpha_t = 1 - \beta_t$, 以及 $\overline{\alpha_t} = \prod_{i=1}^{T}\alpha_t$, 从而可以得到:

$$
\begin{align}
x_t &= \sqrt{1 - \beta_t} x_{t-1}+\beta_t \epsilon_1  \quad \text { where } \; \epsilon_1, \epsilon_2, \ldots \sim \mathcal{N}(0, \mathbf{I}), \; \text{reparameter trick} ; \nonumber \\
&=\sqrt{a_t} x_{t-1}+\sqrt{1-\alpha_t} \epsilon_1\nonumber \\
&=\sqrt{a_t}\left(\sqrt{a_{t-1}} x_{t-2}+\sqrt{1-\alpha_{t-1}} \epsilon_2\right)+\sqrt{1-\alpha_t} \epsilon_1 \nonumber \\
&=\sqrt{a_t a_{t-1}} x_{t-2}+\left(\sqrt{a_t\left(1-\alpha_{t-1}\right)} \epsilon_2+\sqrt{1-\alpha_t} \epsilon_1\right) \tag{3-1} \\
&=\sqrt{a_t a_{t-1}} x_{t-2}+\sqrt{1-\alpha_t \alpha_{t-1}} \bar{\epsilon}_2  \quad \text { where } \quad \bar{\epsilon}_2 \sim \mathcal{N}(0, \mathbf{I}) ; \tag{3-2} \\
&=\ldots \nonumber \\
&=\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \bar{\epsilon}_t.
\end{align}
$$

其中公式 (3-1) 到公式 (3-2) 的推导是由于独立高斯分布的可见性, 有 $\mathcal{N}\left(0, \sigma_1^2\mathbf{I}\right) +\mathcal{N}\left(0,\sigma_2^2 \mathbf{I}\right)\sim\mathcal{N}\left(0, \left(\sigma_1^2 + \sigma_2^2\right)\mathbf{I}\right)$, 因此:

$$
\begin{aligned}
&\sqrt{a_t\left(1-\alpha_{t-1}\right)} \epsilon_2 \sim \mathcal{N}\left(0, a_t\left(1-\alpha_{t-1}\right) \mathbf{I}\right) \\
&\sqrt{1-\alpha_t} \epsilon_1 \sim \mathcal{N}\left(0,\left(1-\alpha_t\right) \mathbf{I}\right) \\
&\sqrt{a_t\left(1-\alpha_{t-1}\right)} \epsilon_2+\sqrt{1-\alpha_t} \epsilon_1 \sim \mathcal{N}\left(0,\left[\alpha_t\left(1-\alpha_{t-1}\right)+\left(1-\alpha_t\right)\right] \mathbf{I}\right) \\
&=\mathcal{N}\left(0,\left(1-\alpha_t \alpha_{t-1}\right) \mathbf{I}\right)  .
\end{aligned}
$$

注意公式 (3-2) 中 $\bar{\epsilon}_2 \sim \mathcal{N}(0, \mathbf{I})$, 因此还需乘上 $\sqrt{1-\alpha_t \alpha_{t-1}}$. 从公式 (3) 可以看出 

$$
\begin{align}
q\left(x_t \mid x_0\right)=\mathcal{N}\left(x_t ; \sqrt{\bar{a}_t} x_0,\left(1-\bar{a}_t\right) \mathbf{I}\right)
\end{align}
$$

注意由于 $\beta_t\in(0, 1)$ 且 $\beta_1 < \ldots < \beta_T$, 而 $\alpha_t = 1 - \beta_t$, 因此 $\alpha_t\in(0, 1)$ 并且有 $\alpha_1 > \ldots>\alpha_T$, 另外由于 $\bar{\alpha}_t=\prod_{i=1}^T\alpha_t$, 因此当 $T\rightarrow\infty$ 时, $\bar{\alpha}_t\rightarrow0$ 以及 $(1-\bar{a}_t)\rightarrow 1$, 此时 $x_T\sim\mathcal{N}(0, \mathbf{I})$. 从这里的推导来看, 在公式 (2) 中的均值 $x_{t-1}$ 前乘上系数 $\sqrt{1-\beta_t} x_{t-1}$ 会使得 $x_{T}$ 最后收敛到标准高斯分布.
 

### 逆向阶段
前向阶段是加噪声的过程, 而逆向阶段则是将噪声去除, 如果能得到逆向过程的分布 $q\left(x_{t-1} \mid x_t\right)$, 那么通过输入高斯噪声 $x_T\sim\mathcal{N}(0, \mathbf{I})$, 我们将生成一个真实的样本. 注意到当 $\beta_t$ 足够小时, $q\left(x_{t-1} \mid x_t\right)$ 也是高斯分布, 具体的证明在 ewrfcas 的知乎文章: [由浅入深了解Diffusion Model](https://zhuanlan.zhihu.com/p/525106459) 推荐的论文中: `On the theory of stochastic processes, with particular reference to applications`. 我大致看了一下, 哈哈, 没太看明白, 不过想到这个不是我关注的重点, 因此 pass. 由于我们无法直接推断 $q\left(x_{t-1} \mid x_t\right)$, 因此我们将使用深度学习模型 $p_{\theta}$ 去拟合分布 $q\left(x_{t-1} \mid x_t\right)$, 模型参数为 $\theta$:


$$
\begin{align}
p_\theta\left(x_{0: T}\right) &=p\left(x_T\right) \prod_{t=1}^T p_\theta\left(x_{t-1} \mid x_t\right) \\
p_\theta\left(x_{t-1} \mid x_t\right) &=\mathcal{N}\left(x_{t-1} ; \mu_\theta\left(x_t, t\right), \Sigma_\theta\left(x_t, t\right)\right)
\end{align}
$$

注意到, 虽然我们无法直接求得 $q\left(x_{t-1} \mid x_t\right)$ (注意这里是 $q$ 而不是模型 $p_{\theta}$), 但在知道 $x_0$ 的情况下, 可以通过贝叶斯公式得到 $q\left(x_{t-1} \mid x_t, x_0\right)$ 为:

$$
\begin{align}
q\left(x_{t-1} \mid x_t, x_0\right) &= \mathcal{N}\left(x_{t-1} ; {\color{blue}{\tilde{\mu}}(x_t, x_0)}, {\color{red}{\tilde{\beta}_t} \mathbf{I}}\right)
\end{align}
$$

推导过程如下:

$$
\begin{aligned}
q(x_{t-1} \vert x_t, x_0) 
&= q(x_t \vert x_{t-1}, x_0) \frac{ q(x_{t-1} \vert x_0) }{ q(x_t \vert x_0) } \\
&\propto \exp \Big(-\frac{1}{2} \big(\frac{(x_t - \sqrt{\alpha_t} x_{t-1})^2}{\beta_t} + \frac{(x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} x_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(x_t - \sqrt{\bar{\alpha}_t} x_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp \Big(-\frac{1}{2} \big(\frac{x_t^2 - 2\sqrt{\alpha_t} x_t \color{blue}{x_{t-1}} \color{black}{+ \alpha_t} \color{red}{x_{t-1}^2} }{\beta_t} + \frac{ \color{red}{x_{t-1}^2} \color{black}{- 2 \sqrt{\bar{\alpha}_{t-1}} x_0} \color{blue}{x_{t-1}} \color{black}{+ \bar{\alpha}_{t-1} x_0^2}  }{1-\bar{\alpha}_{t-1}} - \frac{(x_t - \sqrt{\bar{\alpha}_t} x_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp\Big( -\frac{1}{2} \big( \underbrace{\color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} x_{t-1}^2}_{x_{t-1} \text { 方差 }} - \underbrace{\color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t} x_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} x_0)} x_{t-1}}_{x_{t-1} \text { 均值 }} +  \underbrace{{\color{black}{ C(x_t, x_0)}}}_{\text {与 } x_{t-1} \text { 无关 }} \big) \Big)
\end{aligned}
$$


上面推导过程中, 通过贝叶斯公式巧妙的将逆向过程转换为前向过程, 且最终得到的概率密度函数和高斯概率密度函数的指数部分 $\exp{\left(-\frac{\left(x - \mu\right)^2}{2\sigma^2}\right)} = \exp{\left(-\frac{1}{2}\left(\frac{1}{\sigma^2}x^2 - \frac{2\mu}{\sigma^2}x + \frac{\mu^2}{\sigma^2}\right)\right)}$ 能对应, 即有:

$$
\begin{align}
\tilde{\beta}_t 
&= 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) 
= 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})})
= \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
\tilde{\mu}_t (x_t, x_0)
&= (\frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} x_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \nonumber\\
&= (\frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} x_0) \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t}\nonumber \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0\\
\end{align}
$$

通过公式 (8) 和公式 (9), 我们能得到 $q\left(x_{t-1} \mid x_t, x_0\right)$ (见公式 (7)) 的分布. 此外由于公式 (3) 揭示的 $x_t$ 和 $x_0$ 之间的关系: $x_t =\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \bar{\epsilon}_t$, 可以得到 

$$
\begin{align}
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_t)
\end{align}
$$

代入公式 (9) 中得到:

$$
\begin{align}
\tilde{\mu}_t
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_t)\nonumber \\
&= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_t \Big)}
\end{align}
$$

补充一下公式 (11) 的详细推导过程:

<div align=center><img src="https://img-blog.csdnimg.cn/d74296b2f22f4da0b2cc653c08f529c1.jpeg" width="60%"/></div>


前面说到, 我们将使用深度学习模型 $p_{\theta}$ 去拟合逆向过程的分布 $q\left(x_{t-1} \mid x_t\right)$, 由公式 (6) 知 $p_\theta\left(x_{t-1} \mid x_t\right) =\mathcal{N}\left(x_{t-1} ; \mu_\theta\left(x_t, t\right), \Sigma_\theta\left(x_t, t\right)\right)$, 我们希望训练模型 $\mu_\theta\left(x_t, t\right)$ 以预估 $\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \Big( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_t \Big)$. 由于 $x_t$ 在训练阶段会作为输入, 因此它是已知的, 我们可以转而让模型去预估噪声 $\epsilon_t$, 即令:

$$
\begin{align}
\mu_\theta(x_t, t) &= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \Big)} \\
\text{Thus }x_{t-1} &= \mathcal{N}(x_{t-1}; \frac{1}{\sqrt{\alpha_t}} \Big( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \Big), \boldsymbol{\Sigma}_\theta(x_t, t))
\end{align}
$$


### 模型训练
前面谈到, 逆向阶段让模型去预估噪声 $\epsilon_\theta(x_t, t)$, 那么应该如何设计 Loss 函数 ? 我们的目标是在真实数据分布下, 最大化模型预测分布的对数似然, 即优化在 $x_0\sim q(x_0)$ 下的 $p_\theta(x_0)$ 交叉熵:

$$
\begin{align}
\mathcal{L} = \mathbb{E}_{q(x_0)}\left[-\log{p_\theta(x_0)}\right]
\end{align}
$$

和 [变分自动编码器 VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) 类似, 使用 Variational Lower Bound 来优化: $-\log{p_\theta(x_0)}$ :

$$
\begin{align}
-\log p_\theta\left(x_0\right) &\leq-\log p_\theta\left(x_0\right)+D_{K L}\left(q\left(x_{1: T} \mid x_0\right) \| p_\theta\left(x_{1: T} \mid x_0\right)\right); \quad \text{注: 注意KL散度非负}\nonumber\\
&=-\log p_\theta\left(x_0\right)+\mathbb{E}_{q\left(x_{1: T} \mid x_0\right)}\left[\log \frac{q\left(x_{1: T} \mid x_0\right)}{p_\theta\left(x_{0: T}\right) / p_\theta\left(x_0\right)}\right] ; \; \text { where } \; p_\theta\left(x_{1: T} \mid x_0\right)=\frac{p_\theta\left(x_{0: T}\right)}{p_\theta\left(x_0\right)}\nonumber\\
&=-\log p_\theta\left(x_0\right)+\mathbb{E}_{q\left(x_{1: T} \mid x_0\right)}[\log \frac{q\left(x_{1: T} \mid x_0\right)}{p_\theta\left(x_{0: T}\right)}+\underbrace{\log p_\theta\left(x_0\right)}_{\text {与q无关 }}]\nonumber\\
&=\mathbb{E}_{q\left(x_{1: T} \mid x_0\right)}\left[\log \frac{q\left(x_{1: T} \mid x_0\right)}{p_\theta\left(x_{0: T}\right)}\right] .
\end{align}
$$

对公式 (15) 左右两边取期望 $\mathbb{E}_{q(x_0)}$, 利用到重积分中的 [Fubini 定理](https://en.wikipedia.org/wiki/Fubini%27s_theorem) 可得:

$$
\mathcal{L}_{V L B}=\underbrace{\mathbb{E}_{q\left(x_0\right)}\left(\mathbb{E}_{q\left(x_{1: T} \mid x_0\right)}\left[\log \frac{q\left(x_{1: T} \mid x_0\right)}{p_\theta\left(x_{0: T}\right)}\right]\right)=\mathbb{E}_{q\left(x_{0: T}\right)}\left[\log \frac{q\left(x_{1: T} \mid x_0\right)}{p_\theta\left(x_{0: T}\right)}\right]}_{\text {Fubini定理 }} \geq \mathbb{E}_{q\left(x_0\right)}\left[-\log p_\theta\left(x_0\right)\right]
$$

因此最小化 $\mathcal{L}_{V L B}$ 就可以优化公式 (14) 中的目标函数. 之后对 $\mathcal{L}_{V L B}$ 做进一步的推导, 这部分的详细推导见上面的参考文章, 最终的结论是: 

$$
\begin{align}
\mathcal{L}_{V L B} &= L_T + L_{T - 1} + \ldots + L_0 \\
L_T &= D_{KL}\left(q(x_T|x_0)||p_{\theta}(x_T)\right) \\
L_t &= D_{KL}\left(q(x_t|x_{t - 1}, x_0)||p_{\theta}(x_t|x_{t+1})\right); \quad 1 \leq t \leq T - 1 \\
L_0 &= -\log{p_\theta\left(x_0|x_1\right)}
\end{align}
$$

最终是优化两个高斯分布 $q(x_t|x_{t - 1}, x_0) = \mathcal{N}\left(x_{t-1} ; {\color{blue}{\tilde{\mu}}(x_t, x_0)}, {\color{red}{\tilde{\beta}_t} \mathbf{I}}\right)$ (详见公式 (7)) 与 $p_{\theta}(x_t|x_{t+1}) = \mathcal{N}\left(x_{t-1} ; \mu_\theta\left(x_t, t\right), \Sigma_\theta\right)$ (详见公式(6), 此为模型预估的分布)之间的 KL 散度. 由于多元高斯分布的 KL 散度存在闭式解, 详见: [Multivariate_normal_distributions](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions), 从而可以得到:

$$
\begin{align}
L_t 
&= \mathbb{E}_{x_0, \epsilon} \Big[\frac{1}{2 \| \boldsymbol{\Sigma}_\theta(x_t, t) \|^2_2} \| \color{blue}{\tilde{\mu}_t(x_t, x_0)} - \color{green}{\mu_\theta(x_t, t)} \|^2 \Big] \\
&= \mathbb{E}_{x_0, \epsilon} \Big[\frac{1}{2  \|\boldsymbol{\Sigma}_\theta \|^2_2} \| \color{blue}{\frac{1}{\sqrt{\alpha_t}} \Big( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_t \Big)} - \color{green}{\frac{1}{\sqrt{\alpha_t}} \Big( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(x_t, t) \Big)} \|^2 \Big] \\
&= \mathbb{E}_{x_0, \epsilon} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\epsilon_t - \epsilon_\theta(x_t, t)\|^2 \Big]; \quad \text{其中} \epsilon_t \text{为高斯噪声}, \epsilon_{\theta} \text{为模型学习的噪声} \\
&= \mathbb{E}_{x_0, \epsilon} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\epsilon_t - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon_t, t)\|^2 \Big] 
\end{align}
$$

DDPM 将 Loss 简化为如下形式:

$$
\begin{align}
L_t^{\text {simple }}=\mathbb{E}_{x_0, \epsilon_t}\left[\left\|\epsilon_t-\epsilon_\theta\left(\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon_t, t\right)\right\|^2\right]
\end{align}
$$

因此 Diffusion 模型的目标函数即是学习高斯噪声  $\epsilon_t$  和 $\epsilon_{\theta}$ (来自模型输出) 之间的 MSE loss.


### 最终算法
最终 DDPM 的算法流程如下:

<div align=center><img src="https://img-blog.csdnimg.cn/78b363ccbd264fe18878b4966187e515.png" width="80%"/></div>

**训练阶段**重复如下步骤:

+ 从数据集中采样 $x_0$
+ 随机选取 time step $t$
+ 生成高斯噪声 $\epsilon_t\in\mathcal{N}(0, \mathbf{I})$
+ 调用模型预估 $\epsilon_\theta\left(\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon_t, t\right)$
+ 计算噪声之间的 MSE Loss: $\left\|\epsilon_t-\epsilon_\theta\left(\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon_t, t\right)\right\|^2$, 并利用反向传播算法训练模型.

**逆向阶段**采用如下步骤进行采样:

+ 从高斯分布采样 $x_T$
+ 按照 $T, \ldots, 1$ 的顺序进行迭代:
	+ 如果 $t = 1$, 令 $\mathbf{z} = {0}$; 如果 $t > 1$, 从高斯分布中采样 $\mathbf{z}\sim\mathcal{N}(0, \mathbf{I})$
	+ 利用公式 (12) 学习出均值 $\mu_\theta(x_t, t) = \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \Big)}$, 并利用公式 (8) 计算均方差 $\sigma_t = \sqrt{\tilde{\beta}_t} = \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t}$
	+ 通过重参数技巧采样 $x_{t - 1} = \mu_\theta(x_t, t) + \sigma_t\mathbf{z}$
+ 经过以上过程的迭代, 最终恢复 $x_0$.


## 源码分析
DDPM 文章以及代码的相关信息如下:

+ [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf) 论文, 
	+ 其 TF 源码位于: [https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion), **源码介绍以该版本为主**
	+ PyTorch 的开源实现: [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), 核心逻辑和上面 Tensorflow 版本是一致的, Stable Diffusion 参考的是 pytorch 版本的代码.

本文以分析 Tensorflow 源码为主, Pytorch 版本的代码和 Tensorflow 版本的实现逻辑大体不差的, 变量名字啥的都类似, 阅读起来不会有啥门槛. Tensorlow 源码对 Diffusion 模型的实现位于 [diffusion_utils_2.py](https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py), 模型本身的分析以该文件为主.

### 训练阶段

以 CIFAR 数据集为例. 

在 [run_cifar.py](https://github.com/hojonathanho/diffusion/blob/master/scripts/run_cifar.py) 中进行前向传播计算 Loss:

<div align=center><img src="https://img-blog.csdnimg.cn/a937c5eb474040b9bdf75117167007dc.png" width="80%"/></div>

+ 第 6 行随机选出 $t\sim\text{Uniform}(\{1, \ldots, T\})$
+ 第 7 行 `training_losses` 定义在 [GaussianDiffusion2](https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py) 中, 计算噪声间的 MSE Loss.

进入 [GaussianDiffusion2](https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py) 中, 看到初始化函数中定义了诸多变量, 我在注释中使用公式的方式进行了说明:

<div align=center><img src="https://img-blog.csdnimg.cn/29195dd538fd483282def3712059e1ce.png" width="80%"/></div>

下面进入到 `training_losses` 函数中:

<div align=center><img src="https://img-blog.csdnimg.cn/b0ad91e7a69f42fc8e9380deea7607ba.png" width="80%"/></div>

+ 第 19 行: `self.model_mean_type` 默认是 `eps`, 模型学习的是噪声, 因此 `target` 是第 6 行定义的 `noise`, 即 $\epsilon_t$
+ 第  9 行: 调用 `self.q_sample` 计算 $x_t$, 即公式 (3) $x_t =\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon_t$
+ 第 21 行: `denoise_fn` 是定义在 [unet.py](https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py) 中的 `UNet` 模型, 只需知道它的输入和输出大小相同; 结合第 9 行得到的 $x_t$, 得到模型预估的噪声: $\epsilon_\theta\left(\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon_t, t\right)$
+ 第 23 行: 计算两个噪声之间的 MSE: $\left\|\epsilon_t-\epsilon_\theta\left(\sqrt{\bar{\alpha}_t} x_0+\sqrt{1-\bar{\alpha}_t} \epsilon_t, t\right)\right\|^2$, 并利用反向传播算法训练模型

上面第 9 行定义的 `self.q_sample` 详情如下:

<div align=center><img src="https://img-blog.csdnimg.cn/1c0f9264cbb4413680d9df46eb997e48.png" width="80%"/></div>

+ 第 13 行的 `q_sample` 已经介绍过, 不多说.
+ 第 2 行的  `_extract` 在代码中经常被使用到, 看到它只需知道它是用来提取系数的即可. 引入输入是一个 Batch, 里面的每个样本都会随机采样一个 time step $t$, 因此需要使用 `tf.gather` 来将 $\bar{\alpha_t}$ 之类选出来, 然后将系数 reshape 为 `[B, 1, 1, ....]` 的形式, 目的是为了利用 broadcasting 机制和 $x_t$ 这个 Tensor 相乘.

前向的训练阶段代码实现非常简单, 下面看逆向阶段

### 逆向阶段
逆向阶段代码定义在 [GaussianDiffusion2](https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py) 中:

<div align=center><img src="https://img-blog.csdnimg.cn/368c1f55b0ef4271b40c82df19d818cc.png" width="80%"/></div>

+ 第 5 行生成高斯噪声 $x_T$, 然后对其不断去噪直至恢复原始图像
+ 第 11 行的 `self.p_sample` 就是公式 (6) $p_\theta\left(x_{t-1} \mid x_t\right) =\mathcal{N}\left(x_{t-1} ; \mu_\theta\left(x_t, t\right), \Sigma_\theta\left(x_t, t\right)\right)$ 的过程, 使用模型来预估 $\mu_\theta\left(x_t, t\right)$ 以及 $\Sigma_\theta\left(x_t, t\right)$
+ 第 12 行的 `denoise_fn` 在前面说过, 是定义在 [unet.py](https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py) 中的 `UNet` 模型; `img_` 表示 $x_t$.
+ 第 13 行的 `noise_fn` 则默认是 `tf.random_normal`, 用于生成高斯噪声.

进入 `p_sample` 函数:

<div align=center><img src="https://img-blog.csdnimg.cn/c4f2e2ed32724dc8b47352594f1fdcab.png" width="80%"/></div>

+ 第 7 行调用 `self.p_mean_variance` 生成 $\mu_\theta\left(x_t, t\right)$ 以及 $\log\left(\Sigma_\theta\left(x_t, t\right)\right)$, 其中 $\Sigma_\theta\left(x_t, t\right)$ 通过计算 $\tilde{\beta}_t$ 得到.
+ 第 11 行从高斯分布中采样 $\mathbf{z}$
+ 第 18 行通过重参数技巧采样 $x_{t - 1} = \mu_\theta(x_t, t) + \sigma_t\mathbf{z}$, 其中 $\sigma_t = \sqrt{\tilde{\beta}_t}$

进入 `self.p_mean_variance` 函数:

<div align=center><img src="https://img-blog.csdnimg.cn/a34571934cd74c74a12a9fb031edb6a2.png" width="80%"/></div>

+ 第 6 行调用模型 `denoise_fn`, 通过输入 $x_t$, 输出得到噪声 $\epsilon_t$
+ 第 19 行 `self.model_var_type` 默认为 `fixedlarge`, 但我当时看 `fixedsmall` 比较爽, 因此 `model_variance` 和 `model_log_variance` 分别为 $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$ (见公式 8), 以及 $\log\tilde{\beta}_t$
+ 第 29 行调用 `self._predict_xstart_from_eps` 函数, 利用公式 (10) 得到 $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_t)$
+ 第 30 行调用 `self.q_posterior_mean_variance` 通过公式 (9) 得到 $\mu_\theta(x_t, x_0)  = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0$

 `self._predict_xstart_from_eps` 函数相亲如下:

<div align=center><img src="https://img-blog.csdnimg.cn/ed36c89494f8444bb6c886e30300e64f.png" width="80%"/></div>

+ 该函数计算 $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_t)$

`self.q_posterior_mean_variance` 函数详情如下:

<div align=center><img src="https://img-blog.csdnimg.cn/182a53eb786d447f8dc90290ee056503.png" width="80%"/></div>

+ 相关说明见注释, 另外发现对于 $\mu_\theta(x_t, x_0)$ 的计算使用的是公式 (9) $\mu_\theta(x_t, x_0)  = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0$ 而不是进一步推导后的公式 (11) $\mu_\theta(x_t, x_0)  = \frac{1}{\sqrt{\alpha_t}} \Big( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_t \Big)$. 



## 总结
写文章真的挺累的, 好处是, 我发现写之前我以为理解了, 但写的过程中又发现有些地方理解的不对. 写完后才终于把逻辑理顺. 





