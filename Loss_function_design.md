对于双任务学习而言，比较重要的是设置一个合理的损失函数，它能够给我们的训练设立一个精准的目标。在好的损失函数的指导下训练，通常会获得比较好的结果。

一般而言，损失函数是各个任务损失的加权，即：

$$L = \alpha_1 L_1 + \alpha_2 L2$$

这时我们需要设置的就是权重 $\alpha_1,\alpha_2$，实际上这里不需要考虑约束 $\alpha_1 + \alpha_2 = 1$，因为这样设置的损失函数是乘积不变的，也就是给损失函数乘上一个常数，损失函数关于参数的梯度方向不会改变，而这个常数会被吸收到学习率当中，此时只需要对训练时的学习率做相应的调整，就可以消除这种影响。因此我们只需要把精力放在寻找权重本身，而不是过度在意相互之间的约束。对于多任务学习而言，权重的设置影响着模型对不同任务的偏好，因此很难一下找到合适的权重组合，但这里讨论的范畴仅是双任务学习，不论是搜寻的范围还是训练的成本都大大缩小，也为我们寻找合适的损失函数提供了充足的发挥空间，因此我计划对以下各种损失函数进行一系列实验，分别从理论上和实验上给出有效的支撑。

人为主观地设置权重组合在模型训练经验丰富的前提下也许能起到不错的效果，但对于大部分人而言，这种方法依旧极大降低了训练效率，导致需要很长时间才能找到一个良好的权重组合（即使使用高效的寻参方法如网格搜索等）。因此我们继续思考，或许一个有效的办法是使用初始损失的倒数作为对应损失的权重去代替主观判断，即

$$L = \dfrac{L_1}{L_1^{(init)}} + \dfrac{L_2}{L_2^{(init)}}$$

在这种损失函数的设置下，如果一个任务的初始损失较大，那么对应的权重就会变小，反之则会变大，这就使得整体损失形成了一个有效的平衡。除此之外，这个损失函数还具有缩放不变性，即对 $L_i$ 乘上一个常数，整个损失函数 $L$ 的值不变。

以上方法有用的原因是，初始损失一定程度上能够反映模型学习该任务的难度，但仔细思考会感觉这样的主观判断似乎不一定成立，因此另一个办法是引入先验分布去代替主观判断，假设数据集中一个 $K$ 分类的先验分布为 $[p_1,p_2,\cdots,p_K]$，假设模型能够比较轻松地拟合这种分布，那么对于单条数据而言，设该数据的分类是第 $i$ 类（ $\text{one hot}$ ），使用交叉熵作为损失函数，模型对于这条数据的损失就是 $-\log p_i$，因此整个数据集的先验损失就是

$$L^{(prior)} = H = -\sum_{i=1}^K p_i\log p_i$$

特别地，对于二分类任务，上式中 $K=2$.

此时我们设置的真正损失是

$$L = \dfrac{L_1}{L_1^{(prior)}}+\dfrac{L_2}{L_2^{(prior)}}$$

再将这个损失特化到性别-情绪双任务上，性别必定是二分类的，而情绪可以是多分类的，假设情绪一共有 $K$ 类，则

$$L^{(prior)}_1 = -\sum_{i=1}^K p_i\log p_i, \quad L^{(prior)}_2  = -p_1\log p_1 - p_2\log p_2$$

某种意义上来说，“先验分布”比“初始分布”更能体现出“初始”的本质，它是“就算模型啥都学不会，也知道按照先验分布来随机出结果”的体现，所以此时的损失值更能代表当前任务的初始难度。

不过，前面这两种损失函数的权重始终是在训练之前就事先设定好的固定的值，我们还可以考虑权重随着训练的进程而动态调节的情况，因为只有投入训练，模型才能够真实感受到学习难度，因此随着训练的进程动态调节应该是更好的选择。

在训练过程中，反映模型学习某个任务难度的最直观的指标就是损失，因此一个很自然的想法是把损失设置为

$$L = \dfrac{L_1}{L_1^{(\text{stop gradient})}}+\dfrac{L_2}{L_2^{(\text{stop gradient})}}$$

$\text{stop gradient}$ 即只使用损失的值，而不进行梯度的计算，因此梯度计算只限制在分子部分，在这种情形下，每个任务的损失都变为 $\text{1}$，但梯度却不是 $0$.

进一步地，计算 $\nabla_{\theta}L$:

$$\nabla_{\theta}L = \dfrac{\nabla_{\theta}L_1}{L_1^{(\text{stop gradient})}}+\dfrac{\nabla_{\theta}L_2}{L_2^{(\text{stop gradient})}} = \nabla_{\theta}log L_1 + \nabla_{\theta}log L_2 = 2\nabla_{\theta}\log \sqrt{L_1L_2}$$

可以看到，损失函数 $L$ 和几何平均 $\sqrt{L_1L_2}$ 在梯度方向上一致，而当权重均取 $\dfrac{1}{2}$ 时，损失变为几何平均，我们不妨考虑推广情形，设

$$L = \sqrt[\gamma]{L_1^{\gamma} + L_2^{\gamma}}$$

这样就引入了一个超参数 $\gamma$，为了选取合适的 $\gamma$，有必要研究这个函数，为此，设 

$$f(\gamma) = \ln(\sqrt[\gamma]{L_1^{\gamma} + L_2^{\gamma}}) = \dfrac{\ln(L_1^{\gamma} + L_2^{\gamma})}{\gamma}$$

因为 $0$ 次方根没有定义，因此定义域为 $\mathbb{R}-\{0\}$，对 $\gamma$ 求导得：

$$f'(\gamma) = \dfrac{1}{\gamma^2}\left(\dfrac{\gamma(L_1^{\gamma}\ln L_1+L_2^{\gamma}\ln L_2)}{L_1^{\gamma} + L_2^{\gamma}} - \ln(L_1^{\gamma} + L_2^{\gamma})\right)$$

分子部分是

$$\dfrac{\gamma(L_1^{\gamma}\ln L_1+L_2^{\gamma}\ln L_2)}{L_1^{\gamma} + L_2^{\gamma}} - \ln(L_1^{\gamma} + L_2^{\gamma}) $$

$$= \dfrac{L_1^{\gamma}\ln L_1^\gamma+L_2^{\gamma}\ln L_2^\gamma}{L_1^{\gamma} + L_2^{\gamma}} - \ln(L_1^{\gamma} + L_2^{\gamma})$$ 

$$ = \dfrac{L_1^{\gamma}}{L_1^{\gamma} + L_2^{\gamma}}\ln L_1^{\gamma}+\dfrac{L_2^{\gamma}}{L_1^{\gamma} + L_2^{\gamma}}\ln L_2^{\gamma} - \ln(L_1^{\gamma} + L_2^{\gamma})$$

由于 $\ln x$ 是严格凹函数，并且 $L_1\neq L_2$，因此根据 $\text{Jensen}$ 不等式，上式 $< 0 $，说明 $f(\gamma)$ 单调递减，并且当 $\gamma \rightarrow -\infty/+\infty$ 时，上式 $\rightarrow 0$.

我们再探索一下该函数的极限，不妨设 $L_1 < L_2$，令 $k = \dfrac{L_1}{L_2} \in (0,1)$，当 $\gamma \rightarrow +\infty$ 时，

$$\lim_{\gamma \rightarrow +\infty}f(\gamma) = \lim_{\gamma \rightarrow +\infty}\dfrac{\ln(L_1^{\gamma} + L_2^{\gamma})}{\gamma} = \lim_{\gamma \rightarrow +\infty}\dfrac{\ln(L_2^{\gamma}k^{\gamma} + L_2^{\gamma})}{\gamma} = \lim_{\gamma \rightarrow +\infty}\dfrac{\gamma\ln L_2+\ln(k^{\gamma}+1)}{\gamma} = \ln L_2$$

而当 $\gamma \rightarrow -\infty$ 时，

$$\lim_{\gamma \rightarrow -\infty}f(\gamma) = \lim_{\gamma \rightarrow -\infty}\dfrac{\ln(L_1^{\gamma} + L_2^{\gamma})}{\gamma} = \lim_{\gamma \rightarrow -\infty}\dfrac{\ln(L_1^{\gamma} + L_1^{\gamma}{\dfrac{1}{k}}^\gamma)}{\gamma} = \lim_{\gamma \rightarrow -\infty}\dfrac{\gamma\ln L_1+\ln(k^{-\gamma}+1)}{\gamma} = \ln L_1$$

故得到

$$\lim_{\gamma \rightarrow +\infty}L = L_1, \quad \lim_{\gamma \rightarrow -\infty}L = L_2$$

即当 $\gamma$ 取得比较大的时，损失函数更倾向于两者之中的最大值，当 $\gamma$ 取得比较小的时候，损失函数则更倾向于两者之中的最小值.

由于定义域不含 $0$，因此研究 $0$ 附近的极限是有必要的。

当 $\gamma \rightarrow 0^-$，

$$\lim_{\gamma \rightarrow 0^-}f(\gamma) = \lim_{\gamma \rightarrow 0^-}\dfrac{\ln(L_1^{\gamma} + L_2^{\gamma})}{\gamma} = \lim_{\gamma \rightarrow 0^-}\dfrac{\ln(L_2^{\gamma}k^{\gamma} + L_2^{\gamma})}{\gamma} = \lim_{\gamma \rightarrow 0^-}\dfrac{\gamma\ln L_2+\ln(k^{\gamma}+1)}{\gamma} =-\infty$$

再进行一次一样的推导，只需把 $0^-$ 换为 $0^+$，极限就变为 $+\infty$.

因此，我们得到

$$\lim_{\gamma \rightarrow 0^-}L = 0, \quad \lim_{\gamma \rightarrow 0^+}L = +\infty$$

前面用 $L_i^{(\text{stop gradient})}$ 来对损失函数做归一化，能够取到比较好的效果，因此自然而然地可以想到，如果不是对损失进行归一，而是对梯度进行归一，是否也可以有更好的效果，因此尝试取

$$L = \dfrac{L_1}{||\nabla_{\theta}L_1||} + \dfrac{L_2}{||\nabla_{\theta}L_2||}$$

此时计算损失函数时还需计算梯度，会稍有些麻烦，但因为

$$\nabla_{\theta} L = \dfrac{\nabla_{\theta}L_1}{||\nabla_{\theta}L_1||} + \dfrac{\nabla_{\theta}L_2}{||\nabla_{\theta}L_2||}$$

因此我们在计算梯度时，只需对单任务的损失求梯度并归一化，再相加即可得到总损失函数的梯度。

若使用 $\text{SGD}$，那么梯度变化量为 $\Delta\theta = -\eta \nabla_{\theta} L$，满足

$$\langle\Delta\theta, \nabla_{\theta}L_1\rangle = \langle -\eta \nabla_{\theta} L, \nabla_{\theta}L_1\rangle = \langle -\eta  \left(\dfrac{\nabla_{\theta}L_1}{||\nabla_{\theta}L_1||} + \dfrac{\nabla_{\theta}L_2}{||\nabla_{\theta}L_2||}\right), \nabla_{\theta}L_1\rangle $$

$$= -\eta \langle\dfrac{\nabla_{\theta}L_1}{||\nabla_{\theta}L_1||} + \dfrac{\nabla_{\theta}L_2}{||\nabla_{\theta}L_2||}, \nabla_{\theta}L_1\rangle = -\eta||\nabla_{\theta}L_1||(1+\cos\langle\nabla_{\theta}L_1, \nabla_{\theta}L_2\rangle)\leq 0$$

同理

$$\langle\Delta\theta, \nabla_{\theta}L_1\rangle\leq 0 $$

因此这个损失函数保证了每次 $\text{SGD}$ 都能够保证两个任务的损失均下降，这在其他损失函数中一般是不成立的。

实际上，我们研究的是基于性别-情绪的双任务情绪 $\text{EEG}$ 识别，可以看出这里的任务有主次之分，我们的主要目标是提升情绪识别的准确性，而辅助任务是性别分类，前面的广义平均损失函数能够根据 $\gamma$ 的设置倾向于关注更大的损失或是更小的损失，但我们在训练之前并不知道两个任务损失的大小关系，而如果只是给予情绪分类任务更大的权重，和前面讲述的一样，这种事先设定的方法似乎又太过于死板。我们选择从另一个角度出发，绕过定义损失的权重组合，在更加基本的方面切入。

我们对于任一损失函数 $L$，设计一个适合主次双任务的优化器，假设情绪任务的损失为 $L_1$，性别任务的损失为 $L_2$，根据泰勒展开，

$$L_i(\theta+\Delta \theta) = \nabla_{\theta}L_i\cdot\Delta\theta + o((\Delta\theta)^2)$$

因为 $\Delta\theta$ 一般取比较小的数，那么只要 $\langle\Delta\theta,\nabla_{\theta}L_i\rangle \leq 0$，就能够实现损失函数的下降。在主次任务中，我们当然希望 $\langle\Delta\theta,\nabla_{\theta}L_1\rangle$ 越小越好，并且保持 $\langle\Delta\theta,\nabla_{\theta}L_2\rangle\leq 0$，但为了防止模型通过大量增加 $\Delta\theta$ 使得 $\langle\Delta\theta,\nabla_{\theta}L_1\rangle$ 很小，我们还需要增加一个正则项，设 $\Delta\theta = -\eta h$，上述可以归结到一个优化问题：

$$\max_{h}\langle h,\nabla_{\theta}L_1\rangle - \dfrac{1}{2}||h||^2,\quad\langle h,\nabla_{\theta}L_2\rangle\leq 0$$

通过最优化方法，我们知道上述优化问题等价于

$$\max_{h}\min_{\lambda\geq 0}\langle h,\nabla_{\theta}L_1\rangle - \dfrac{1}{2}||h||^2+\lambda\langle h,\nabla_{\theta}L_2\rangle$$

因为----根据冯・诺依曼极大极小定理，上述优化问题等价于

$$\min_{\lambda\geq 0}\max_{h}\langle h,\nabla_{\theta}L_1+\lambda\nabla_{\theta}L_2\rangle - \dfrac{1}{2}||h||^2$$

假设上述问题的优化目标为 $P(\lambda, h)$，先求 $\max_{h}$，对 $P(\lambda, h)$ 关于 $h$ 求导，得

$$\dfrac{\partial P}{\partial h} = \nabla_{\theta}L_1+\lambda\nabla_{\theta}L_2 -h$$

而 $\dfrac{\partial^2 P}{\partial h^2} = -1 < 0$，故最大值点为 $h = \nabla_{\theta}L_1+\lambda\nabla_{\theta}L_2$，故优化问题化为

$$\min_{\lambda\geq 0}\dfrac{1}{2}||\nabla_{\theta}L_1+\lambda\nabla_{\theta}L_2||^2$$

解上述不等式，设优化目标为 $Q(\lambda)$，对 $\lambda$ 求导，得

$$Q'(\lambda) = \langle\nabla_{\theta}L_1+\lambda\nabla_{\theta}L_2,\nabla_{\theta}L_2\rangle = \langle\nabla_{\theta}L_1,\nabla_{\theta}L_2\rangle + \lambda||\nabla_{\theta}L_2||^2$$

当 $\langle\nabla_{\theta}L_1,\nabla_{\theta}L_2\rangle\geq 0$ 时，上式 $\geq 0$，原函数单调递增，考虑到 $\lambda\geq 0$，则解为 $\lambda = 0$， $h = \nabla_{\theta}L_1$；

当 $\langle\nabla_{\theta}L_1,\nabla_{\theta}L_2\rangle\leq 0$时，解为 $\lambda = -\dfrac{\langle\nabla_{\theta}L_1,\nabla_{\theta}L_2\rangle}{||\nabla_{\theta}L_2||^2}$，

$$h = \nabla_{\theta}L_1+\lambda\nabla_{\theta}L_2 = \nabla_{\theta}L_1 -\dfrac{\langle\nabla_{\theta}L_1,\nabla_{\theta}L_2\rangle}{||\nabla_{\theta}L_2||^2} \nabla_{\theta}L_2$$

为了充分发挥 $\text{GPU}$ 的并行算力，我们将上述两种情况合并为一个表达式，即

$$h = \nabla_{\theta}L_1 +\dfrac{\text{ReLU}(-\langle\nabla_{\theta}L_1,\nabla_{\theta}L_2\rangle)}{||\nabla_{\theta}L_2||^2} \nabla_{\theta}L_2$$

故

$$\Delta\theta = -\eta\left(\nabla_{\theta}L_1 +\dfrac{\text{ReLU}(-\langle\nabla_{\theta}L_1,\nabla_{\theta}L_2\rangle)}{||\nabla_{\theta}L_2||^2} \nabla_{\theta}L_2\right)$$

在这个优化器下，我们无需在意损失函数的形式，从“参数量变化如何最大地保证主任务损失下降”的角度切入，直接更新参数。




