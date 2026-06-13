# 课程07：RMSNorm 与激活函数

> 这一课更适合按“完整流程”来读。先知道一个 Decoder Layer 是怎么一步一步跑的，再分别拆开 `RMSNorm`、残差流和 `SwiGLU`，会顺很多。

## 先认几个英文词

如果你是第一次看这类代码，最容易卡住的往往不是公式，而是变量名和模块名。  
下面这些词先有个中文印象，后面读起来会轻松很多：

- `Decoder Layer`：解码器的一层，可以理解成大模型里反复堆叠的一个标准模块
- `主分支`：当前这层正在认真加工的那条路，对应代码里通常就是 `hidden_states`
- `hidden_states`：隐藏状态，也就是当前主分支里正在流动的特征表示
- `residual`：残差分支，也可以理解成“直通车”或“捷径分支”，它保留的是一份较早的表示，方便后面再接回来
- `input_layernorm`：进入注意力层之前的归一化层
- `self_attn`：self-attention，自注意力
- `post_attention_layernorm`：注意力计算之后的归一化层
- `MLP`：多层感知机，这里指前馈网络
- `FFN` / `Feed-Forward Network`：前馈网络，很多资料里和这里的 `MLP` 基本就是一回事
- `前馈`：可以先理解成“信息只从前往后算，不在这一小块里兜圈子”；在 Transformer 里，它通常表示“每个 token 各自做一次特征变换，不和别的 token 直接交互”
- `前向传播`：把输入一路算到输出的过程；注意，它和“前馈网络”不是一回事
- `SwiGLU`：一种带门控的激活结构
- `gate`：门，负责决定哪些信息通过
- `value`：内容分支，也就是被门控制的那部分信息
- `shortcut`：捷径分支，基本可以和这里的残差分支一起理解
- `token`：切分后的一个文本单位，很多时候你可以先把它粗略理解成“一个字或一个词片段”
- `dtype`：数据类型，比如 `float32`、`fp16`、`bf16`
- `float32`：32 位浮点数，精度更高
- `fp16` / `bf16`：16 位浮点数，更省显存，但精度更低
- `rsqrt`：reciprocal square root，倒平方根
- `kernel`：GPU 底层执行的一次计算任务
- `torch.compile`：PyTorch 的编译优化功能
- `Pre-Norm`：先做归一化，再进入子层的结构

有个小技巧：  
如果一个英文词一时记不住，就先把它当成“标签”，只抓它在流程里负责什么，别急着背名字。

## 先看完整流程图

先看 **后续层** 的常规路径，也就是 `residual` 已经存在时的情况。  
这是这节课最核心的一张图：

```text
输入:
  hidden_states（当前主分支里的表示）
  residual（残差分支 / 捷径分支）

      hidden_states -----+
                         |
      residual ----------+--> input_layernorm（输入归一化层）
                               |
                               | 先相加: tmp = hidden_states + residual
                               | 保存残差: residual = tmp
                               | 再归一化: hidden_states = RMSNorm(tmp)
                               v
                         self_attn（自注意力）
                               |
                               v
                      post_attention_layernorm（注意力后归一化层）
                               |
                               | 先相加: tmp = attn_out + residual
                               | 保存残差: residual = tmp
                               | 再归一化: hidden_states = RMSNorm(tmp)
                               v
                              MLP（前馈网络）
                               |
                               | gate_up_proj（一次投影出两份）
                               | -> 切成 gate / value（门 / 内容）两半
                               | -> silu(gate) * value
                               | -> down_proj（投回原维度）
                               v
输出:
  hidden_states（继续走主分支）
  residual（继续留给下一层做残差）
```

如果是 **第一层**，只有第一步略特殊，因为这时还没有历史 `residual`：

```text
输入 hidden_states（当前主分支表示）
   |
   +--> input_layernorm（输入归一化层）
   |      |
   |      | hidden_states = RMSNorm(hidden_states)
   |      | residual = 原始 hidden_states
   |      v
   +--> self_attn（自注意力） -> post_attention_layernorm（注意力后归一化层） -> MLP（前馈网络）
```

你可以先记一个总原则：

- 第一层：先 `RMSNorm`，再把原输入存成 `residual`
- 后续层：先和 `residual` 相加，再 `RMSNorm`

只要这件事没丢，整篇文档就不会再散。

## 为什么主分支和残差分支要一起进入

这是很多人第一次看 Transformer 代码时最容易疑惑的地方。

先把两条分支分清：

- 主分支：就是当前这层正在处理的那份表示，也就是代码里的 `hidden_states`
- 残差分支：就是一条“别急着改，我先留一份原表示”的捷径，也就是代码里的 `residual`

你可以把它想成：

- 主分支负责“加工新信息”
- 残差分支负责“保留旧信息”

那为什么要一起进入？

因为模型既想要：

- 用当前子层学到新的变化
- 又不想把之前已经有用的信息一下子改坏

所以最自然的办法就是：

```text
新的表示 + 旧的表示
```

也就是把两条路重新合在一起。

这样做有两个直接好处：

1. 旧信息不会轻易丢掉  
   就算当前这层学得一般，残差分支也能把之前那份表示带过去。

2. 深层网络更容易训练  
   因为梯度可以沿着残差这条“直通车”更顺地往前传。

所以你可以把残差连接理解成：

**这一层不是要把旧表示彻底推翻，而是在旧表示的基础上做增量修改。**

本项目里之所以看起来像“两条分支一起进入 `input_layernorm`”，本质上是因为作者把两步合并写了：

1. 先把主分支和残差分支相加
2. 再对相加结果做 `RMSNorm`

也就是代码里的：

```python
x = x.float().add_(residual.float())
...
x.mul_(torch.rsqrt(var + self.eps))
```

所以别把它理解成“两个输入同时做两套不同计算”，而要理解成：

**先汇合，再归一化。**

## 按流程看，每一步负责什么

### 1. `input_layernorm`

它负责把“即将进入注意力层”的张量整理好。

- 第一层时：直接做 `RMSNorm`，顺手把原输入保存成 `residual`
- 后续层时：先把当前主分支和历史残差相加，再做 `RMSNorm`

一句话理解：

**它负责把注意力层的输入准备好，并维护残差流。**

### 2. `self_attn`

它负责真正做注意力计算，让当前 `token` 去和上下文交互，提取依赖关系。

这里的 `token` 你先可以粗略理解成：

- 一个字
- 一个词
- 或者一个词片段

不用一开始抠得太细，你只要先知道它表示“模型处理文本时的基本单位”就够了。

一句话理解：

**它负责“看上下文，聚合信息”。**

### 3. `post_attention_layernorm`

注意力算完后，输出还要再和残差分支汇合一次。  
这一层又会重复一遍：

- 先加残差
- 再做 `RMSNorm`
- 更新 `residual`

一句话理解：

**它负责把注意力结果重新接回主干，并继续维持稳定的数值尺度。**

### 4. `MLP`

这里不是简单的两层线性层，而是 `SwiGLU` 风格的门控前馈。

这里最容易混淆的一点是：

- `前向传播`：指整个模型从输入算到输出
- `前馈网络`：指 Transformer 里注意力后面的那一小块网络，也就是这里的 `MLP`

所以“前馈”不是在说训练还是推理，  
也不是在说“现在正在做前向传播”，  
它说的是：**这一块模块本身的类型**。

这里的“前馈”可以专门多理解一句：

- 注意力层负责让一个 `token` 去看别的 `token`
- 前馈层负责让“当前这个 token 自己”把特征再加工一遍

也就是说：

**注意力负责信息交换，前馈负责特征加工。**

为什么叫“前馈”？

因为在这块小网络里面，信息是这样流动的：

```text
输入 -> 线性变换 -> 激活 / 门控 -> 线性变换 -> 输出
```

它就是一路往前算，不会像循环神经网络那样在这块里反复回流。  
所以名字里才会有“前馈”。

再结合 Transformer 的语境，你可以把它记成：

- 注意力：负责和别的 `token` 交换信息
- 前馈：负责当前这个 `token` 自己做深加工

它负责：

- 把特征投影到更高维空间
- 用一半通道当“门”
- 控制另一半通道哪些信息应该通过
- 再投影回隐藏维度

一句话理解：

**它负责做更强的非线性特征变换。**

### 5. 输出的 `hidden_states` 和 `residual`

这一层不会只输出一个张量，而是输出两条信息：

- `hidden_states`：给后续计算继续当主输入
- `residual`：继续当下一次残差汇合时的 `shortcut`（捷径分支）

一句话理解：

**主分支继续算，残差分支继续留。**

---

## 本课目标

- 先建立一个 Decoder Layer 的完整流程感。
- 知道 `RMSNorm` 和 `LayerNorm` 的核心差别。
- 看懂 `nanovllm/layers/layernorm.py` 里的 `rms_forward` 和 `add_rms_forward`。
- 理解为什么残差流要结合 `qwen3.py` 一起看。
- 看懂 `SiluAndMul` 为什么就是 `SwiGLU`。
- 顺手理解 `rsqrt`、`float32` 统计量、`torch.compile` 这些工程细节。

---

## 一、RMSNorm 到底在做什么

### 1. 先别背公式，先抓直觉

把最后一维的向量想成一组数：

```text
x = [x1, x2, x3, ..., xd]
```

模型在前向传播时，很怕这组数越来越大，或者越来越小。  
因为一旦尺度失控，后面的矩阵乘、激活函数、梯度传播都会变得不稳定。

所以归一化层最重要的任务之一，就是：

**把这组数的“整体量级”调到一个稳定范围。**

`RMSNorm` 做的正是这件事。

### 2. 它和 LayerNorm 有什么不同

`LayerNorm` 的思路是：

1. 先算均值。
2. 把每个元素减去均值。
3. 再按标准差缩放。

所以它会让向量“围绕 0 居中”。

`RMSNorm` 更简单：

1. 不减均值。
2. 只看这组数整体有多大。
3. 然后按这个尺度缩放。

所以你可以把它理解成：

- `LayerNorm`：先平移，再缩放。
- `RMSNorm`：只缩放，不平移。

### 3. 公式只需要记到这个程度

`RMSNorm` 先计算：

\[
\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}
\]

然后输出：

\[
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\mathrm{RMS}(x)}
\]

这里的重点只有两个：

- 分母是“均方根”，本质上是在测这组数的整体尺度。
- `\gamma` 是可学习参数，对应代码里的 `weight`。

### 4. 为什么大模型里常用 RMSNorm

因为在很多 LLM 里，模型更需要的是：

- 数值尺度稳定；
- 计算开销更低；
- 实现更简单。

而“必须把均值变成 0”这件事，在这些模型里往往不是最关键的。

所以工程上常见的结论是：

**RMSNorm 往往已经够用，而且更省。**

---

## 二、先对照代码看最简单的 `RMSNorm`

对应实现文件是 [nanovllm/layers/layernorm.py](../../nanovllm/layers/layernorm.py)。

先看最简单的 `rms_forward`：

```python
orig_dtype = x.dtype
x = x.float()
var = x.pow(2).mean(dim=-1, keepdim=True)
x.mul_(torch.rsqrt(var + self.eps))
x = x.to(orig_dtype).mul_(self.weight)
return x
```

把它翻成人话：

1. 先记住输入原来的 dtype。
2. 临时转成 `float32`。
3. 计算最后一维的平方均值。
4. 用 `rsqrt(var + eps)` 做缩放。
5. 转回原来的 dtype。
6. 乘上可学习参数 `weight`。

这里几乎每个英文都可以直接翻成中文：

- `dtype`：数据类型
- `float32`：32 位浮点数
- `weight`：可学习权重
- `eps`：一个很小的保护值，防止分母太小

这里最关键的一行是：

```python
var = x.pow(2).mean(dim=-1, keepdim=True)
```

它算出来的就是“均方”。

接着：

```python
x.mul_(torch.rsqrt(var + self.eps))
```

就是在做：

\[
x \leftarrow x \cdot \frac{1}{\sqrt{\mathrm{var} + \epsilon}}
\]

也就是“按整体尺度把向量缩放一下”。

---

## 三、为什么 `add_rms_forward` 这么容易把人看晕

### 1. 因为它不只是 Norm，它还在处理残差

很多人第一次看这个函数，会以为它只是“另一个 RMSNorm 写法”。

其实不是。

`add_rms_forward` 做了两件事：

1. 把主分支和残差分支加起来。
2. 再对相加后的结果做 RMSNorm。

代码如下：

```python
orig_dtype = x.dtype
x = x.float().add_(residual.float())
residual = x.to(orig_dtype)
var = x.pow(2).mean(dim=-1, keepdim=True)
x.mul_(torch.rsqrt(var + self.eps))
x = x.to(orig_dtype).mul_(self.weight)
return x, residual
```

直接按顺序解释：

### 2. 第一步：先加残差

```python
x = x.float().add_(residual.float())
```

这表示：

```text
x = x + residual
```

而且是在更高精度的 `float32` 上做加法，避免低精度误差太大。

### 3. 第二步：把“相加后的结果”保存下来

```python
residual = x.to(orig_dtype)
```

这里很关键。

它不是把旧的 `residual` 原封不动传下去，  
而是把“这一次相加后的结果”存下来，作为下一轮的残差流。

也就是说，函数返回的第二个值：

```python
return normalized_x, new_residual
```

其中：

- `normalized_x` 给当前子层继续用；
- `new_residual` 给后面的层当 `shortcut`（捷径分支）用。

### 4. 第三步：对相加结果做 RMSNorm

后面几行就和 `rms_forward` 一样了：

- 算均方；
- 用 `rsqrt` 缩放；
- 乘 `weight`。

所以这个函数的真正含义可以压缩成一句话：

**先把主路和残差路汇合，再把汇合后的结果做 RMSNorm，同时把“汇合后的未归一化张量”保存成新的残差。**

---

## 四、这段代码一定要结合 `DecoderLayer` 一起看

只读 `RMSNorm` 这个类，很容易不知道 `residual` 究竟从哪来、又到哪去。  
真正的上下文在 [nanovllm/models/qwen3.py](../../nanovllm/models/qwen3.py)。

核心代码是：

```python
if residual is None:
    hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
else:
    hidden_states, residual = self.input_layernorm(hidden_states, residual)

hidden_states = self.self_attn(positions, hidden_states)
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.mlp(hidden_states)
```

### 1. 第一层怎么走

当 `residual is None` 时，说明这是刚进入网络，还没有历史残差。

这时做的是：

```text
hidden_states = RMSNorm(hidden_states)
residual = 原始 hidden_states
```

也就是说：

- 归一化后的张量，送去做注意力；
- 未归一化的原始输入，先存起来，准备走残差支路。

### 2. 后续层怎么走

当 `residual` 已经存在时，调用的是：

```python
self.input_layernorm(hidden_states, residual)
```

也就是走 `add_rms_forward`。

逻辑变成：

```text
tmp = hidden_states + residual
residual = tmp
hidden_states = RMSNorm(tmp)
```

这就是为什么文档里一直强调：

**RMSNorm 在这个项目里不是孤立出现的，而是嵌在 `Pre-Norm`（先归一化、再进子层）的残差流中。**

### 3. 用一句话记住这件事

第一层：

```text
先 norm，再把原输入记成 residual
```

后续层：

```text
先和 residual 相加，再 norm，并把相加结果更新成新的 residual
```

如果你能把这一点讲明白，这一部分就算真正读懂了。

---

## 五、SwiGLU 其实没有名字看起来那么复杂

### 1. 先看代码做了什么

MLP 在 [nanovllm/models/qwen3.py](../../nanovllm/models/qwen3.py) 里是这样写的：

```python
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size,
    [intermediate_size] * 2,
    bias=False,
)
self.down_proj = RowParallelLinear(
    intermediate_size,
    hidden_size,
    bias=False,
)
self.act_fn = SiluAndMul()
```

前向里是：

```python
gate_up = self.gate_up_proj(x)
x = self.act_fn(gate_up)
x = self.down_proj(x)
```

这说明：

1. 先用一个线性层一次性投影出两份结果。
2. 把这两份结果交给 `SiluAndMul`。
3. 再用 `down_proj` 投回隐藏维度。

### 2. `SiluAndMul` 到底干了什么

对应实现是 [nanovllm/layers/activation.py](../../nanovllm/layers/activation.py)：

```python
x, y = x.chunk(2, -1)
return F.silu(x) * y
```

它的意思特别直接：

- 把最后一维切成两半；
- 前一半做 `SiLU`；
- 再和后一半逐元素相乘。

所以你可以把它理解成：

- `y` 是内容；
- `silu(x)` 是门；
- `门 * 内容` 就是门控。

### 3. 为什么叫 SwiGLU

因为它本质上是 GLU 家族的一种：

- `GLU`：一支当门，一支当内容；
- `SwiGLU`：门那一支用 `SiLU / Swish` 激活。

这里名字看起来长，其实可以拆开记：

- `Swi`：来自 `Swish/SiLU`
- `GLU`：Gated Linear Unit，门控线性单元

所以 `SwiGLU` 这个词，说白了就是：

**“用 SiLU 当门的 GLU”。**

如果只记最直观的理解，那就是：

**MLP 不再是“直接把所有特征都放行”，而是让模型自己学会控制哪些特征应该通过。**

### 4. 为什么 `gate_up_proj` 要一次输出两份

因为这样比写两个独立线性层更省事：

- 只做一次大矩阵乘；
- 更利于 kernel 融合；
- 更适合张量并行切分。

所以 `MergedColumnParallelLinear` 的直觉是：

**把原本两次投影，合成一次更宽的投影。**

---

## 六、几个工程细节，第一次读可以轻看

### 1. 为什么用 `torch.rsqrt`

因为：

```text
1 / sqrt(v) == rsqrt(v)
```

但在框架和硬件实现上，`rsqrt` 往往更符合常见优化路径。  
第一次阅读时，你只需要把它当成“倒平方根”即可。

### 2. 为什么先转 `float32`

因为 `fp16` / `bf16` 在做：

- 平方；
- 求均值；
- 开根号倒数；

这些操作时，更容易出现数值不稳定。

所以常见工程写法是：

1. 先把张量临时转成 `float32`；
2. 把统计量算完；
3. 再转回原 dtype。

这不是这份代码独有的技巧，而是混合精度里很常见的写法。

### 3. `eps` 是干什么的

就是防止分母太小，甚至接近 0，导致数值炸掉。

### 4. `torch.compile` 在这里有什么意义

`RMSNorm` 和 `SiluAndMul` 都属于“很短、但会被高频调用”的小模块。

给它们加上 `@torch.compile`，主要是为了：

- 融合小算子；
- 减少 kernel 启动次数；
- 提高重复执行时的效率。

但也要知道：

- 第一次编译有开销；
- 形状太动态时，可能带来重编译成本。

所以它是工程优化点，不是理解主逻辑的前置条件。

如果这些词陌生，可以先这样记：

- `compile`：编译优化
- `kernel`：GPU 一次底层计算任务

你不需要一开始就理解底层实现，只要知道它们都在讨论“怎么跑得更快”就行。

---

## 七、如果面试只让你讲 30 秒，可以这样说

### 1. RMSNorm vs LayerNorm

可以这样答：

> LayerNorm 会先减均值再按方差缩放；RMSNorm 不减均值，只按均方根缩放。RMSNorm 更简单，算得更轻，在很多 LLM 里已经足够稳定，所以现在很常见。

### 2. 为什么这里的 `RMSNorm` 要结合残差一起看

可以这样答：

> 在这个项目里，`RMSNorm` 不只是单独做归一化。后续层里它会先把主分支和 residual 相加，再做 norm，并把相加结果保存成新的 residual。这是整个 Pre-Norm 残差流的一部分。

### 3. `SwiGLU` 的一句话解释

可以这样答：

> `SwiGLU` 就是把 MLP 中间结果分成两半，一半经过 `SiLU` 作为门，去控制另一半内容分支的通过量，所以比普通 ReLU FFN 更有表达力。

---

## 八、把这节课压缩成一张心智图

```text
输入 x
  |
  |-- 如果没有 residual:
  |      hidden = RMSNorm(x)
  |      residual = x
  |
  |-- 如果有 residual:
         tmp = x + residual
         residual = tmp
         hidden = RMSNorm(tmp)

hidden -> Attention / MLP

MLP:
  gate_up_proj(x) -> 切成两半
  一半做 SiLU，当门
  一半做 value，当内容
  两者相乘
  down_proj
```

只要你脑子里能稳定保持这张图，这一课的大部分内容就不会再乱。

---

## 小结

这节课真正的主线并不复杂：

- `RMSNorm` 的目标是稳住尺度，不是强行把均值调成 0。
- 本项目里的 `RMSNorm` 要放进 `DecoderLayer` 的残差流里一起理解。
- `add_rms_forward` 的关键是“先加 residual，再 norm，再把相加结果存成新的 residual”。
- `SwiGLU` 的本质是“门控后的前馈网络”。
- `rsqrt`、`float32` 统计量、`torch.compile` 都是重要的工程细节，但它们不是阅读主线的第一优先级。

如果你能把上面这五句话复述出来，这篇文档就算真正读懂了。

## 下一课预告

下一课进入 **Qwen3 模型架构**。到那时你会发现，这节课最重要的价值不是记公式，而是看懂：

- `RMSNorm` 怎么嵌进整个 Decoder Layer；
- `SwiGLU` 怎么嵌进 MLP；
- 这些“小模块”如何组成完整的 Qwen3 解码器。
