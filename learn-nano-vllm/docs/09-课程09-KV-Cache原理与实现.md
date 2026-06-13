# 课程09：KV Cache 原理与实现

> 自回归解码每一步只新增一个 token，却若每次都从头重算整段序列的 K/V，会浪费 \(O(T^2)\) 级别的重复算力；KV Cache 把历史位置的 K、V 存下来，让解码步近似 \(O(T)\) 地增长。

## 本课目标

- 说清楚 **为何需要 KV Cache**（相对「每步全量重算」）。
- 背熟并会推导 **显存估算公式**（与 nano-vllm 张量形状一致）。
- 读懂 **`allocate_kv_cache`**：块数、可用显存、`torch.empty` 六维张量。
- 区分 **Prefill** 与 **Decode** 阶段对 KV Cache 的读写模式。
- 整理 **面试高频追问**（量化、多请求、GQA 对公式的影响）。

## 先用人话讲完

如果你只想先抓住主线，记住下面四句话就够了：

1. **KV Cache 存的不是“文本”，而是每一层 attention 算出来的 K 和 V。**
2. **有了 KV Cache，模型生成下一个 token 时，不用把前面所有 token 的 K/V 再算一遍。**
3. **Prefill 像“第一次把整段提示词录进去”，Decode 像“之后每次只追加 1 个新 token”。**
4. **nano-vllm 会先在 GPU 上开一整片 KV 内存池，再用 block 去切分和复用。**

换句话说，KV Cache 做的事非常朴素：**把“已经算过、后面还会反复用到的历史 K/V”留下来，别重复劳动。**

### 一个最小例子

假设输入是：

```text
我 喜 欢
```

现在模型要继续生成下一个 token。

- **没有 KV Cache**：
  - 为了算当前位置的 attention，你往往又得把 `我`、`喜`、`欢` 对应的 K/V 全部重新算一次。
- **有 KV Cache**：
  - 前三个位点的 K/V 之前已经算好并存起来了；
  - 这一步只需要为“当前新 token”算出新的 K/V，然后追加到缓存里；
  - 当前 query 直接去读“历史缓存 + 当前新增”即可。

所以它优化的重点不是“少算一点点”，而是**避免每一步都把整段历史重复做一遍**。

### 这篇文档真正想回答的三个问题

1. **为什么推理一定要 KV Cache？**
   因为自回归解码是“每次只新增 1 个 token”，历史 K/V 天然适合复用。
2. **KV Cache 到底占多少显存？**
   这就是那条「层数 × 长度 × KV 头数 × head_dim × 2」公式在回答的问题。
3. **在 nano-vllm 里，KV 是怎么放到 GPU 内存里的？**
   关键代码就是 `ModelRunner.allocate_kv_cache()`、`prepare_prefill()`、`prepare_decode()` 和 `Attention.forward()`。

## 核心概念

### 1. 注意力里重复计算从何而来

对长度 \(T\) 的序列，第 \(t\) 步若从零计算 attention，需要所有位置 \(1..t\) 的 K、V 参与。但 **第 \(t\) 步新增的只是位置 \(t\) 的 query**；位置 \(1..t-1\) 的 K、V 与上一步相比 **不变**（模型权重与已生成 token 固定时）。

因此可把 **过去所有步已算过的 K、V** 缓存在 GPU 上，本步只算当前 token 的 K、V 并 **追加** 到 cache，再与历史一起做注意力（常配合因果掩码或「只 attend 到过去」的实现）。

**省的是什么**：避免对历史 token 重复做 K/V 投影与（在部分实现中）重复写入中间结果。复杂度从「每步像做一次长序列 prefill」降为「每步常数级或线性于当前上下文」的增量更新（具体常数依赖 kernel 与 head 配置）。

### 2. 经典显存估算公式（与课程一致）

对每层、每个序列位置，需要存 **K** 与 **V** 两个张量，形状与 `num_kv_heads`、`head_dim` 相关。

粗算 **总 KV Cache 字节数**（与下面 nano-vllm 实现维度一致时可写作）：

\[
\text{KV\_bytes} \approx 2 \times L \times T \times H_{kv} \times D \times S_{\mathrm{dtype}} \times B
\]

这条公式算的是：**KV Cache 大概要占多少字节显存**。本质上它只是在数两件事：

```text
总共要存多少个元素
×
每个元素占多少字节
```

### 2.1 公式里的每个参数到底是什么意思

- **`2`**
  - 表示要存两份缓存：`K` 和 `V`。
  - 如果 attention 只需要一份历史信息，公式里就不会有这个 `2`，但标准自注意力需要同时保存 K 和 V。

- **`L`**
  - 表示模型层数，也就是 `num_hidden_layers`。
  - 因为 **每一层 attention 都有自己独立的 KV Cache**，所以显存会随层数线性增长。
  - 例如 Qwen3-0.6B 的 `L = 28`，就意味着同一批 token 的 KV 要存 28 层。

- **`T`**
  - 表示单条序列当前缓存了多少个 token。
  - 最好把它理解成 **“当前已经占住 KV Cache 的上下文长度”**，而不是死记为 `max_model_len`。
  - 例如 prompt 有 1000 个 token，模型又生成了 200 个 token，那么这条序列当前大约就是 `T = 1200`。
  - `T` 越大，KV Cache 线性变大，这也是长上下文特别吃显存的原因。

- **`H_{kv}`**
  - 表示 `num_key_value_heads`，也就是 **KV 头数**。
  - 这里最容易搞错：它通常 **不是** `num_attention_heads`。
  - 在 GQA 或 MQA 里，多个 query head 会共享同一组 KV head，所以缓存时只需要按 **真实 KV 头数** 计算。
  - 面试时要强调：**GQA 用 \(H_{kv}\) 而非 \(H\)**，这是与标准 MHA 公式的重要区别。
  - 如果做张量并行，还要按 rank 看本地头数，也就是代码里的 `num_key_value_heads // world_size`。

- **`D`**
  - 表示 `head_dim`，也就是每个 head 的向量维度。
  - 你可以把它理解成：**一个 KV head 不是一个数，而是一整段长度为 `D` 的向量。**
  - 在这个仓库的 Qwen3 配置里，`D = 128`。
  - `D` 越大，每个 token 每层要存的数据就越多。

- **`S_{\mathrm{dtype}}`**
  - 表示单个元素占多少字节。
  - 常见值：
    - `fp32 = 4` bytes
    - `fp16 = 2` bytes
    - `bf16 = 2` bytes
    - `int8 = 1` byte
  - 这个仓库默认用 `bfloat16`，所以这里通常取 `2`。
  - 工程上常说的 “KV Cache 量化”，本质上就是在降低这个因子。

- **`B`**
  - 表示并发序列数，可以先近似理解成 batch size。
  - 因为每条序列都要维护自己的历史 KV，所以并发请求越多，总 KV Cache 占用越大。
  - 不过这里要注意：`B × T` 是一个 **便于记忆的近似写法**，它默认所有序列长度都差不多。

### 2.2 一个更精确但不那么好背的写法

如果不同请求长度差很多，更精确的写法其实是：

\[
\text{KV\_bytes} \approx 2 \times L \times H_{kv} \times D \times S_{\mathrm{dtype}} \times \sum_i T_i
\]

这里的 \(\sum_i T_i\) 表示 **所有活跃序列当前占用的 token 总数**。

所以：

- `B × T` 适合快速估算和面试回答；
- `\sum_i T_i` 更接近真实工程里的资源占用。

### 2.3 这条公式到底在“数什么”

换一种方式理解：

- 每个 token
- 在每一层
- 要存 `H_{kv} × D` 个 K 元素
- 还要存 `H_{kv} × D` 个 V 元素

所以 **每个 token、每一层** 需要存：

\[
2 \times H_{kv} \times D
\]

个元素。

再乘上：

- `L` 层
- `B` 条序列
- 每条序列 `T` 个 token

最后再乘每个元素的字节数 `S_{\mathrm{dtype}}`，就得到了总字节数。

所以这条公式不是“背出来的”，而是顺着张量维度一项一项数出来的。

### 2.5 代入 nano-vllm 的真实配置看一眼

仓库自带的 `Qwen3-0.6B` 配置里，关键参数是：

- `num_hidden_layers = 28`
- `num_key_value_heads = 8`
- `head_dim = 128`
- `torch_dtype = bfloat16`，所以每个元素是 `2` 字节
- 默认 `block_size = 256`

如果先不考虑张量并行（`world_size = 1`），那么 **一个 block 在所有层上的 KV 总占用** 大约是：

```text
2 × 28 × 256 × 8 × 128 × 2 bytes
= 29,360,128 bytes
≈ 28 MB
```

这就是后面 `block_bytes` 那行代码在算的东西。

所以你可以把 `num_kvcache_blocks` 直接理解成：

```text
GPU 剩余可分给 KV Cache 的显存 / 每个 block 的成本
```

这比硬背公式更容易落地。

### 3. nano-vllm 中的「块」与全局张量

nano-vllm 不是为「每个请求 malloc 一块连续 KV」的简单模式，而是预先分配 **固定块数 × 块大小** 的大池，由 **BlockManager** 管理映射（下一课）。`allocate_kv_cache` 负责 **池子本体** 的显存与 **按层绑定** 到各 attention 模块的 `k_cache` / `v_cache` 视图。

### 4. Prefill vs Decode

- **Prefill（提示阶段）**：一次性处理 prompt，可并行算多个 token 的 Q/K/V，向 KV Cache **写入** 一段连续区间；计算形态常为「大张量、高并行」。
- **Decode（生成阶段）**：每步通常只处理 **1 个新 token**（或少量），对 KV Cache **增量追加**，计算形态常为「小 batch、内存带宽敏感」。

KV Cache 在 decode 阶段的收益最大：若无 cache，每步都要对全长重算历史 K/V，延迟爆炸。

---

## 源码解析：`ModelRunner.allocate_kv_cache`

下面与当前仓库的 [`nanovllm/engine/model_runner.py`](/home/qrh/project/nano-vllm/nanovllm/engine/model_runner.py:103) 一致。

```python
def allocate_kv_cache(self):
    config = self.config
    hf_config = config.hf_config
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
    config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
    assert config.num_kvcache_blocks > 0
    self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

### 显存余量：`total * gpu_memory_utilization - used - peak + current`

- **`mem_get_info`**：当前设备「空闲/总」显存。
- **`used = total - free`**：非空闲部分（含框架缓存等，语义以 CUDA 运行时为准）。
- **`peak` / `current`**：分配器统计的峰值与当前分配，用于修正「warmup 已分配但未必常驻」等差异。

整体意图：**在不超过用户设定利用率** 的前提下，估算还能容纳多少 **完整 KV block**。

你可以把这段逻辑理解成：

```text
先看看 GPU 还剩多少预算
再算一个 block 要花多少钱
最后决定最多能开几个 block
```

### `block_bytes` 的含义

单块、单层、单 rank 的 KV 双线？注意公式：

```text
2 * num_layers * block_size * num_kv_heads * head_dim * itemsize
```

这是 **一个 KV cache block slot** 占用的字节数：横跨 **所有层**（\(2 \times L\) 因子把 K/V 与层数都折进「每块成本」），从而 `num_kvcache_blocks = 可用字节 // block_bytes` 得到 **块槽位数**。

（若从维度上理解：`self.kv_cache` 第一维 2 为 K/V；第二维为 layer；块在第三、四维。`block_bytes` 把「一层一块」扩展为「所有层同一 block_id 的总占用」，与 `empty` 形状一致。）

### 六维张量 `self.kv_cache` 形状解析

```text
(2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
```

| 维 | 含义 |
|----|------|
| **2** | K 与 V 两个池（索引 0/1） |
| **num_hidden_layers** | 每层独立子张量，便于按层绑定模块 |
| **num_kvcache_blocks** | PagedAttention 的块个数 |
| **block_size** | 每块容纳的 token 槽位数 |
| **num_kv_heads** | 本 rank 上的 KV 头数（已除以 `world_size`） |
| **head_dim** | 每头维度 |

### 按层绑定

遍历 `self.model.modules()`，凡同时具有 `k_cache`、`v_cache` 的模块（各层 Attention），把 **该层** 对应 `layer_id` 的视图指过去：

```text
k_cache = kv_cache[0, layer_id]   # 形状去掉前两维中的 K/V 与 layer
v_cache = kv_cache[1, layer_id]
```

这样前向时模块直接写自己的层切片，无需每层单独 `torch.empty`。

---

## 把代码执行路径串起来看

只看 `allocate_kv_cache` 还是会有点悬空，因为它只解释了“内存池怎么开”，还没解释“谁往里写、谁从里读”。把下面四段代码串起来，KV Cache 就清楚很多了。

### 第一步：启动时先分配整块 KV 内存池

位置：[`nanovllm/engine/model_runner.py`](/home/qrh/project/nano-vllm/nanovllm/engine/model_runner.py:103)

`ModelRunner.allocate_kv_cache()` 做两件事：

1. 估算 GPU 还能放下多少个 KV block。
2. 创建一个六维大张量 `self.kv_cache`，再把每层 attention 的 `k_cache`、`v_cache` 指向自己那一层的切片。

这一步之后，每层 attention 都拿到了自己的“专属缓存视图”。

### 第二步：Prefill/Decode 先决定“这次写到哪个槽位”

位置：[`nanovllm/engine/model_runner.py`](/home/qrh/project/nano-vllm/nanovllm/engine/model_runner.py:129)

- `prepare_prefill()` 会把一批 prompt 中这次要处理的 token，映射到一串连续或分页后的 `slot_mapping`。
- `prepare_decode()` 会给每个序列只准备 **一个** 追加位置，也就是“下一个 token 应该写进哪个 slot”。

这里的 `slot_mapping` 你可以简单理解成：

```text
逻辑上的第几个 token
-> 物理上应该写进 KV Cache 的哪个格子
```

所以 **BlockManager 管的是“块号”**，而 **slot_mapping 管的是“最终写入地址”**。

### 第三步：Attention 前向时，先把新的 K/V 写入缓存

位置：[`nanovllm/layers/attention.py`](/home/qrh/project/nano-vllm/nanovllm/layers/attention.py:33)

`Attention.forward()` 里最关键的是这句：

```python
store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

意思是：

- 当前这一步模型已经算出了新的 `k`、`v`
- 现在按照 `slot_mapping`，把它们写到该层的 `k_cache`、`v_cache` 对应位置

这里调用的是 Triton kernel `store_kvcache_kernel`。它做的事其实很直接：**按槽位把新 K/V 拷贝进缓存池**。

### 第四步：写完之后，再决定这次 attention 怎么读缓存

还是在 [`nanovllm/layers/attention.py`](/home/qrh/project/nano-vllm/nanovllm/layers/attention.py:43)：

- 如果是 **Prefill**：
  - 走 `flash_attn_varlen_func(...)`
  - 一次处理多个 token
  - 如果存在前缀复用，还会结合 `block_table` 直接从已有 cache 里读历史块
- 如果是 **Decode**：
  - 走 `flash_attn_with_kvcache(...)`
  - 当前步通常只有一个 query
  - 直接读取整段历史 `k_cache` / `v_cache`

这也正是 KV Cache 最有价值的地方：**decode 时只新增一个 token，却能直接读完整历史缓存**。

### 一句话串起来

整个链路可以压缩成一句话：

```text
先分配 KV 大池
-> 再为本次请求算出写入槽位
-> attention 前向时把新 K/V 写进去
-> 然后立刻拿历史 cache 做注意力计算
```

---

## 图解

### KV 随时间追加（概念）

```text
step 0:  [K0]
step 1:  [K0, K1]
...
step t:  [K0 ... Kt]
```

V 同理；实现上落在 block 池的离散块中，而非简单向量追加。

### Prefill vs Decode（对比）

```text
Prefill:  一次写入多个 token 的 KV（并行度高）
Decode:   每步写入 1 个 token（带宽敏感，强依赖 cache）
```

### 与块管理器的关系（预告）

```text
allocate_kv_cache  -->  一大块物理池
BlockManager         -->  逻辑块 <-> 序列 token 的映射表
```

---

## 面试考点

### 为何公式里用 `num_kv_heads` 而不是 `num_heads`

GQA/MQA 下多个 query 头共享 KV 头，缓存只存 **物理 KV 头**。

### 张量并行如何进入公式

每 rank 只存 **本分片** 的 KV 头：`num_kv_heads // world_size`（代码变量名 `num_kv_heads` 已除过）。

### 量化 KV Cache（追问）

INT8/FP8 等降低 \(S_{\mathrm{dtype}}\)，但需反量化或专用 kernel；公式结构不变，改 **每元素字节数** 与 **精度损失** 讨论。

### `assert config.num_kvcache_blocks > 0`

配置过大 `max_model_len`、过高利用率、或显存过小时可能为 0；工程上要报错提示用户调参。

---

## 常见面试题

1. **只有 KV Cache，没有 Q Cache？**  
   每步只需求当前位置的 Q；历史 Q 不参与当前步 attention 的「与过去 token 匹配」时不需要存全历史 Q（标准自回归解码）。

2. **KV Cache 会和梯度一起反传吗？**  
   推理路径无梯度；训练时通常用 FlashAttention 等变体，cache 语义不同。

3. **块大小 `block_size` 影响什么？**  
   粒度 vs 碎片：小块更灵活但元数据开销大；大块可能浪费尾部空间。

4. **为什么要 `warmup_model` 再分配 KV？**  
   先触发峰值分配与 cudnn/cublas 工作区，再扣减 `peak`，使块数估计更接近真实运行（与源码顺序一致）。

5. **batch 变大时 KV 显存线性涨吗？**  
   多序列各占槽位，总占用随并发序列数增加；具体是否线性取决于是否共享前缀、是否分页等。

---

## 小结

KV Cache 避免对历史 token 重复计算 K/V，是低延迟解码的核心；显存可按「层 × 长 × KV 头 × 头维 × 精度 × 2」估算；nano-vllm 用 **单大六维张量 + 按层视图** 管理池，`allocate_kv_cache` 根据 GPU 余量与块成本计算 **可用块数**。Prefill 批量写、Decode 增量写，二者对系统瓶颈（算力 vs 带宽）影响不同。

## 下一课预告

下一课 **PagedAttention 与 BlockManager**：操作系统分页类比、xxhash 前缀块复用、`allocate`/`may_append` 与引用计数，把「块池」真正接到「多请求并发」上。
