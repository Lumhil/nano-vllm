# 第16课：CUDA Graph 优化

> **学习目标**：先用人话理解 CUDA Graph 到底优化了什么；分清 `eager / capture / replay / 固定地址 / 固定 shape / graph_bs / 占位槽` 这些容易混的概念；看懂 nano-vllm 当前源码里 `capture_cudagraph()` 和 `run_model()` 的真实逻辑；能够把“为什么 decode 适合 Graph、prefill 不适合”讲清楚。

---

## 零、先讲人话版

如果你第一次读 CUDA Graph，最容易被带偏的地方是：

> **它不是让模型“少算了”，也不是让矩阵乘法“数学上更快了”，它主要是在减少 CPU 反复向 GPU 发 kernel 的启动开销。**

也就是说，Graph 优化的重点不是：

- attention 的公式变了
- matmul 的 FLOPs 变少了
- KV Cache 的读写方式变了

而是：

- 原来每一步 decode 都要由 CPU 一次次把一长串 kernel 发给 GPU
- 现在先把这一整串 GPU 动作录下来
- 后面每步 decode 只需要 `graph.replay()` 一次

先记住一句最关键的话：

> **CUDA Graph 适合“每一步都长得差不多”的 decode，不适合 shape 和流程经常变化的 prefill。**

把 nano-vllm 里的主线压缩成人话，就是：

```text
启动时：
  -> 如果没有 enforce_eager
  -> 提前录几张常用 batch size 的 decode graph

运行时：
  -> 如果这一步是 prefill，直接 eager
  -> 如果这一步是 decode，就选一张已经录好的 graph
  -> 把真实输入拷进固定 buffer
  -> replay 整张图
  -> 取前 bs 条真正有效的输出
```

这里先提前澄清 6 个最容易混的点：

1. **Graph 优化的是 launch 开销，不是模型数学。**
2. **capture 是“录图”**，只发生在初始化阶段。
3. **replay 是“放图”**，发生在每一步 decode 时。
4. **Graph 录的是固定地址、固定 shape 的 GPU 动作序列。**
5. **`graph_bs` 里的数字，在 decode 语境下指“同时 decode 的序列条数”。**
6. **如果真实 batch 没有正好命中某张 graph，就会向上取整到最近一档，剩下的槽位当占位槽。**

如果你读到后面开始晕，随时退回来，只盯这 3 句话：

- `eager`：现算现发，每个 kernel 都由 CPU 现场提交。
- `capture`：先把一整步 decode 的 GPU 动作录下来。
- `replay`：后面重复用这张录好的图，减少 CPU 参与。

---

## 一、CUDA Graph 到底优化了什么

### 1.1 普通 eager 执行在做什么

在普通 PyTorch eager 模式下，一次模型前向不是“GPU 自动知道整条流水线”，而是：

1. Python 代码执行到某一层；
2. PyTorch 调用对应 CUDA kernel；
3. CPU 把这个 kernel 的启动命令发给 GPU；
4. 再继续下一层、下一个 kernel；
5. 整个前向里会重复很多次。

所以 **一次前向 = 很多次 kernel launch**。

如果计算本身很大，这些 launch 开销占比不显眼；但如果每一步都很小，而且一步里 kernel 很多，CPU 这边反复“发命令”的成本就会变得显眼。

### 1.2 CUDA Graph 在做什么

CUDA Graph 的核心思想是：

> **先把“这一整串 GPU 动作”录成图，后面直接重放。**

可以把它类比成：

- `eager`：每次演出都让导演逐句喊台词
- `graph`：先录成一条完整片段，后面直接播放

所以 Graph 的收益主要来自：

- 少了很多次 CPU 发 kernel 的动作
- 少了很多次驱动层调度开销
- GPU 更容易连续执行，不用老等 CPU 下一条命令

### 1.3 它不解决什么

Graph **不**直接解决这些问题：

- 模型参数太大
- KV Cache 不够
- attention 算法本身太慢
- 某个 matmul 算得不够高效

所以，面试时如果被问“CUDA Graph 为什么快”，一个稳的回答是：

> **它快在减少 launch overhead，而不是改变模型本身的计算量。**

---

## 二、为什么 decode 特别适合 CUDA Graph

### 2.1 Prefill 和 decode 的差别

把课程 14 的心智模型搬过来：

| 维度 | Prefill | Decode |
|------|---------|--------|
| 每条序列这一轮输入 token 数 | 可能很多 | 固定 1 个 |
| 总 token 数 | 波动很大 | 相对稳定 |
| shape | 经常变 | 很稳定 |
| 适不适合 Graph | 差 | 很适合 |

### 2.2 decode 为什么“长得更像模板”

在当前实现里，decode 路径每条序列只会放一个输入 token：

```python
input_ids.append(seq.last_token)
positions.append(len(seq) - 1)
```

这意味着 decode 每一步的结构非常像：

- 每条序列 1 个 token
- 每层还是同一套 attention / MLP 流程
- 变化主要集中在“这步有多少条序列一起 decode”

这正是 Graph 最喜欢的场景：

> **流程固定，只是内容在变。**

### 2.3 prefill 为什么不适合

prefill 的问题不是“不能算”，而是：

- 每条序列这轮可能算很多 token
- 不同请求长度差很多
- `cu_seqlens`、`max_seqlen_q / k`、token 总数都经常变化
- 还可能带 prefix cache / partial prefill 等分支

所以 prefill 的 **shape 和上下文结构都不够稳定**。

当前 nano-vllm 的选择很直接：

```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    return self.model.compute_logits(self.model(input_ids, positions))
```

也就是：

- prefill：直接 eager
- 强制 eager：直接 eager
- 超大 batch：也直接 eager

---

## 三、当前代码里的整体位置

源码路径：`nanovllm/engine/model_runner.py`

### 3.1 什么时候 capture

在 `ModelRunner.__init__()` 里：

```python
self.warmup_model()
self.allocate_kv_cache()
if not self.enforce_eager:
    self.capture_cudagraph()
```

也就是说：

1. 先 warmup；
2. 再分配 KV Cache；
3. 如果没有 `enforce_eager`，再录 graph。

这个顺序非常重要，因为 graph capture 期间：

- 模型已经要能正常前向
- attention 层要能访问 KV Cache
- graph 里读写的 buffer 也要和真实运行环境一致

### 3.2 什么时候 replay

在 `run_model()` 里：

```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    ...
else:
    ...
    graph.replay()
```

也就是说：

- Graph 只服务 decode 分支
- 而且只服务“已经 capture 过的那部分 batch size 范围”

---

## 四、`capture_cudagraph()` 到底在做什么

这一节最容易被写晦涩。先记住一句话：

> **录图不是“把 Python 函数录下来”，而是“把固定地址、固定 shape 的那一串 GPU 操作录下来”。**

这句话会直接决定后面所有设计。

### 4.1 先准备固定 buffer

当前实现一开始就分配：

```python
input_ids = torch.zeros(max_bs, dtype=torch.int64)
positions = torch.zeros(max_bs, dtype=torch.int64)
slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
context_lens = torch.zeros(max_bs, dtype=torch.int32)
block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
outputs = torch.zeros(max_bs, hf_config.hidden_size)
```

这些张量不是“临时随便造一下”，而是后面所有 graph 都会长期复用的固定工位。

你可以把它理解成：

```text
先在 GPU 上摆一排固定工位
  -> capture 时在这些工位上录图
  -> replay 时继续用这些工位
  -> 每次只改工位里的内容，不换工位地址
```

为什么要这样？

因为 Graph 绑定的是：

- 这块 tensor 的地址
- 这块 tensor 的 shape

不是“某个抽象的变量名”。

### 4.2 `max_bs` 是什么

当前代码：

```python
max_bs = min(self.config.max_num_seqs, 512)
```

它表示：

> **愿意为 CUDA Graph 预录的 decode 最大 batch size 上界。**

这里的 `batch size` 在 decode 语境下，指的是：

> **这一轮同时 decode 的序列条数。**

因为 decode 路径里每条序列只贡献 1 个 `last_token`，所以：

```text
input_ids.size(0) = 这步一起 decode 的序列数
```

### 4.3 为什么 `graph_bs = [1, 2, 4, 8] + 16 步进`

当前实现：

```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
```

这组数字表示：

- 预录 `graph-1`
- 预录 `graph-2`
- 预录 `graph-4`
- 预录 `graph-8`
- 再预录 `graph-16`, `graph-32`, `graph-48` ...

这里的 `graph-8` 不是 8 个 token 长度，不是 8 层，也不是 8 个 head，而是：

> **“同时处理 8 条 decode 序列”的那张图。**

### 4.4 为什么小 batch 要录得更细

如果不细录，小 batch 很容易浪费很多占位槽。

比如如果只录：

```text
graph-16
```

那真实来了：

```text
bs = 3
```

也只能用 `graph-16`，前 3 个槽位是真数据，后 13 个槽位是占位槽，浪费非常大。

一个直观的近似公式是：

```text
浪费比例 = (captured_bs - real_bs) / captured_bs
```

例如：

- `bs=3` 用 `graph-4`，浪费约 `25%`
- `bs=5` 用 `graph-8`，浪费约 `37.5%`
- `bs=9` 用 `graph-16`，浪费约 `43.75%`

这里的“浪费”主要是：

- 无效计算
- 显存带宽
- 这些占位槽对应的上下文读写

所以小 batch 录得细，是为了少浪费。

### 4.5 为什么不用把 `1..512` 全录一遍

因为那样也有成本：

- 初始化更慢
- graph 太多
- graph 管理更复杂
- 显存占用更重

所以当前实现做的是一个折中：

- 小 batch：细一点
- 大 batch：稀一点

### 4.6 为什么从大到小录

源码：

```python
for bs in reversed(self.graph_bs):
```

也就是先录最大 batch，再录更小 batch。

原因是后面有：

```python
if self.graph_pool is None:
    self.graph_pool = graph.pool()
```

这意味着第一张图会先建立 memory pool，后面的图继续复用这套 pool。

从大到小录的好处是：

> **先让 pool 见过最大需求，后面小图更容易复用，不容易反复扩张。**

### 4.7 warmup 为什么要在 capture 前再做一次

当前实现：

```python
outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
with torch.cuda.graph(graph, self.graph_pool):
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
```

这里的 warmup 不是多余的。

它的作用是让：

- 首次 kernel 初始化
- 某些底层 lazy setup
- 可能的一些缓存建立

先在 capture 外面完成掉。

这样正式录图时，图里记录的是“稳定执行路径”，而不是“首次运行特有的初始化噪音”。

### 4.8 capture 时为什么还要 `set_context(False, ...)`

当前代码：

```python
set_context(False, slot_mapping=slot_mapping[:bs],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs])
```

这是因为 decode 图里 attention 会通过全局 `context` 去读：

- `slot_mapping`
- `context_lens`
- `block_tables`

capture 时必须先把这些字段指向“固定 buffer 的那一段 view”，这样 replay 时只要改 buffer 内容，graph 里的 kernel 就能读到更新后的数据。

### 4.9 `graph_vars` 是什么

capture 结束后，代码会保存：

```python
self.graph_vars = dict(
    input_ids=input_ids,
    positions=positions,
    slot_mapping=slot_mapping,
    context_lens=context_lens,
    block_tables=block_tables,
    outputs=outputs,
)
```

它本质上就是：

> **后面 replay 时要反复往里写真实输入、再从里取真实输出的那套固定桥梁张量。**

---

## 五、`run_model()` 里怎么 replay

### 5.1 当前分支逻辑

当前实现：

```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    return self.model.compute_logits(self.model(input_ids, positions))
else:
    ...
    graph.replay()
    return self.model.compute_logits(graph_vars["outputs"][:bs])
```

主线可以读成：

```text
prefill / 强制 eager / 超大 batch
  -> 直接 eager

其余 decode
  -> 选择一张 graph
  -> 把真实数据写进 graph_vars
  -> replay
  -> 取输出前 bs 项
```

### 5.2 运行时怎么选图

当前代码：

```python
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
```

意思是：

> **从已经录好的档位里，找第一张“不小于真实 bs”的图。**

例如：

```text
graph_bs = [1, 2, 4, 8, 16, 32, ...]
```

如果这步真实：

```text
bs = 6
```

那就选：

```text
graph-8
```

也就是说，不会临时新录一个 `graph-6`，而是拿 `graph-8` 来复用。

### 5.3 为什么能拿 `graph-8` 跑 `bs=6`

因为 replay 前会只把前 `bs` 个槽位写成真数据：

```python
graph_vars["input_ids"][:bs] = input_ids
graph_vars["positions"][:bs] = positions
...
```

于是你可以把它想成：

```text
graph-8 的 8 个槽位里：
  0~5 放真实 6 条序列
  6~7 放占位槽
```

graph 依然会把 8 个槽位整包跑完，但最后：

```python
graph_vars["outputs"][:bs]
```

只取前 6 条真正有效的输出。

### 5.4 为什么要先清 `slot_mapping` 和 `context_lens`

当前代码：

```python
graph_vars["slot_mapping"].fill_(-1)
graph_vars["slot_mapping"][:bs] = context.slot_mapping
graph_vars["context_lens"].zero_()
graph_vars["context_lens"][:bs] = context.context_lens
```

这是因为固定 buffer 会被反复复用。

如果上一次是 `bs=8`，这一次是 `bs=4`，那后半段旧值如果不清掉，就可能污染下一次 replay。

所以这里：

- `slot_mapping.fill_(-1)`：先把整块 buffer 清成无效 slot
- `context_lens.zero_()`：先把整块长度清零

这样尾部占位槽就不会被误认为是真实样本。

### 5.5 `block_tables` 为什么只覆盖前面那块

当前代码：

```python
graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
```

意思是：

- 只覆盖真实样本的前 `bs` 行
- 每行只覆盖这轮真实需要的 block table 宽度

本质上和前面的 `input_ids[:bs]`、`positions[:bs]` 是同一套思路：

> **固定工位不换，但这轮只把真实需要的那部分内容写进去。**

### 5.6 `graph.replay()` 之后发生了什么

当调用：

```python
graph.replay()
```

发生的是：

- CPU 不再逐个 launch 那一长串 kernel
- 而是一次性让 GPU 重放已经录好的图
- 图里的 kernel 按 capture 时记录的顺序和依赖自动执行

所以这里快的根源，不是“单个 kernel 更快”，而是：

> **少了很多 CPU 侧重复 launch。**

### 5.7 为什么 `compute_logits()` 在 graph 外面

当前实现：

```python
graph.replay()
return self.model.compute_logits(graph_vars["outputs"][:bs])
```

也就是说：

- graph 里只录了 `self.model(...)` 这段主干前向
- `compute_logits()` 没录进去

你可以把它理解成一种工程折中：

- 先把最核心、最重复、最稳定的 decode 主干录成 graph
- 最后 logits 这一步保持在 graph 外面，保留一些灵活性

当前源码就是这样，不要把它想成“整条采样链都被录进 graph 了”。

---

## 六、`eager`、`capture`、`replay` 三个词别再混

### 6.1 `eager` 是什么

在这套代码里，`eager` 就是：

> **代码走到哪一层，PyTorch 就把那一层对应的 kernel 现场发给 GPU。**

优点：

- 灵活
- shape 怎么变都行
- 调试方便

缺点：

- 每一步都要反复 launch 很多 kernel

### 6.2 `capture` 是什么

`capture` 是：

> **初始化阶段，把某个固定 batch size 的 decode 前向在固定 buffer 上录成一张图。**

它不是每一步都发生。

### 6.3 `replay` 是什么

`replay` 是：

> **运行时拿已经录好的图，往固定 buffer 填入这一步真实数据，然后整张图重放。**

所以三者关系是：

```text
eager：现算现发
capture：先录图
replay：后放图
```

---

## 七、为什么还要 `reset_context()`

这点虽然在课程 14 讲过，但在 Graph 这里尤其重要。

`context` 不是长期模型状态，而是：

> **这一步前向临时挂在全局位置上的执行上下文。**

里面装的是：

- `is_prefill`
- `slot_mapping`
- `context_lens`
- `block_tables`
- prefill 时的 `cu_seqlens`

这些信息只对**当前这一步**有意义。

capture 阶段和 replay 阶段都在依赖这份 context 和固定 buffer 之间的契约，所以用完必须：

```python
reset_context()
```

否则下一轮可能吃到上一轮残留的：

- `slot_mapping`
- `context_lens`
- `block_tables`
- `is_prefill`

在 Graph 这种“固定地址反复复用”的模式下，这类脏状态会更难查。

---

## 八、`enforce_eager` 到底在干嘛

配置里：

```python
enforce_eager: bool = False
```

如果设成 `True`，效果非常直接：

- 初始化时不 capture graph
- 运行时全部走 eager

它的意义主要不是“更快”，而是：

- 调试方便
- 排查数值问题方便
- 遇到不适合 Graph 的环境时可以兜底

所以可以把它理解成：

> **强制关闭 CUDA Graph，全部退回普通即时执行。**

---

## 九、这篇最该记住的主线

如果你只记一条，记这个：

```text
启动时：
  -> 为若干常用 decode batch size 录图
  -> 图绑定固定 buffer 地址

运行时：
  -> 如果是 decode，就选一张不小于真实 bs 的图
  -> 把真实数据写进固定 buffer
  -> replay
  -> 只取前 bs 条有效输出
```

这条主线其实已经解释了：

- 为什么只适合 decode
- 为什么要 `graph_bs`
- 为什么会有占位槽
- 为什么要清 buffer
- 为什么 Graph 快的是 launch，不是数学

---

## 十、面试高频考点

### Q1：CUDA Graph 优化的核心收益是什么？

**标准回答：**

核心收益是减少 CPU 侧重复 kernel launch 的开销。它不是让 attention 或 matmul 的数学计算量变少，而是把原本一次前向里那一长串固定 GPU 操作先录成图，后续用 `graph.replay()` 一次性提交给 GPU，减少 CPU 调度成本。

### Q2：为什么 decode 比 prefill 更适合 CUDA Graph？

**标准回答：**

decode 每条序列每步只处理 1 个 token，shape 和执行路径都更稳定；prefill 则经常是变长、多 token、`cu_seqlens` 和总 token 数不断变化，还可能带 prefix cache 等分支，固定 shape 的 Graph 更难复用，所以当前 nano-vllm 只对 decode 使用 Graph。

### Q3：`graph_bs` 里的数字表示什么？

**标准回答：**

在当前 decode 语境下，`graph_bs` 里的数字表示“同时 decode 的序列条数”。例如 `graph-8` 不是 8 个 token 长度，也不是 8 层，而是“这一步有 8 条序列一起 decode”的那张 graph。

### Q4：为什么要向上取整到最近一档 graph？

**标准回答：**

因为当前实现只提前录了一组离散 batch size 的图，不会在运行时临时新录图。所以真实 `bs=6` 时，会找第一张 `>=6` 的图，也就是 `graph-8`。前 6 个槽位放真实数据，剩下 2 个槽位做占位，最后只取前 6 条输出。

### Q5：这里说的“浪费比例大”到底浪费的是什么？

**标准回答：**

浪费的是占位槽陪跑带来的无效计算和访存。比如真实 `bs=3` 却用了 `graph-4`，多出来的 1 个槽位也会跟着前向一起执行，只是最后结果会被丢弃。一个直观近似是：`(captured_bs - real_bs) / captured_bs`。

### Q6：为什么 capture 时必须使用固定 buffer？

**标准回答：**

因为 CUDA Graph 绑定的是 capture 时那批 tensor 的地址和 shape。replay 时不能重新换一套新 tensor，只能继续用同一批 buffer，把这一步真实数据拷进去，再重放图。这也是 `graph_vars` 存在的原因。

### Q7：为什么 `slot_mapping` 要先 `fill_(-1)`、`context_lens` 要先 `zero_()`？

**标准回答：**

因为固定 buffer 会被反复复用。如果上一轮 batch 更大，这一轮 batch 更小，尾部旧值如果不清理，就可能污染新的 replay。`fill_(-1)` 和 `zero_()` 本质上是在把尾部占位槽显式标成“无效”。

### Q8：`compute_logits()` 为什么放在 Graph 外面？

**标准回答：**

因为当前 nano-vllm 的 graph 主要录的是 `self.model(...)` 这段 decode 主干前向，而不是把 logits 计算和采样整条链全塞进去。这样做更灵活，也和当前源码保持一致。

### Q9：`enforce_eager=True` 的作用是什么？

**标准回答：**

它会关闭 CUDA Graph：初始化时不 capture，运行时也不 replay，而是全部走 eager。主要用于调试、排错和兜底，而不是为了性能。

### Q10：为什么 Graph 快的不是 GPU 算力本身？

**标准回答：**

因为 Graph 没有改变模型的数学结构，也没有减少 FLOPs。它主要减少的是 CPU 反复提交 kernel 的启动和调度成本，让 GPU 执行更连续，所以收益点在 launch overhead，而不是单个算子的理论算力。

---

## 十一、小结

这篇真正要记住的，不是零散 API 名字，而是下面这条主线：

```text
CUDA Graph = 先录 decode 主干前向
  -> 后面按固定工位反复 replay
  -> 少做 CPU launch
  -> 让 decode 更顺
```

再把几个最关键的判断句记住：

- Graph 优化 launch，不优化模型数学
- decode 稳定，所以适合 Graph
- prefill 变长，所以当前实现直接 eager
- `graph_bs` 是 decode 并发序列条数
- 没有正好命中就向上取整，剩下槽位做占位
- 固定 buffer 必须反复复用，所以脏状态一定要清

如果要把 CUDA Graph 浓缩成一句面试回答，可以直接说：

> **nano-vllm 的 CUDA Graph，本质上是在初始化阶段为若干常用 decode batch size 预录好固定 shape、固定地址的前向图，运行时把真实数据写进固定 buffer 后直接 replay，以减少 CPU 侧反复 kernel launch 的开销。**
