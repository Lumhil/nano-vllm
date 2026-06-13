# 课程 14：ModelRunner 模型执行器

> **学习目标**：把 `ModelRunner` 放回 nano-vllm 的完整执行链路中理解；分清它和 `Scheduler`、`Attention`、`Sampler` 的职责边界；掌握 `prepare_prefill()` / `prepare_decode()` 两条输入组织路径；理解 `slot_mapping`、`cu_seqlens`、`context_lens`、`block_tables` 各自服务哪个 kernel；看懂 CUDA Graph、多 GPU SharedMemory 通信，以及为什么这层是“调度器和 GPU 之间的翻译层”。

---

## 零、先讲人话版

如果你觉得这一课容易读晕，通常不是你不会，而是几个层次很容易被揉在一起了：

- **请求（request）**：用户的一次生成任务。
- **Sequence**：请求在引擎里的运行时对象。
- **Scheduler**：决定“这轮谁上、每条序列这轮算多少 token”。
- **ModelRunner**：决定“这些已被选中的序列，怎样变成 GPU 能吃的张量和上下文”。
- **Attention**：真正读取这些张量和上下文，写 KV Cache，并做注意力计算。

先记住一句最重要的话：

> **ModelRunner 不决定“谁该算”，它只负责把 `Scheduler` 已经选好的序列，翻译成模型和 attention kernel 能执行的输入。**

把整条链压缩成人话，就是：

```text
用户请求
  -> 变成 Sequence
  -> Scheduler.schedule() 选出这轮的 seqs，并写好 num_scheduled_tokens / is_prefill
  -> ModelRunner.prepare_*() 把它们整理成 input_ids / positions / 各种上下文张量
  -> Attention.forward() 写入 KV Cache 并做注意力
  -> lm_head + sampler 产出 token
  -> Scheduler.postprocess() 更新缓存进度、必要时 append_token、判断是否结束
```

这里先提前澄清 4 个最容易混淆的点：

1. **`waiting` / `running` 是调度队列概念，不是 attention kernel 路径。**
   某条序列被 `schedule()` 从 `waiting` 挪到 `running`，不代表它当前这一步已经在 decode；它只是说明“这轮 prompt 如果算完了，后续就进入 decode 轮转”。

2. **`run(seqs, is_prefill)` 里的 `is_prefill` 是“这一整步走哪条执行路径”。**
   在 nano-vllm 里，一次 `schedule()` 返回的这一批，要么整批 prefill，要么整批 decode。

3. **`seq.is_prefill` 和 `run(..., is_prefill)` 不是一个层面的东西。**
   前者是 `Sequence` 自己的调度态字段，尤其会影响跨进程序列化；后者是 `ModelRunner.run()` 这一轮到底走 `prepare_prefill()` 还是 `prepare_decode()`。

4. **`prepare_prefill()` 不是“默认把整段剩余 prompt 全算完”。**
   它只会处理 `Scheduler` 这轮安排给它的那一段，也就是 `seq.num_scheduled_tokens` 指定的 chunk。

如果你读到后面开始晕，随时退回来，只盯这两句话：

- `Scheduler` 决定“这轮算谁、算多少”。
- `ModelRunner` 决定“这些 token 该怎么摆成 GPU 能执行的样子”。

---

## 一、先把边界讲清楚

### 1.1 `ModelRunner` 到底不做什么

很多人第一次读到这里，会把 `ModelRunner` 想成“大总管”。其实它不是。

它**不负责**：

- 决定 waiting 队列里谁优先
- 决定 running 队列里谁被抢占
- 决定 KV Cache block 分给哪条序列
- 决定一条序列这一轮该算 1 个 token 还是 256 个 token

这些都是 `Scheduler` + `BlockManager` 的职责。

### 1.2 它真正负责什么

`ModelRunner` 主要做 5 件事：

| 职责 | 核心问题 | 对应方法 |
|------|---------|---------|
| 初始化模型执行环境 | 模型怎么加载到 GPU，执行环境怎么准备好？ | `__init__()` / `warmup_model()` |
| 分配真实 KV Cache 张量，这里要和blockmanager做区别 | 显存里那块“装 K/V 的大仓库”怎么建？ | `allocate_kv_cache()` |
| 组织本轮输入 | 已选中的 `Sequence` 如何变成 `input_ids / positions / context`？ | `prepare_prefill()` / `prepare_decode()` |
| 执行模型前向 | 这轮走 eager 还是 CUDA Graph？ | `run_model()` |
| 多卡协同执行 | rank 0 如何把方法调用和序列信息同步给其他 rank？ | `call()` / `write_shm()` / `loop()` |

### 1.3 它和其他模块的关系

把几个相邻模块并排看，边界会清楚很多：

| 模块 | 它负责回答的问题 |
|------|----------------|
| `Scheduler` | 这轮哪些序列上场？每条序列这轮算多少 token？ |
| `BlockManager` | 这些序列占用哪些物理 block？是否还能 append / allocate？ |
| `ModelRunner` | 这些序列如何整理成 attention kernel 能吃的张量？ |
| `Attention` | 新算出来的 K/V 写到哪里？历史 K/V 从哪里读？怎么做注意力？ |
| `Sampler` | logits 如何按 temperature 采样出下一个 token？ |

所以，面试时如果被问“ModelRunner 的本质是什么”，一个很稳的回答是：

> **它是调度层和算子层之间的翻译层。**

---

## 二、初始化流程：让执行器进入“可跑”状态

源码路径：`nanovllm/engine/model_runner.py`

### 2.1 当前源码里的真实顺序

当前 `__init__()` 的关键顺序是：

```python
dist.init_process_group(...)
torch.cuda.set_device(rank)
torch.set_default_dtype(hf_config.dtype)
torch.set_default_device("cuda")

self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)

self.sampler = Sampler()
self.warmup_model()
self.allocate_kv_cache()

if not self.enforce_eager:
    self.capture_cudagraph()

torch.set_default_device("cpu")
torch.set_default_dtype(default_dtype)

if self.world_size > 1:
    if rank == 0:
        self.shm = SharedMemory(...)
        dist.barrier()
    else:
        dist.barrier()
        self.shm = SharedMemory(...)
        self.loop()
```

建议你把这个顺序记成：

```text
先把分布式和 CUDA 环境立起来
  -> 再建模型并加载权重
  -> 再 warmup
  -> 再算显存预算、分 KV Cache
  -> 再捕获 CUDA Graph
  -> 最后进入多卡协同运行
```

### 2.2 为什么 `warmup_model()` 要在 `allocate_kv_cache()` 之前

因为 `allocate_kv_cache()` 不是拍脑袋分配，而是要看：

- 当前 GPU 总显存
- 模型权重已经占了多少
- warmup 过程中前向传播的峰值显存是多少
- 当前 CUDA allocator 还保留了多少工作区

`warmup_model()` 的真实做法也不是“随便跑个零张量”，而是：

1. 先清空缓存、重置峰值统计；
2. 构造几条假的 `Sequence`；
3. 把它们的 `num_scheduled_tokens` 设成一个较大的 `seq_len`；
4. 直接调用一次 `self.run(seqs, True)`；
5. 再清一次缓存。

也就是说，warmup 是用“和真实执行路径尽量接近”的方式，把：

- Triton / CUDA kernel 编译
- CUDA 内存池初始化
- 前向峰值显存统计

这些成本前置掉。

### 2.3 `allocate_kv_cache()`：分配真正的 KV 仓库

当前实现会先算每个 block 需要多少字节：

```python
block_bytes = (
    2
    * num_hidden_layers
    * block_size
    * num_kv_heads
    * head_dim
    * dtype.itemsize
)
```

这里的 `2` 表示 K 和 V 两份缓存。

然后根据显存预算算出：

```python
config.num_kvcache_blocks = available_bytes // block_bytes
```

最后分配一个大张量：

```python
self.kv_cache = torch.empty(
    2,
    num_hidden_layers,
    num_kvcache_blocks,
    block_size,
    num_kv_heads,
    head_dim,
)
```

它的维度可以读成：

```text
[K_or_V, layer, physical_block, token_offset_in_block, kv_head, head_dim]
```

接着 `allocate_kv_cache()` 会把每层 attention 模块的：

- `module.k_cache`
- `module.v_cache`

直接指向这块全局 KV Cache 的某一层切片。这样后面 `Attention.forward()` 写缓存时，就不需要再去“找仓库”，它已经拿到了自己这层对应的那一片。

### 2.4 为什么 CUDA Graph 捕获放在后面

因为 graph capture 期间，模型前向里会真的访问：

- `slot_mapping`
- `context_lens`
- `block_tables`
- 以及 attention 内部已经挂到模块上的 `k_cache / v_cache`

如果 KV Cache 还没分好，capture 就没有可靠的执行环境。

---

## 三、Prefill 路径：`prepare_prefill()` 到底在准备什么

### 3.1 一句话先说透

`prepare_prefill(seqs)` 干的不是“把整条 prompt 都送进去”，而是：

> **把这些序列本轮被调度到的 prompt 片段，拼成 FlashAttention prefill kernel 需要的输入。**

这里“本轮被调度到的片段”是关键，因为 `Scheduler` 可能让一条长 prompt 分多轮 prefill。

### 3.2 当前源码里的核心变量

源码里最关键的 4 个量是：

```python
start = seq.num_cached_tokens
seqlen_q = seq.num_scheduled_tokens
end = start + seqlen_q
seqlen_k = end
```

它们分别表示：

| 变量 | 含义 |
|------|------|
| `start` | 这条序列前面已经有多少 token 的 KV Cache 了 |
| `seqlen_q` | 这轮真正要计算多少个新 token |
| `end` | 这轮算完以后，历史总长度会走到哪里 |
| `seqlen_k` | 这轮注意力能看到的总上下文长度，也就是 `end` |

注意这个设计的含义：

- `Q` 只对应“这轮新算的那一段”
- `K/V` 的逻辑上下文长度对应“到当前为止全部可见的历史”

这就是为什么 prefill 场景里经常会出现：

```text
seqlen_k > seqlen_q
```

### 3.3 这和“整段 prompt prefill”有什么区别

假设一条序列当前状态是：

```text
prompt 总长 = 1000
num_cached_tokens = 256
num_scheduled_tokens = 128
```

那这轮 `prepare_prefill()` 处理的不是剩下全部 `744` 个 token，也不是整个 `1000` 个 token，而是：

```text
start = 256
end = 384

这轮 input_ids = seq[256:384]
这轮 positions = [256, 257, ..., 383]
这轮 Q 长度 = 128
这轮 K 长度 = 384
```

也就是说，这轮只是“继续把 prompt 的第 256 到 383 个 token 算掉”。

这正是课程 12 里讲的部分 prefill / chunked prefill 思路，在 `ModelRunner` 里的真实落地。

### 3.4 `input_ids` 和 `positions` 怎么拼

当前实现会把所有序列本轮要算的 token 直接拼成一维：

```python
input_ids.extend(seq[start:end])
positions.extend(range(start, end))
```

所以 prefill 的输入布局不是 `[batch, seq_len]` 的 dense padding 形式，而是：

```text
多条变长序列的本轮 Q token，直接首尾相接拼成一条长向量
```

这正是 `flash_attn_varlen_func` 喜欢的输入组织方式。

### 3.5 `cu_seqlens_q` / `cu_seqlens_k` 是干什么的

因为大家都被拼到了一起，所以必须额外告诉 kernel：

- 每条序列的 Q 边界在哪里
- 每条序列的 K 边界在哪里

这就是 `cu_seqlens_*` 的作用。

例如 3 条序列这轮的 `seqlen_q` 分别是 `[4, 6, 3]`：

```text
cu_seqlens_q = [0, 4, 10, 13]
```

FlashAttention 就知道：

- 第 0 条序列的 Q 在 `[0, 4)`
- 第 1 条序列的 Q 在 `[4, 10)`
- 第 2 条序列的 Q 在 `[10, 13)`

而 `cu_seqlens_k` 用同样方法描述 K 的边界。

### 3.6 `slot_mapping`：新算出的 K/V 到底写去哪里

这部分特别重要。

`slot_mapping` 不是逻辑 token 下标，而是 KV Cache 里的**物理扁平 slot**：

```text
slot = physical_block_id * block_size + offset_in_block
```

`prepare_prefill()` 会只为本轮 `[start, end)` 这段 token 生成 slot：

```python
for i in range(start_block, end_block):
    ...
    slot_mapping.extend(range(slot_start, slot_end))
```

也就是说，`slot_mapping` 的长度会和这轮新算的 token 数严格对齐。

后面在 `attention.py` 里，`Attention.forward()` 一上来就会：

```python
store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

意思非常直接：

- 这轮新算出来的每个 `k / v`
- 按照 `slot_mapping`
- 写进对应的物理缓存位置

### 3.7 `block_tables` 为什么有时是 `None`

当前实现里：

```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
    block_tables = self.prepare_block_tables(seqs)
```

这说明只要“可见历史长度”大于“本轮新算长度”，就需要给 attention 一个 `block_tables`。

这类场景不只包括“跨请求前缀缓存命中”，还包括更广义的：

- 前缀缓存命中
- 上一轮已经 prefill 过一部分，这一轮继续 partial prefill

从 attention kernel 的视角看，它们本质上都是同一件事：

> **Q 只是一小段新 token，但 K/V 需要同时覆盖已经在 KV Cache 里的旧前缀。**

`prepare_block_tables(seqs)` 会把每条序列的 `seq.block_table` padding 到同样长度，并用 `-1` 填补无效位置，形成二维 `int32` 张量。

### 3.8 `set_context(True, ...)`：把元数据交给后面的层

prefill 最后会把这些元信息塞进全局上下文：

```python
set_context(
    True,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    slot_mapping,
    None,
    block_tables,
)
```

后面模型里的 attention 层、lm head 都会通过 `get_context()` 读取这些信息。

---

## 四、Decode 路径：`prepare_decode()` 为什么完全是另一套组织方式

### 4.1 Decode 的核心心智模型

Decode 阶段每条序列这一轮只做一件事：

> **拿“当前最后一个 token”作为输入，结合历史 KV Cache，预测“下一个 token”。**

所以它和 prefill 最大的差别是：

- prefill 一条序列这一轮可能算很多 token
- decode 一条序列这一轮只算 1 个 token

### 4.2 当前源码里的真实构造

源码是：

```python
for seq in seqs:
    input_ids.append(seq.last_token)
    positions.append(len(seq) - 1)
    context_lens.append(len(seq))
    slot_mapping.append(
        seq.block_table[-1] * self.block_size
        + seq.last_block_num_tokens - 1
    )
```

逐项解释：

| 字段 | 含义 |
|------|------|
| `input_ids.append(seq.last_token)` | 这轮输入就是当前最后一个 token |
| `positions.append(len(seq) - 1)` | 它在序列中的位置就是当前最后位置 |
| `context_lens.append(len(seq))` | 这轮注意力要看到的历史长度 |
| `slot_mapping.append(...)` | 这个 `last_token` 自己对应的 KV 应该写到哪里 |

### 4.3 最容易讲错的一点：decode 的 `slot_mapping` 指向什么

很多资料会把这里说成“给下一个 token 预留位置”，这在当前实现里是不准确的。

当前 decode 步里，模型输入的是：

```text
当前 last_token
```

因此，这一步算出来的 `k / v` 也是这个 `last_token` 自己的 K/V，它应当写到：

```text
当前最后一个已占用逻辑位置对应的物理 slot
```

也就是：

```python
seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1
```

可以这样理解：

1. 上一步 `postprocess()` 已经把新 token append 进 `Sequence` 了；
2. 这一轮 decode 读取的就是这个新的 `last_token`；
3. 所以它的 K/V 当然也该落在它自己的位置上；
4. 这一轮模型输出的 logits，用来预测**再下一个** token。

### 4.4 `context_lens` 和 `block_tables` 为什么 decode 必需

decode 不需要 `cu_seqlens_q / cu_seqlens_k`，因为每条序列的 Q 长度固定就是 1。

但它必须告诉 kernel 两件事：

1. **每条序列当前历史有多长**：`context_lens`
2. **这些历史 token 分布在哪些物理 block 里**：`block_tables`

所以 decode 最后会做：

```python
block_tables = self.prepare_block_tables(seqs)
set_context(
    False,
    slot_mapping=slot_mapping,
    context_lens=context_lens,
    block_tables=block_tables,
)
```

### 4.5 Prefill 和 Decode 的对照表

| 维度 | Prefill | Decode |
|------|---------|--------|
| 每条序列本轮输入 token 数 | `num_scheduled_tokens`，可能大于 1 | 固定为 1 |
| `input_ids` 来源 | `seq[start:end]` | `seq.last_token` |
| `positions` | `range(start, end)` | `len(seq) - 1` |
| 描述边界的方法 | `cu_seqlens_q / cu_seqlens_k` | `context_lens` |
| `slot_mapping` 含义 | 本轮新算 token 的所有物理写入位置 | 当前 `last_token` 的物理写入位置 |
| `block_tables` | 仅在需要读历史 KV 时准备 | 总是需要 |
| 对应 attention kernel | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |

---

## 五、Attention 层眼里，`ModelRunner` 其实在喂什么

如果你只看 `ModelRunner`，很容易觉得它在“拼很多杂乱的辅助张量”。但一旦把它和 `Attention.forward()` 对上，所有字段都会非常清晰。

源码路径：`nanovllm/layers/attention.py`

### 5.1 `Attention.forward()` 的真实顺序

当前实现是：

```python
context = get_context()

if k_cache.numel() and v_cache.numel():
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

if context.is_prefill:
    if context.block_tables is not None:
        k, v = k_cache, v_cache
    o = flash_attn_varlen_func(...)
else:
    o = flash_attn_with_kvcache(...)
```

这段代码非常值得反复看，因为它把 `ModelRunner` 的职责边界钉死了：

> **ModelRunner 不做 attention 数学本身，它主要负责把“本轮这些 token 的上下文关系和物理落点”描述清楚。**

### 5.2 这些上下文字段分别服务谁

| 字段 | 谁在使用 | 用来干什么 |
|------|---------|-----------|
| `slot_mapping` | `store_kvcache()` | 把新算出来的 K/V 写入正确物理位置 |
| `cu_seqlens_q / cu_seqlens_k` | `flash_attn_varlen_func` | 告诉 prefill kernel 变长序列边界 |
| `max_seqlen_q / max_seqlen_k` | `flash_attn_varlen_func` | 告诉 kernel 本批最大长度，便于 launch |
| `context_lens` | `flash_attn_with_kvcache` | 告诉 decode kernel 每条序列当前历史长度 |
| `block_tables` | 两条路径都可能用 | 把逻辑 token 位置映射到分页 KV Cache 的物理 block |

### 5.3 为什么 prefill 里有时也会去读 `k_cache / v_cache`

这点经常被忽略。

在 prefill 路径里，如果：

```python
context.block_tables is not None
```

当前实现会先：

```python
k, v = k_cache, v_cache
```

意思是：

- 本轮的 Q 只针对新 token
- 但 K/V 不再只来自本轮临时算出来的那一小段
- 而是要从整块 paged KV Cache 里把历史前缀也一起拿进来参与注意力

这正是 prefix cache / partial prefill 续算能够成立的关键。

### 5.4 为什么 `run()` 最后总能返回“每条序列一个 token”

一个容易疑惑的问题是：

> prefill 输入明明可能有很多 token，为什么 `run()` 最后仍然只返回 `len(seqs)` 个采样结果？

答案藏在 `ParallelLMHead.forward()` 里：

```python
if context.is_prefill:
    last_indices = context.cu_seqlens_q[1:] - 1
    x = x[last_indices].contiguous()
```

也就是说，在 prefill 场景下，lm head 会只取：

```text
每条序列本轮 Q 片段的最后一个位置
```

再对这些位置算 logits 并采样。

注意这带来一个很细的结论：

- 如果这轮 prefill 已经把整条 prompt 覆盖完了，那么这个采样结果会在 `postprocess()` 里真的 `append_token()`，成为首个生成 token；
- 如果这轮只是部分 prefill，那么这个采样结果虽然算出来了，但 `postprocess()` 会先继续补缓存，不会立刻把它接到序列后面。

---

## 六、`run_model()`：何时 eager，何时 CUDA Graph

### 6.1 当前实现的真实分支

当前 `run_model()` 不是简单的“prefill 永远 eager，decode 永远 graph”，而是：

```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    return self.model.compute_logits(self.model(input_ids, positions))
else:
    ...
    graph.replay()
    return self.model.compute_logits(graph_vars["outputs"][:bs])
```

所以真正的逻辑是：

- **prefill**：直接 eager
- **强制 eager 模式**：直接 eager
- **decode 但 batch 太大（> 512）**：也直接 eager
- 其余 decode：走 CUDA Graph

> - 这里的 eager，就是“即时执行模式”:Python 代码走到哪一层，PyTorch 就立刻把这一层对应的 CUDA kernel 发到 GPU 执行。Python 代码走到哪一层，PyTorch 就立刻把这一层对应的 CUDA kernel 发到 GPU 执行，所以它最灵活，形状变了也没关系，但每次都会有一串 kernel launch 开销
> - CUDA Graph：预录好整段 GPU 动作再回放,先录一次，再反复回放，启动开销小，适合 decode 这种每步都很像的小计算

### 6.2 为什么 decode 更适合 CUDA Graph

因为 decode 的特征是：

- 每条序列每轮只算 1 个 token
- 单步算量不大
- 但每层仍然要发起一长串 kernel

此时 CPU 发 kernel 的 launch 开销占比会变得更明显，Graph replay 的收益更高。

而 prefill 通常：

- token 数更多
- shape 波动更大
- 单步算量更重

所以 launch 开销相对没那么显眼，capture 的复用价值也更低。

### 6.3 当前 CUDA Graph 的捕获方式

`capture_cudagraph()` 里会先准备一套最大 buffer：

- `input_ids`
- `positions`
- `slot_mapping`
- `context_lens`
- `block_tables`
- `outputs`

然后构造一组批大小：

```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
```

也就是：

- 小 batch 精细捕获：1、2、4、8
- 大一些的 batch 按 16 对齐：16、32、48 ...

运行时，如果本轮 decode 的真实 batch size 是 `bs`，就选：

```python
next(x for x in self.graph_bs if x >= bs)
```

也就是**第一个不小于 `bs` 的 graph** 来回放。

### 6.4 Graph replay 前为什么要把上下文 buffer 重写一遍

因为 graph capture 期间绑定的是固定地址的 buffer。

所以实际 decode 时，不是重新分配新 tensor，而是把本轮的数据拷进旧 buffer：

```python
graph_vars["input_ids"][:bs] = input_ids
graph_vars["positions"][:bs] = positions
graph_vars["slot_mapping"].fill_(-1)
graph_vars["slot_mapping"][:bs] = context.slot_mapping
graph_vars["context_lens"].zero_()
graph_vars["context_lens"][:bs] = context.context_lens
graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
```

这里：

- `slot_mapping.fill_(-1)` 是把无效位置清掉
- `context_lens.zero_()` 是把未使用尾部清掉
- `block_tables` 则把本轮实际需要的那部分覆盖进去

这样 graph replay 才能在“地址不变”的前提下，吃到“内容更新”的输入。

---

## 七、`run()`：把准备、执行、采样串起来

### 7.1 当前 `run()` 的真实流程

```python
def run(self, seqs, is_prefill):
    input_ids, positions = (
        self.prepare_prefill(seqs)
        if is_prefill else
        self.prepare_decode(seqs)
    )
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()
    return token_ids
```

主线非常短：

```text
先准备输入
  -> 再跑模型
  -> 再采样
  -> 再清 context
```

### 7.2 为什么只有 rank 0 采样

因为在张量并行里：

- 各 rank 都要参与模型前向
- 但真正要把 logits 变成 token，只需要做一次

当前 `ParallelLMHead` 在 TP>1 时会把各 rank 的 vocab 分片 gather 到 rank 0，所以最终只有 rank 0 真正拿到完整 logits，采样也自然只在 rank 0 做。

### 7.3 `reset_context()` 为什么不能漏

`context` 是全局状态。

如果不在一次 `run()` 结束后清掉，下一轮可能会错误复用上一轮的：

- `is_prefill`
- `slot_mapping`
- `cu_seqlens`
- `block_tables`

这类“脏上下文串到下一轮”的 bug 非常难查，所以 `reset_context()` 是这里一个很关键但常被忽略的收尾动作。

---

## 八、多 GPU：`call()`、SharedMemory 和 `Sequence` 序列化

### 8.1 rank 0 和其他 rank 的分工

在 TP > 1 时：

- `rank 0` 负责和调度器所在主流程对接
- `rank > 0` 在初始化后直接进入 `loop()`

`loop()` 会一直做：

```python
while True:
    method_name, args = self.read_shm()
    self.call(method_name, *args)
    if method_name == "exit":
        break
```

也就是等待 rank 0 发来“要调用哪个方法、参数是什么”，然后在本地执行同一个方法。

### 8.2 `call()` 的本质：一个很轻量的本地 RPC

当前实现里：

```python
def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)
    method = getattr(self, method_name, None)
    return method(*args)
```

这意味着：

1. rank 0 先把方法名和参数写入 SharedMemory；
2. 通过 `Event` 唤醒其他 rank；
3. 所有 rank 各自在本进程里调用同名方法；
4. 模型内部的张量并行通信由 NCCL 自己完成。

所以 SharedMemory 传的不是大张量，而是：

- 方法名
- `Sequence` 对象
- `is_prefill` 之类的小元信息

### 8.3 当前 SharedMemory 协议长什么样

`write_shm()` 会把：

```python
[method_name, *args]
```

先 `pickle.dumps()` 成 bytes，然后：

- 前 4 字节写长度
- 后面写真实 payload

其他 rank 在 `read_shm()` 里反序列化回来。

这个设计朴素，但很适合 nano-vllm 这种教学实现：

- 它不追求跨节点
- 不追求复杂 RPC 语义
- 只追求“同机多进程把这轮调用同步起来”

### 8.4 `Sequence.__getstate__()` 的优化点

这一节最好和课程 11 连起来看。

当前 `Sequence` 的序列化逻辑是：

```python
def __getstate__(self):
    last_state = self.last_token if not self.is_prefill else self.token_ids
    return (
        self.num_tokens,
        self.num_prompt_tokens,
        self.num_cached_tokens,
        self.num_scheduled_tokens,
        self.block_table,
        last_state,
    )
```

这说明：

- **prefill** 时，要把完整 `token_ids` 发过去，因为 `prepare_prefill()` 需要切片 `seq[start:end]`
- **decode** 时，只发 `last_token` 就够了，因为 `prepare_decode()` 根本不需要整条 `token_ids`

这是一个很实用的优化，因为 decode 阶段真正需要的只是：

- 当前总长度 `num_tokens`
- prompt 长度 `num_prompt_tokens`
- 当前缓存进度 `num_cached_tokens`
- 本轮调度量 `num_scheduled_tokens`
- `block_table`
- `last_token`

而不是整条历史 token 列表。

### 8.5 一个细节：这里影响序列化的是 `seq.is_prefill`

注意，`__getstate__()` 分支依据的是：

```python
seq.is_prefill
```

而不是 `run(..., is_prefill)` 的参数。

这再次说明：

- `run(..., is_prefill)` 是这一步批量执行路径
- `seq.is_prefill` 是序列自己的调度态字段

两者通常会对齐，但概念层面不是一回事。

---

## 九、面试高频考点

### Q1：`ModelRunner` 和 `Scheduler` 的核心分工是什么？

**标准回答：**

`Scheduler` 决定“谁上场、每条序列这轮算多少 token、是否需要抢占、物理 block 是否够用”；`ModelRunner` 不做这些决策，它只接收已经被选中的 `seqs` 和 `is_prefill`，把它们组织成 `input_ids`、`positions`、`slot_mapping`、`cu_seqlens`、`context_lens`、`block_tables` 等张量，再驱动模型前向和采样。可以把 `ModelRunner` 概括为“调度层到 attention kernel 的翻译层”。

### Q2：为什么说 `prepare_prefill()` 不是“把剩余 prompt 一次性全算完”？

**标准回答：**

因为当前实现严格按 `Scheduler` 写进 `Sequence` 的 `num_scheduled_tokens` 来执行。`prepare_prefill()` 里真正的核心是：

```python
start = seq.num_cached_tokens
seqlen_q = seq.num_scheduled_tokens
end = start + seqlen_q
seqlen_k = end
```

也就是说，它只处理本轮 `[start, end)` 这段新 token。长 prompt 可能被拆成多轮 partial prefill，每轮只算一段，而不是默认整段剩余 prompt 全部送进模型。

### Q3：`prepare_prefill()` 和 `prepare_decode()` 的本质区别是什么？

**标准回答：**

prefill 是“每条序列本轮可能输入多个 token 的变长批处理”，所以它要拼一维 `input_ids`，再用 `cu_seqlens_q / cu_seqlens_k` 标边界；decode 是“每条序列本轮只输入 1 个 token”，所以它不需要 `cu_seqlens`，而是需要 `context_lens` 和 `block_tables` 告诉 kernel 每条序列已有多长历史、历史分布在哪些物理 block 里。两者对应的 attention kernel 也不同：prefill 走 `flash_attn_varlen_func`，decode 走 `flash_attn_with_kvcache`。

### Q4：`slot_mapping`、`block_tables`、`cu_seqlens`、`context_lens` 各自是干什么的？

**标准回答：**

- `slot_mapping`：把“这轮新算出来的 token”映射到 KV Cache 的物理写入位置，`store_kvcache()` 会直接按它写 K/V。
- `block_tables`：把逻辑 token 位置映射到物理 block，供 paged KV 读取历史缓存。
- `cu_seqlens_q / cu_seqlens_k`：只在 prefill 里用，告诉 `flash_attn_varlen_func` 变长拼接后每条序列的边界。
- `context_lens`：只在 decode 里用，告诉 `flash_attn_with_kvcache` 每条序列当前历史长度。

### Q5：为什么 warmup 要先于 KV Cache 分配？

**标准回答：**

因为 KV Cache 的可分配大小依赖“模型权重加载后还剩多少显存”和“真实前向传播峰值显存是多少”。`warmup_model()` 会触发 kernel 编译、内存池初始化，并让 `torch.cuda.memory_stats()` 记录更接近真实运行的峰值；只有在这之后，`allocate_kv_cache()` 才能比较准确地计算还能拿多少显存给 KV Cache。

### Q6：为什么 decode 更适合 CUDA Graph？

**标准回答：**

decode 每条序列每轮通常只算 1 个 token，单步计算量小，但每层依然有一串 kernel，CPU launch 开销占比会更明显；CUDA Graph 能把这一串 launch 打包成一次 replay，收益更高。prefill 的 token 更多、shape 波动更大、单步算量更重，所以 graph capture 的复用性和边际收益都更差。当前实现里也不是所有 decode 都强制用 graph，`enforce_eager=True` 或 `batch_size > 512` 时仍会直接 eager。

### Q7：nano-vllm 的多 GPU `ModelRunner` 是怎么协同的？

**标准回答：**

rank 0 负责接收调度器调用，然后通过 SharedMemory + `Event` 把 `[method_name, *args]` 广播给其他 rank；其他 rank 在 `loop()` 里阻塞等待，收到后在本地执行同名方法。也就是说，SharedMemory 负责传“方法调用和序列元信息”，真正模型内部的张量并行通信仍由 NCCL 完成。为了减少 IPC 开销，`Sequence.__getstate__()` 在 decode 场景下只发送 `last_token` 而不是完整 `token_ids`。

### Q8：为什么一条序列可能已经被挪进 `running`，但这一轮 `ModelRunner` 仍然走的是 prefill？

**标准回答：**

因为 `running` / `waiting` 是调度队列语义，而 `is_prefill` / `is_decode` 是这一轮执行路径语义，它们不是同一维度。当前 `schedule()` 在“这轮已经把 prompt 需要的 token 全部排进来了”时，就会先把序列从 `waiting` 挪到 `running`，表示它接下来具备进入 decode 轮转的资格；但这一轮真正交给 `ModelRunner.run(seqs, True)` 的仍然是 prefill 路径，直到这轮执行完、`postprocess()` 更新状态后，下一轮它才会按 decode 方式参与运行。

---

## 十、小结

这一课真正要记住的，不是零散 API 名字，而是下面这条主线：

```text
Scheduler 决定这轮谁上、上多少 token
  -> ModelRunner 把这些序列翻译成模型输入和 attention 上下文
  -> Attention 按 slot_mapping / block_tables 读写 paged KV Cache
  -> lm_head + sampler 产出每条序列一个候选 token
  -> Scheduler.postprocess() 决定这个 token 是否真的接到序列后面
```

如果要把 `ModelRunner` 浓缩成一句面试回答，可以直接说：

> **它是推理引擎的执行层入口，负责把调度结果组织成 GPU 可执行的张量与上下文，并在 eager / CUDA Graph、多卡协同、KV Cache 读写之间完成衔接。**

最后给你一个记忆口诀：

> `Scheduler` 管“谁上场”，`ModelRunner` 管“怎么摆”；  
> prefill 看 `cu_seqlens`，decode 看 `context_lens`；  
> `slot_mapping` 定写回位置，`block_tables` 定历史映射；  
> `Attention` 真正算注意力，`postprocess()` 才决定 token 是否落袋。
