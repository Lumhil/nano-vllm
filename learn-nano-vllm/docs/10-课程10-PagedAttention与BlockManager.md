# 课程10：PagedAttention 与 BlockManager

> 如果只记一句话：`ModelRunner` 先分配一整块 KV Cache 大池，`BlockManager` 再负责回答两个问题:
>
> 1. 某条序列现在用了哪些物理 block？
> 2. 新来的序列能不能复用已有前缀 block？
>
> 这样做的结果是：KV Cache 不必按“每条序列一整段连续空间”来管理，而是像操作系统分页一样，按固定大小 block 动态分配、回收、共享。

## 本课目标

- 用一个具体例子看懂 `block_table` 到底是什么。
- 分清 `BlockManager`、`Scheduler`、`ModelRunner` 三者各自负责什么。
- 理解 prefill 时的 `can_allocate` / `allocate`。
- 理解 decode 时的 `can_append` / `may_append` / `hash_blocks`。
- 说清楚前缀缓存为什么只能复用“整块的、稳定的” KV。

## 一、先别看代码，先看图

### 1.1 为什么不能简单给每条序列一整段连续 KV

如果每条请求都预留一大段连续 KV 空间，会遇到三个老问题：

- **浪费**：短序列也要占一整段，内部碎片严重。
- **难复用**：请求结束后中间会留下洞，动态增长和回收都麻烦。
- **难共享**：多个请求有相同 system prompt 时，KV 不能轻松共用。

PagedAttention 的核心想法就是：

- 把 KV Cache 切成固定大小的 block。
- 每条序列只记录“自己的第 0 块、第 1 块、第 2 块”分别映射到哪几个物理 block。
- 如果两条序列前缀完全一样，就让它们共享同一个物理 block。

### 1.2 一个最小例子

真实代码里 `block_size` 默认是 `256`。为了方便画图，下面临时用 `4` 来举例。

假设某条序列有 9 个 token：

```text
token_ids = [A B C D E F G H I]
```

按 `block_size = 4` 切分后，逻辑上会变成 3 个 block：

```text
逻辑 block 0: [A B C D]
逻辑 block 1: [E F G H]
逻辑 block 2: [I]
```

如果它在物理 KV 池里拿到的 block 是 `3`、`7`、`15`，那么：

```text
block_table = [3, 7, 15]
```

含义是：

- 这条序列的第 0 个逻辑块，存在物理 block 3
- 第 1 个逻辑块，存在物理 block 7
- 第 2 个逻辑块，存在物理 block 15

注意这 3 个物理 block **不需要连续**。这就是 “paged” 的味道。

## 二、这课真正容易混的，是三层概念

很多人第一次看这里会晕，不是因为代码太难，而是把三层东西混成了一层。

| 层次 | 由谁管理 | 负责什么 |
|------|----------|-----------|
| **KV Cache 大张量** | `ModelRunner.allocate_kv_cache()` | 真正申请 GPU 显存，创建大池 |
| **block_table / free_block_ids / ref_count** | `BlockManager` | 管理“块号”和生命周期 |
| **slot_mapping** | `ModelRunner.prepare_prefill()` / `prepare_decode()` | 把“这次要写的 token”映射到具体写入槽位 |

一句话分工：

- **`ModelRunner` 管真实显存张量**
- **`BlockManager` 管逻辑块和物理块的对应关系**
- **`slot_mapping` 管这次前向时新 K/V 应该写到哪一个 slot**

如果这个分工没分清，后面 `block_table`、`slot_mapping`、`hash_blocks` 一定会看乱。

## 三、BlockManager 里到底有什么

源码位置：`nanovllm/engine/block_manager.py`

### 3.1 `Block`

每个 `Block` 都是一个物理 block 的元数据：

- `block_id`：这个 block 在大池中的编号
- `ref_count`：当前有多少条序列在引用它
- `hash`：当这个 block 被“完整填满”后，为它计算的内容哈希
- `token_ids`：这个 block 对应的 token 内容，主要用于防止哈希误命中

对应源码：

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
```

### 3.2 `free_block_ids` 与 `used_block_ids`

- `free_block_ids`：当前空闲、可分配的物理 block 队列
- `used_block_ids`：当前正在被某些序列使用的 block 集合

### 3.3 `hash_to_block_id`

这是“前缀缓存”的索引表：

```text
block 内容哈希  ->  对应的物理 block_id
```

有了它，系统就能快速回答：

```text
“我是不是以前见过完全一样的这一整块 token？”
```

### 3.4 一个容易忽略但很重要的点

**block 被释放后，它的内容并不会立刻被清空。**

在当前实现里：

- `deallocate()` 只会把 `ref_count` 降到 0，并把 block 放回 `free_block_ids`
- 真正等到这个 block 被重新分配、准备覆盖时，`_allocate_block()` 才会 `reset()`

这意味着：

- 某个 block 虽然当前“没人用”，但它的 KV 内容和哈希仍然还在
- 未来如果来了一个完全相同的前缀，它甚至可以被“重新激活”

这正是前缀缓存能工作的重要原因之一。

## 四、Prefill 阶段：第一次把序列放进 KV 池

Prefill 阶段要解决的问题是：

```text
这条序列一共有多少个逻辑 block？
其中前面有多少整块可以直接复用？
剩下的块要不要新分配？
```

### 4.1 `can_allocate(seq)` 干什么

核心逻辑：

1. 从序列开头开始，按 block 遍历
2. 只检查“完整 block”是否命中前缀缓存
3. 统计能复用多少个 cached block
4. 再看剩余空闲 block 是否够用

对应源码：

```python
def can_allocate(self, seq):
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1
```

这里有两个关键点：

- 它遍历的是 `range(seq.num_blocks - 1)`，也就是**默认只拿前面的完整 block 做复用判断**
- 命中哈希后还会再比较一次 `token_ids`，避免哈希碰撞造成误复用

### 4.2 为什么最后一个 block 通常不参与复用判断

因为最后一个 block 往往是未填满的：

```text
[A B C D] [E F]
```

第二块还会继续长，它的内容还不稳定。如果现在就把它当成“全局可复用块”，后面生成新 token 后内容又变了，索引就不可靠了。

所以当前实现的思路很朴素：

- **整块、稳定了**，再考虑加入缓存索引
- **未满块**，先不要作为全局缓存键

### 4.3 `allocate(seq, num_cached_blocks)` 干什么

`can_allocate()` 只是“算一算”，`allocate()` 才是真正写入 `seq.block_table`。

它分两段做事：

1. 前面 `num_cached_blocks` 个逻辑块，尝试复用已有物理 block
2. 后面的逻辑块，从空闲队列里申请新 block

对应源码：

```python
def allocate(self, seq, num_cached_blocks):
    assert not seq.block_table
    h = -1
    for i in range(num_cached_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id[h]
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            block.ref_count += 1
        else:
            block.ref_count = 1
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
        seq.block_table.append(block_id)
    for i in range(num_cached_blocks, seq.num_blocks):
        seq.block_table.append(self._allocate_block())
    seq.num_cached_tokens = num_cached_blocks * self.block_size
```

这段代码最值得记住的地方是：

- **命中且当前正在使用**：`ref_count += 1`
- **命中但当前在 free 列表里**：重新激活它
- **没命中**：新分配 block

也就是说，前缀缓存不只是“共享正在使用的块”，还可以“捞回刚释放但内容还没被覆盖的块”。

## 五、Decode 阶段：每次只追加 1 个 token

这一段是当前文档最容易让人误解的地方，建议直接记时间顺序。

### 5.1 decode 时，`len(seq)` 和 `num_cached_tokens` 不是一回事

在 `Scheduler.postprocess()` 里，顺序是：

1. `hash_blocks(seq)`
2. `seq.num_cached_tokens += seq.num_scheduled_tokens`
3. `seq.append_token(token_id)`

也就是说，模型刚采样出的新 token 会先被追加到 `seq.token_ids`，但它的 KV 还要等**下一轮 decode 前向**时才真正写进 cache。

所以在 decode 阶段，经常会看到：

```text
len(seq) = num_cached_tokens + 1
```

这个“多出来的 1 个 token”，正是下一轮要拿来算 K/V 的那个最新 token。

### 5.2 `can_append(seq)` 在检查什么

源码只有一行：

```python
def can_append(self, seq):
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

这个判断第一次看很怪，其实意思非常具体：

- 如果 `len(seq) % block_size != 1`，说明最新 token 仍然落在当前尾块里，不需要新 block
- 如果 `len(seq) % block_size == 1`，说明这个最新 token 是一个**新逻辑块的第 1 个 token**
- 既然它是新块的第一个 token，那么在本轮 decode 开始前，就必须先给它分一个新的物理 block

因为在 Python 里：

- `True` 会被当成 `1`
- `False` 会被当成 `0`

所以这行代码本质上等价于：

```python
if len(seq) % block_size == 1:
    return free_blocks >= 1
else:
    return True
```

### 5.3 `may_append(seq)` 只做一件事：必要时追加一个新 block

源码：

```python
def may_append(self, seq):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```

注意，这里**没有算哈希，也没有登记 `hash_to_block_id`**。

它只是在 decode 开始前，发现“最新 token 已经进入一个新逻辑块”，于是先把新物理 block 预留出来。

### 5.4 真正给 block 计算哈希的是 `hash_blocks(seq)`

源码：

```python
def hash_blocks(self, seq):
    start = seq.num_cached_tokens // self.block_size
    end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
    if start == end:
        return
    h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
    for i in range(start, end):
        block = self.blocks[seq.block_table[i]]
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block.update(h, token_ids)
        self.hash_to_block_id[h] = block.block_id
```

这段逻辑表达的是：

- 本轮刚刚写进 cache 的 token，也许让某个 block “恰好填满了”
- 只有在 block 被填满时，才把它登记进 `hash_to_block_id`

所以职责一定要分开记：

- `may_append()`：**需要新块时先分块**
- `hash_blocks()`：**块满了以后再登记哈希**

## 六、把 decode 时间线走一遍

继续用 `block_size = 4` 举例。

假设当前已有 8 个 token 已写入 cache：

```text
cached = [A B C D] [E F G H]
num_cached_tokens = 8
```

模型采样出了新 token `I`，并在 `postprocess()` 末尾执行了：

```text
append_token(I)
```

此时状态变成：

```text
token_ids          = [A B C D E F G H I]
len(seq)           = 9
num_cached_tokens  = 8
```

下一轮 decode 会发生什么？

### 第 1 步：`can_append()`

因为 `9 % 4 == 1`，说明 `I` 是新逻辑块的第一个 token，所以这轮开始前必须先有一个新 block。

### 第 2 步：`may_append()`

给这个新逻辑块分配物理 block，比如：

```text
block_table: [3, 7]  ->  [3, 7, 15]
```

### 第 3 步：`prepare_decode()`

它会把最新 token `I` 映射到新 block 的具体写入位置：

```text
slot = seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1
```

也就是“物理 block 15 的第 0 个槽位”。

### 第 4 步：attention 前向

这一步真正把 `I` 的 K/V 写进 KV Cache。

### 第 5 步：`hash_blocks()`

此时 block `[I]` 还没满，所以**不会**加入 `hash_to_block_id`。

直到这个块以后长成：

```text
[I J K L]
```

并在某一轮前向后恰好填满，`hash_blocks()` 才会把它登记成可复用前缀块。

## 七、Scheduler 和 ModelRunner 是怎么配合它的

这课如果只盯着 `BlockManager` 看，仍然会有点悬空。把调用链接上就顺了。

### 7.1 prefill 调度链

`nanovllm/engine/scheduler.py`

```python
if not seq.block_table:
    num_cached_blocks = self.block_manager.can_allocate(seq)
    ...
if not seq.block_table:
    self.block_manager.allocate(seq, num_cached_blocks)
```

含义：

- 对一个刚进系统、还没有 `block_table` 的序列
- 先算最多能复用多少前缀 block
- 再把整条序列需要的 block 号真正填到 `seq.block_table`

### 7.2 decode 调度链

```python
while not self.block_manager.can_append(seq):
    ...
seq.num_scheduled_tokens = 1
seq.is_prefill = False
self.block_manager.may_append(seq)
```

含义：

- decode 每轮只调度 1 个 token
- 如果这个 token 会落到新逻辑块里，先分一个新 block

### 7.3 前向结束后的收尾

```python
self.block_manager.hash_blocks(seq)
seq.num_cached_tokens += seq.num_scheduled_tokens
seq.num_scheduled_tokens = 0
seq.append_token(token_id)
```

含义：

- 先检查这轮写入是否让某个 block 填满
- 再更新“已有多少 token 已写入 cache”
- 最后把模型采样出的下一个 token 挂到序列尾部，等待下一轮 decode

### 7.4 `block_table` 最终如何变成真实写入地址

`ModelRunner.prepare_prefill()` / `prepare_decode()` 会把 `block_table` 进一步展开成 `slot_mapping`。

一句话理解：

```text
block_table 负责告诉你“这个逻辑块在哪个物理块”
slot_mapping 负责告诉你“这次这个 token 该写到哪个具体槽位”
```

所以：

- `BlockManager` 不直接写 K/V 张量
- 它只是提供 block 级映射
- 真正的写入发生在 `Attention.forward()` 里的 `store_kvcache(...)`

## 八、最容易答错的 5 个点

### 8.1 `block_table` 不是 KV 内容

它只是：

```text
逻辑 block 下标 -> 物理 block_id
```

不是 token 列表，更不是 K/V 张量本身。

### 8.2 前缀缓存是“块级复用”，不是“任意 token 粒度复用”

当前实现只会把**完整 block**登记为可复用对象。

未满块不稳定，所以不会作为全局缓存键。

### 8.3 `may_append()` 不负责哈希

这是这课最值得纠正的一点。

- `may_append()`：只负责在 decode 边界时追加物理 block
- `hash_blocks()`：负责把新填满的 block 计算哈希并登记到全局表

### 8.4 `ref_count == 0` 不等于“内容被擦除”

它只表示：

```text
当前没有序列在引用它了
```

但只要这个 block 还没被新的分配覆盖，它的历史内容依然可能被未来的前缀缓存重新命中。

### 8.5 哈希命中后还要比较 `token_ids`

因为 `xxhash` 是高性能非密码学哈希，理论上可能碰撞。

所以实现里不是“只看哈希就直接复用”，还会再检查一次 `token_ids` 是否真的一致。

## 九、面试时怎么用一句话讲清楚

可以这样说：

> PagedAttention 的核心是把 KV Cache 按固定大小 block 管理，用 `block_table` 建立“逻辑块到物理块”的映射，从而支持按需增长、减少碎片，并通过块级哈希实现前缀缓存；在 nano-vllm 里，`BlockManager` 负责块的分配/回收/共享，`ModelRunner` 负责把这些块映射成真实的写入槽位。

## 十、小结

把这课压缩成 4 句话：

1. **`ModelRunner` 先申请一个大的 KV Cache 池。**
2. **`BlockManager` 决定每条序列的逻辑块对应哪些物理 block。**
3. **prefill 时靠 `can_allocate` / `allocate` 复用完整前缀块。**
4. **decode 时靠 `may_append` 追加新块，靠 `hash_blocks` 在块填满后登记可复用哈希。**

如果这 4 句话已经顺了，这一课的主干就真的打通了。

## 下一课预告

下一课看 `Sequence` 与调度器时，建议重点把这几件事连起来：

- `block_table` 是谁写的
- `slot_mapping` 是谁算的
- `num_cached_tokens` 为什么总比 `len(seq)` 落后一拍

一旦这三点连上，连续批处理和 decode 流程就会清晰很多。
