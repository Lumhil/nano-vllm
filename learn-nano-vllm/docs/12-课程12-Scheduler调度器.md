# 课程 12：Scheduler 调度器

> **学习目标**：深入理解 nano-vllm 调度器的完整工作机制；掌握 waiting / running 双队列模型；理解 prefill 优先调度策略和 decode 轮转调度的设计思想；掌握抢占（preempt）机制的实现细节；理解 postprocess 后处理流程；能够在面试中对比分析不同调度策略的优劣。

---

## 零、先讲人话版

如果上一课里 `Sequence` 是“请求的档案袋”，那这一课里的 `Scheduler` 就是“每一轮决定谁上 GPU、上去算多少、算完后怎么收尾的人”。

先不要急着钻每个 `while`。先记住一轮 `LLMEngine.step()` 里，调度器只做 3 件事：

```text
1. schedule()     -> 这轮选哪些 seq，走 prefill 还是 decode
2. model_runner   -> 真正把这轮选中的 token 送进模型
3. postprocess()  -> 更新缓存/追加 token/判断是否结束
```

把 `Scheduler` 的工作拆开，本质上只有下面 4 个问题：

1. **waiting 里的新请求，这轮能不能先做 prefill？**
2. **如果不能做 prefill，那 running 里的老请求，这轮谁来 decode？**
3. **如果 decode 需要新 block，但显存不够，该抢占谁？**
4. **这轮算完以后，是继续等下一轮，还是已经结束？**

读这篇时，建议一直盯下面这 5 个变量：

- `waiting`：还没彻底进入 decode 轮转的序列队列
- `running`：已经进入 decode 轮转的序列队列
- `num_scheduled_tokens`：这一轮实际要算多少 token
- `num_cached_tokens`：已经有 KV Cache 的 token 有多少
- `is_prefill`：这一轮走 prefill 路径还是 decode 路径

一个最关键、也最容易漏掉的事实是：

> **一次 `schedule()` 并不一定意味着“这条请求会立刻生成一个新 token”。**

如果这轮只是 **部分 prefill**，那这轮做的事情只是“继续把 prompt 算完并写入 KV Cache”，`postprocess()` 会更新缓存进度，但**不会**立刻 `append_token()`。

带着这个前提再看后面的源码，理解会顺很多。

---

## 一、调度器的角色与职责

### 1.1 为什么需要调度器

大模型推理引擎通常同时服务多个用户请求。每个请求在不同时间到达，需要不同长度的 prompt 处理和不同数量的 token 生成。GPU 的显存和算力是有限的，不可能同时处理所有请求。

调度器的核心职责：

1. **决定每一步执行哪些序列**（选取 + 排序）
2. **区分 prefill 和 decode 阶段**（不同阶段的资源特征截然不同）
3. **管理 KV Cache 资源**（通过 BlockManager 分配 / 释放物理 block）
4. **处理资源不足**（抢占低优先级序列，释放 block 给高优先级序列）
5. **后处理**（追加 token、判断终止条件、清理已完成序列）

### 1.2 类比理解

把调度器想象成餐厅的领班：

| 餐厅 | 推理引擎 |
|------|---------|
| 领班 | Scheduler |
| 等位顾客 | waiting 队列 |
| 正在用餐的顾客 | running 队列 |
| 餐桌（有限资源） | KV Cache blocks |
| 安排入座 | schedule() - 分配 block |
| 催促买单让位 | preempt() - 抢占 |
| 上菜 + 确认是否用完 | postprocess() |

### 1.3 调度器在系统中的位置

```
LLMEngine.step()  ← 引擎的"心跳"
    │
    ├── scheduler.schedule()             ← 选出本步参与的序列 + 分配 block
    │       ↓ 返回 (seqs, is_prefill)
    ├── model_runner.run(seqs, is_prefill)  ← 前向推理
    │       ↓ 返回 token_ids
    └── scheduler.postprocess(seqs, token_ids, is_prefill)  ← 更新缓存、必要时追加 token、判断终止
```

每个 `step()` 调用一轮 schedule → run → postprocess，构成引擎的**心跳循环**。调度器是这个循环的**第一个环节**，决定了整个系统的效率。

---

## 二、Scheduler 的数据结构

### 2.1 构造函数

源码路径：`nanovllm/engine/scheduler.py`

```python
class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
```

### 2.2 关键属性详解

| 属性 | 类型 | 说明 |
|------|------|------|
| `max_num_seqs` | int | 单步最多处理的序列数，限制批大小。防止同时处理太多请求导致每个请求延迟过高 |
| `max_num_batched_tokens` | int | 单步最多处理的 token 总数，限制计算量。决定 GPU 单步最大工作负载 |
| `eos` | int | EOS token ID，用于判断生成是否自然终止 |
| `block_size` | int | KV Cache block 大小，和 `Sequence.block_size` 保持一致，便于调度阶段做长度与 block 数换算 |
| `block_manager` | BlockManager | KV Cache 物理 block 管理器，负责分配和回收 block |
| `waiting` | deque[Sequence] | 等待队列：存放尚未开始或被抢占的序列 |
| `running` | deque[Sequence] | 运行队列：存放正在参与推理的序列 |

### 2.3 为什么使用 deque 而非 list

`deque`（双端队列）vs `list` 的性能对比：

| 操作 | deque | list |
|------|-------|------|
| 左端添加 `appendleft` | O(1) | O(n) |
| 左端弹出 `popleft` | O(1) | O(n) |
| 右端添加 `append` | O(1) | 均摊 O(1) |
| 右端弹出 `pop` | O(1) | O(1) |
| 随机访问 `[i]` | O(n) | O(1) |
| 中间删除 `remove` | O(n) | O(n) |

调度器的核心操作是**从队列头部取出序列**和**在头/尾部添加序列**，这些操作在 deque 上都是 O(1)。

### 2.4 两个队列的关系

```
                   schedule()
  ┌──────────┐    选中并分配    ┌──────────┐
  │ waiting  │──────────────→ │ running  │
  │  队列    │                │  队列    │
  └──────────┘                └──────────┘
       ↑                         │  │
       │      preempt()          │  │ postprocess()
       │      资源不足回退        │  │ 完成后移除
       └─────────────────────────┘  │
                                    ↓
                              序列完成，从 running 移除
```

---

## 三、schedule() 方法完整流程

### 3.1 方法签名与返回值

```python
def schedule(self):
    # 返回: (scheduled_seqs: list[Sequence], is_prefill: bool)
```

- `scheduled_seqs`：本步参与推理的序列列表
- `is_prefill`：True 表示本步执行 prefill，False 表示执行 decode

### 3.2 核心设计原则：Prefill 优先

nano-vllm 的调度策略是**prefill 优先**：只要 waiting 队列中有序列可以调度，就优先处理它们（即使 running 队列中有序列在等待 decode）。

**为什么 prefill 优先？**

1. **用户体验**：新请求需要先完成 prefill 才能开始生成，prefill 越快，用户等待首个 token 的时间越短（Time To First Token, TTFT）
2. **计算效率**：prefill 是计算密集型（compute-bound），可以高效利用 GPU 算力
3. **避免饥饿**：如果 decode 优先，新请求可能长时间无法得到处理

### 3.3 Prefill 调度阶段（当前源码）

```python
def schedule(self):
    # ---------- prefill ----------
    scheduled_seqs = []
    num_batched_tokens = 0
    while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.waiting[0]
        remaining = self.max_num_batched_tokens - num_batched_tokens
        if remaining == 0:
            break
        if not seq.block_table:
            num_cached_blocks = self.block_manager.can_allocate(seq)
            if num_cached_blocks == -1:
                break
            num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
        else:
            num_tokens = seq.num_tokens - seq.num_cached_tokens
        if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
            break
        if not seq.block_table:
            self.block_manager.allocate(seq, num_cached_blocks)
        seq.num_scheduled_tokens = min(num_tokens, remaining)
        num_batched_tokens += seq.num_scheduled_tokens
        if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, True
    # ---------- decode 见下一小节 ----------
```

**逐行解析**：

1. **prefill 永远先看 `waiting[0]`**：这是 FCFS 的核心。队头的大请求过不去，后面的小请求也不会被跳过。
2. **`remaining`** 表示这一步还剩多少 token 预算。它不是“序列长度上限”，而是“本轮还能再算多少 token”。
3. **`can_allocate(seq)` 返回的不是 bool，而是 `num_cached_blocks` 或 `-1`**：
   - `-1`：空闲 block 不够，当前队头请求这轮不能启动；
   - 非负整数：说明能启动，而且前面有多少整块前缀可以直接复用。
4. **`if not seq.block_table`** 的语义是“这条请求还没真正拿到物理 block”：
   - 第一次进入 prefill 时会走这里；
   - 如果只是上一步做了**部分 prefill**，它还留在 waiting 中，但 `block_table` 已经有值了，下一步会从 `else` 分支续算。
5. **`num_tokens` 是“这轮还没缓存、理论上需要继续算的 token 数”**：
   - 第一次进来时，要减去前缀缓存命中的整块；
   - 续算部分 prefill 时，要减去 `num_cached_tokens`。
6. **只有队头请求允许部分 prefill**：
   `if remaining < num_tokens and scheduled_seqs: break`
   这句的意思是：如果当前请求太长，只有当它是这轮第一个请求时，才允许“先切一段出来算”；否则直接停下，不再往后看。
7. **`num_scheduled_tokens` 是本轮最关键的输出之一**：
   它告诉 `ModelRunner` 这轮到底要吃多少 token，也告诉 `postprocess()` 这轮完成后缓存要前进多少。
8. **只有当整段未缓存 token 都算完时，序列才真正从 waiting 进入 running**：
   `seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens`
   如果条件不成立，说明这轮只是部分 prefill，序列仍留在 waiting。
9. **只要这轮调度到了任何 prefill 序列，就立即返回 `(..., True)`**：
   所以 nano-vllm 在单次 `schedule()` 调用里依然是“prefill / decode 二选一”，不会在同一批里混跑两种路径。

### 3.4 一个很容易漏掉的事实：`WAITING` 不等于“完全没算过”

在当前实现里，`WAITING` 有两种来源：

1. 刚创建的新请求，确实还没开始算；
2. 做了**部分 prefill**，但 prompt 还没全部算完，所以还不能进入 running。

也就是说，某个 `seq` 仍然在 `waiting` 中，并不代表它一定是“冷启动”。它可能已经：

- 拿到了 `block_table`
- 累积了 `num_cached_tokens`
- 只差后半段 prompt 还没跑完

这也是为什么后面的 `postprocess()` 里会先更新缓存进度，再决定要不要真的追加新 token。

### 3.5 Decode 调度阶段（当前源码）

```python
    # ---------- decode（仅当上面 prefill 未返回时执行）----------
    while self.running and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)
                break
        else:
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
    assert scheduled_seqs
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False
```

**逐行解析**：

1. **decode 只在“这轮一个 prefill 都没调度成功”时才会发生**。
2. **`self.running.popleft()`**：从 running 队头取序列，形成 FIFO 轮转。
3. **`can_append(seq)`** 检查的是：这条序列这一步能不能继续 decode。
   它最多只会要求“再多 1 个 block”，因为一次 decode 只处理 1 个 token。
4. **内层 `while not can_append`**：如果当前序列继续不下去，就不断抢占 running 队尾的其他序列，直到：
   - 腾出了足够空间；
   - 或者 running 里已经没人可抢，只能抢占自己。
5. **进入 `else` 才表示当前序列真的能参加这轮 decode**：
   - `seq.num_scheduled_tokens = 1`
   - `seq.is_prefill = False`
   - `may_append(seq)` 在必要时补 1 个新 block
6. **`extendleft(reversed(scheduled_seqs))`** 会把这轮选中的序列按原顺序塞回 running 队头。
   这个写法的效果是：下一轮还会从这些序列开始继续轮转，而不会打乱相对顺序。
7. **`assert scheduled_seqs`** 的含义是：既然已经走到 decode 分支，理论上至少得成功调度出一个序列；否则说明容量或状态出现了不满足预期的情况。

### 3.6 while...else 语法详解

这是 Python 中一个不太常见但非常优雅的语法：

```python
while condition:
    ...
    if some_check:
        break
else:
    # 只在 while 正常结束（condition 变为 False）时执行
    # 如果 break 退出，则不执行
    ...
```

在 decode 调度中：
- 如果成功释放了足够空间（`can_append` 变为 True，while 正常结束）→ 执行 else，将序列加入调度
- 如果没有可抢占的序列了（break 退出）→ 不执行 else，该序列不参与本步

---

## 四、抢占机制（Preempt）

### 4.1 为什么需要抢占

在 decode 阶段，每个序列每步生成一个 token，需要在 KV Cache 中写入一个新的 KV 对。如果某个序列的最后一个 block 已满，就需要分配新的物理 block。但物理 block 是有限的——如果已经用完，就需要**抢占（preempt）**其他序列来释放 block。

### 4.2 preempt() 源码

```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

四步操作：

1. **状态回退**：将序列状态从 RUNNING 改为 WAITING
2. **路径重置**：将 `is_prefill` 重新设为 `True`，表示后续恢复时要按 prefill 路径重新准备输入
3. **释放资源**：通过 BlockManager 释放该序列占用的所有物理 block
4. **重新排队**：使用 `appendleft` 将序列放到 waiting 队列**头部**

### 4.3 为什么用 appendleft

被抢占的序列使用 `appendleft`（放到头部）而非 `append`（放到尾部），是为了**保证公平性**：

- 被抢占的序列已经等待了一段时间，不应该排到新请求后面
- 放到头部确保它们在下一轮调度中**优先被重新调度**
- 这避免了"饥饿"问题——某个序列被反复抢占却永远无法完成

### 4.4 抢占策略：LIFO

```python
self.preempt(self.running.pop())  # pop() 从尾部取出
```

nano-vllm 使用 **LIFO（Last In First Out）** 抢占策略——最后加入 running 队列的序列最先被抢占。

**为什么选择 LIFO？**

1. **最小化浪费**：最后加入的序列可能才刚开始生成，抢占它浪费的计算量最少
2. **资源释放量大**：如果最后加入的序列有较长的 prompt，它的 block 较多，释放后更可能满足空间需求
3. **简单高效**：deque.pop() 是 O(1) 操作

### 4.5 抢占的代价

抢占不是免费的：

```
序列 A 在 running 中，已完成 prefill（1000 token），生成了 200 个 token
  → 占用 ceil(1200/256) = 5 个 block

抢占 A：
  → 释放 5 个 block（KV Cache 全部丢失）
  → A 回到 waiting 队列
  → 重新调度 A 时，需要重新做 1000 token 的 prefill
  → 浪费了之前的全部计算
```

这就是 nano-vllm 的简化设计——**recompute 策略**：被抢占的序列需要完全重新计算。在 vLLM 的完整版本中，还有 **swap 策略**：将 KV Cache 从 GPU 交换到 CPU 内存，避免重复计算。

### 4.6 抢占自己的场景

```python
if self.running:
    self.preempt(self.running.pop())
else:
    self.preempt(seq)  # 抢占自己
    break
```

当 running 队列为空（所有其他序列都被抢占了），但当前序列仍然无法获得足够的 block 时，只能**抢占自己**。这种情况意味着：

- 当前序列再往前走一步还需要 1 个新 block
- 但即使把其他 running 序列都抢掉，空闲 block 依然不够

这通常意味着系统总 block 容量偏紧，当前序列单独占着资源也无法继续扩张。抢占自己后，序列回到 waiting 队列，等待后续条件变化。

---

## 五、Decode 调度的资源检查

### 5.1 can_append vs can_allocate

| 方法 | 调用时机 | 检查内容 |
|------|---------|---------|
| `can_allocate(seq)` | Prefill 调度 | 返回可复用的前缀 block 数，或在空间不足时返回 `-1` |
| `can_append(seq)` | Decode 调度 | 这一步 decode 如需新 block，当前是否有足够空闲 block 支撑 |

### 5.2 may_append 的条件性分配

```python
def may_append(self, seq):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```

这里看起来有点反直觉，为什么条件是 `== 1` 而不是 `== 0`？

关键在于 decode 阶段的输入是 **`last_token`**。这一步不是“为下一个还没生成出来的 token 分配位置”，而是“为当前 `last_token` 的 KV 写入位置做准备”。

以 `block_size = 256` 为例：

- prefill 结束后，prompt 长度是 256
- 第一次采样后，`append_token()` 把第 257 个 token 加进 `token_ids`
- 下一轮 decode 时，`len(seq) == 257`，这时第 257 个 token 属于一个**新的逻辑 block**
- 所以在这轮真正计算它的 KV 之前，就必须先补 1 个新的物理 block

因此，`len(seq) % block_size == 1` 恰好表示“当前 `last_token` 是新 block 的第一个 token”，这时才需要分配新 block。

### 5.3 Decode 调度的完整流程图

```
开始 decode 调度
    │
    ▼
从 running 队列头部取序列 seq
    │
    ▼
can_append(seq)?
    │
    ├── Yes ──→ may_append(seq) → 加入 scheduled_seqs → 取下一个序列
    │
    └── No ──→ running 中有其他序列？
                    │
                    ├── Yes ──→ preempt(running.pop()) → 重新检查 can_append
                    │
                    └── No ──→ preempt(seq) → break（本步无法调度此序列）
```

---

## 六、postprocess 后处理

### 6.1 源码解读

```python
def postprocess(self, seqs, token_ids, is_prefill):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue
        seq.append_token(token_id)
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

### 6.2 处理流程

对于本步参与推理的每个序列：

1. **把这轮新完成的 block 做哈希登记**：`hash_blocks(seq)`，为后续 prefix cache 复用做准备。
2. **推进缓存进度**：`num_cached_tokens += num_scheduled_tokens`。
3. **清空本轮调度计数**：`num_scheduled_tokens = 0`。
4. **如果这是 prefill，而且 prompt 还没全部算完，就先 `continue`**：
   这一步只是在继续铺 KV Cache，**不会生成新 token**。
5. **只有“完整 prefill 结束”或“decode”这两种情况，才会真正 `append_token(token_id)`**。
6. **追加完 token 后，再检查是否终止**：
   - 条件 1：遇到 EOS 且没有忽略 EOS
   - 条件 2：completion token 数达到 `max_tokens`
7. **如果终止**：
   - 标记为 `FINISHED`
   - 释放 block
   - 从 running 队列移除

### 6.3 终止条件的逻辑表达式

```python
(not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens
```

用真值表分析：

| ignore_eos | token == EOS | completion == max | 是否终止 |
|-----------|-------------|------------------|---------|
| False | True | - | **是**（自然结束） |
| False | False | True | **是**（达到上限） |
| False | False | False | 否 |
| True | True | False | 否（忽略了 EOS） |
| True | True | True | **是**（达到上限） |
| True | False | True | **是**（达到上限） |
| True | False | False | 否 |

核心逻辑：`max_tokens` 是硬性上限，无论如何不能超过；`EOS` 是软性终止，可以被 `ignore_eos` 覆盖。再补一句：如果当前只是**部分 prefill**，那么这轮根本还走不到这个终止判断。

### 6.4 为什么用 `self.running.remove(seq)` 而非 `popleft`

因为完成的序列不一定在队列头部——批次中任何位置的序列都可能先完成。`remove()` 根据值查找并删除，时间复杂度 O(n)，但由于 running 队列通常很短（几十个序列），这不是性能瓶颈。

### 6.5 postprocess 只处理本轮被调度的序列

更准确地说，`postprocess` **不会去遍历整个 waiting 队列**，它只处理本步参与推理的 `seqs`。

但要注意一个细节：在**部分 prefill** 场景里，某个 `seq` 这轮已经参与了推理，却**仍然留在 waiting 队列里**。因此：

- `postprocess` 处理的是“本轮被调度的序列”
- 而不是“只处理 running 里的序列”

这也是为什么它先更新缓存进度，再决定这轮到底要不要追加新 token。

---

## 七、add 方法

### 7.1 源码

```python
def add(self, seq: Sequence):
    self.waiting.append(seq)
```

极其简单：将新序列加入 waiting 队列尾部。FIFO 顺序保证先到的请求先被处理。

### 7.2 add 的调用时机

```python
# LLMEngine 中
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)
    seq = Sequence(prompt, sampling_params)
    self.scheduler.add(seq)
```

### 7.3 is_finished 方法

```python
def is_finished(self):
    return not self.waiting and not self.running
```

- 只要 waiting 和 running 都空了，说明所有请求都结束了
- `LLMEngine.is_finished()` 最终就是委托这个方法来判断推理循环是否结束

---

## 八、调度器的完整生命周期示例

### 8.1 场景设置

假设系统参数：
- `max_num_seqs = 4`
- `max_num_batched_tokens = 1024`
- `block_size = 256`
- 共有 10 个物理 block

### 8.2 执行时间线

```
T=0: 请求 A 到达 (prompt 300 token)
     waiting: [A(300)]
     running: []

T=1: schedule() — prefill 阶段
     A 需要 2 个 block，分配成功
     waiting: []
     running: [A]
     → 返回 ([A], is_prefill=True)
     → ModelRunner 执行 A 的 prefill

T=1: postprocess()
     A 生成 token_301
     A 未结束
     running: [A(301)]

T=2: 请求 B 到达 (prompt 500 token)
     waiting: [B(500)]
     running: [A(301)]

T=2: schedule() — prefill 优先
     B 需要 2 个 block，分配成功
     waiting: []
     running: [A(301), B(500)]
     → 返回 ([B], is_prefill=True)

T=2: postprocess()
     B 生成 token_501
     running: [A(301), B(501)]

T=3: schedule() — 无 waiting，进入 decode
     A: can_append? Yes (block 未满)
     B: can_append? Yes (block 未满)
     → 返回 ([A, B], is_prefill=False)

T=3: postprocess()
     A 生成 token_302, B 生成 token_502
     running: [A(302), B(502)]

... 正常 decode ...

T=10: A 遇到 EOS
      postprocess: A.status = FINISHED, 释放 2 个 block
      running: [B(510)]

T=11: 请求 C 到达
      waiting: [C]
      schedule(): prefill 优先，调度 C
```

---

## 九、调度策略的深度对比

### 9.1 FCFS（先来先服务）

nano-vllm 的 waiting 队列本质上是 FCFS——先到的请求先被调度。

**优点**：简单、公平
**缺点**：不能区分请求优先级，长 prompt 可能阻塞短 prompt

### 9.2 Prefill 优先 vs Decode 优先

| 策略 | TTFT | TPOT | 吞吐量 | 实现复杂度 |
|------|------|------|--------|-----------|
| Prefill 优先 | 低 | 可能较高 | 中 | 低 |
| Decode 优先 | 高 | 低 | 中 | 低 |
| 混合调度 | 中 | 中 | 高 | 高 |

- **TTFT**（Time To First Token）：用户等待第一个输出 token 的时间
- **TPOT**（Time Per Output Token）：每个输出 token 的生成时间

nano-vllm 选择 prefill 优先是因为 TTFT 对用户体验影响更大——用户更在意"什么时候开始有回复"而非"回复速度有多快"。

### 9.3 轻量 chunked prefill

很多人读完当前源码后，会以为 nano-vllm **完全没有** chunked prefill。其实不完全对。

当前实现里已经有一个**轻量版**：

- 只允许 **waiting 队头**请求做 chunk
- 只允许在 **prefill 分支内部**分块
- **不会**和 decode 混到同一个 step 里一起跑

也就是这句：

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
```

它表达的真实语义是：

- 如果当前队头请求很长，但它是这轮第一个请求，那么允许先切一段出来算
- 如果它不是第一个请求，就不允许切 chunk，直接停下

所以更准确地说，nano-vllm 具备的是“**prefill-only、head-only 的轻量 chunked prefill**”，而不是生产级系统那种“和 decode 交错混跑”的完整方案。

真正更先进的系统会把长 prompt 的 prefill 分成多个 chunk，与 decode 序列交错执行。这样既不会因为长 prompt 阻塞 decode 序列，又能保持较低的 TTFT。

```
传统 prefill 优先:
Step 1: [A(prefill 2000 tokens)]  ← decode 序列被阻塞
Step 2: [B(decode), C(decode)]

chunked prefill:
Step 1: [A(prefill chunk1 512 tokens), B(decode), C(decode)]
Step 2: [A(prefill chunk2 512 tokens), B(decode), C(decode)]
Step 3: [A(prefill chunk3 512 tokens), B(decode), C(decode)]
Step 4: [A(prefill chunk4 464 tokens), B(decode), C(decode)]
```

因此，如果面试里被问到“nano-vllm 支不支持 chunked prefill”，更稳妥的回答应该是：

> 支持一个非常简化的版本，只允许 waiting 队头请求按 step 分块续跑，但不支持 prefill/decode 混合调度。

### 9.4 Priority scheduling

在生产环境中，不同用户/请求可能有不同优先级（如付费用户 > 免费用户）。这需要在 waiting 队列中实现优先队列（如 heap），而非简单的 FIFO。

---

## 十、BlockManager 交互

### 10.1 调度器与 BlockManager 的协作

```python
# 调度器不直接管理物理 block，而是委托给 BlockManager

# Prefill 时：
self.block_manager.can_allocate(seq)            # 询问：空间够不够，以及前缀能复用多少整块
self.block_manager.allocate(seq, num_cached_blocks)  # 执行：分配 block 并填充 seq.block_table

# Decode 时：
self.block_manager.can_append(seq)     # 询问：能追加一个 token 吗？
self.block_manager.may_append(seq)     # 执行：如果需要，分配新 block

# 抢占/完成时：
self.block_manager.deallocate(seq)     # 执行：释放该序列的所有 block
```

### 10.2 资源管理的两阶段检查

nano-vllm 采用**先检查后执行**的模式：

1. `can_xxx` 方法：只读查询，不修改状态
2. `allocate/deallocate/may_append` 方法：实际执行资源操作

这种设计允许调度器在决策阶段安全地"试探"资源状况，而不会因为检查操作产生副作用。

---

## 十一、调度器的核心设计原则

### 11.1 Prefill 和 Decode 不混合

在同一步中，要么全做 prefill，要么全做 decode。这简化了 ModelRunner 的实现——不需要在同一个 batch 中混合两种不同的计算模式。

```python
if scheduled_seqs:
    return scheduled_seqs, True   # 有 prefill，本步只做 prefill
# ...
return scheduled_seqs, False      # 否则做 decode
```

但这也意味着 decode 中的序列在有新 prefill 请求到来时会**暂停一步**。

### 11.2 保守调度

调度器倾向于保守——宁可少调度几个序列，也不要因为资源不足导致系统崩溃：

- token 数检查：`remaining == 0` 或 `remaining < num_tokens and scheduled_seqs` → break
- block 检查：`can_allocate(seq) == -1` → break

一旦遇到无法调度的序列，立即停止调度后续序列，即使后续序列可能更小。这是 FCFS 的特性——不会跳过队头的大请求去调度后面的小请求。

### 11.3 单步原子性

每次 `schedule()` 调用产生一个完整的调度结果。调度过程中的所有操作（分配 block、修改状态、移动队列）要么全部成功，要么需要回滚。在 nano-vllm 中，由于是单线程执行，不会出现并发问题。

---

## 十二、调度器的潜在改进

### 12.1 支持 Swap（交换到 CPU）

当前的 preempt 策略是**全部释放**（recompute），被抢占的序列需要重新做 prefill。改进方案是将 KV Cache 从 GPU 交换到 CPU 内存，后续恢复时只需从 CPU 传回 GPU，避免重新计算。

### 12.2 支持优先级调度

将 waiting 队列从 deque 改为优先队列，支持基于优先级或等待时间的调度。

### 12.3 支持更完整的 Chunked Prefill

在当前“head-only、prefill-only”的轻量版本基础上，进一步支持长 prompt 与 decode 交错执行，平衡 TTFT 和 TPOT。

### 12.4 支持 Speculative Decoding

投机解码需要调度器支持"草稿模型 + 验证"的两阶段执行模式。

---

## 十三、源码对照总结

| Scheduler 方法/属性 | 调用者 | 目的 |
|---------------------|--------|------|
| `__init__` | LLMEngine | 初始化调度器和 BlockManager |
| `add(seq)` | LLMEngine.add_request | 新序列入队 |
| `schedule()` | LLMEngine.step | 选出本步序列，返回 (seqs, is_prefill) |
| `postprocess(seqs, token_ids, is_prefill)` | LLMEngine.step | 更新缓存进度、必要时追加 token、判断终止 |
| `preempt(seq)` | schedule() 内部 | 抢占序列释放资源 |
| `is_finished()` | LLMEngine | 判断是否所有任务完成 |

---

## 十四、面试考点

### 考点 1：请描述 nano-vllm 调度器的 schedule() 方法的完整流程

**标准回答**：先把几个概念说清。**请求**是用户的一次生成任务，进入引擎后会被包装成一个 `Sequence`。`Sequence` 是“请求的运行时对象”，里面同时记录 token、KV block、缓存进度和状态。**Prefill** 不是“一个新请求”，而是这条 `Sequence` 的一个计算阶段，作用是把它**当前已有上下文里还没缓存的 token**算进 KV Cache。这里的“已有上下文”在第一次进入时主要是 prompt；如果请求被抢占后重新回来，也可能包括 prompt 加上已生成的部分 token。**Decode** 也不是“另一条请求”，而是同一条 `Sequence` 在 prefill 完成后的下一个阶段：此时历史上下文已经都有 KV Cache，之后每一轮通常只处理 1 个 `last_token`，并采样出下一个 token。

**waiting 队列**里放的是“**还没有准备好进入 decode 轮转**”的序列”，包括三类：刚到达的新请求、只做了一部分 prefill 还没铺完上下文的请求、以及被抢占后回退的请求。**running 队列**里放的是“**当前已有上下文已经完整缓存，可以参与后续 decode 轮转**”的活跃序列；已经 `FINISHED` 的序列会被移出，不再留在 running。

在这个前提下再看 `schedule()`：它总是先尝试 prefill，优先检查 `waiting` 队头。调度器结合 `max_num_batched_tokens`、`can_allocate(seq)` 和 prefix cache 命中情况，决定这轮给该 `Sequence` 安排多少 `num_scheduled_tokens`。如果这轮结束后满足 `seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens`，表示这条序列**当前已有上下文已经全部被 KV Cache 覆盖**。这里要特别注意：这不表示“请求结束了”，只表示“**prefill 阶段结束了**”。也正因为如此，调度器才会把它从 `waiting` 挪到 `running`，表示它从“还在补上下文”切换成了“后续可以参加 decode 轮转”的活跃序列。相反，如果这个条件不成立，说明这轮只是**部分 prefill**，上下文还没铺完，它虽然已经做了一些计算、甚至已经拿到了 `block_table`，但仍然留在 `waiting`，下一轮继续 prefill。

只要这轮成功调度了任何 prefill 序列，`schedule()` 就会直接返回 `is_prefill=True`，本轮不会再进入 decode。只有当这轮一个 prefill 都没调度成功时，才进入 decode 分支：从 `running` 队头按 FIFO 轮转取序列，检查 `can_append(seq)`；如果空间不够，就按 LIFO 抢占 `running` 队尾序列；如果成功，就给当前序列安排 1 个 decode token，也就是设置 `num_scheduled_tokens = 1`，然后返回 `is_prefill=False`。

### 考点 2：为什么采用 prefill 优先策略？这种策略有什么优缺点？

**标准回答**：Prefill 优先降低了 TTFT（首 token 延迟），让新请求更快看到首个输出。Prefill 也是 compute-bound，GPU 利用通常更高。代价是 running 中正在 decode 的序列，在有新 prefill 请求到来且可调度时会暂停一步，TPOT 可能变差。当前 nano-vllm 已经支持一个轻量版 chunked prefill，但仍然不做 prefill/decode 混跑；更完整的优化方向是生产级的混合调度。

### 考点 3：描述抢占（preempt）机制的实现，为什么用 LIFO 策略？

**标准回答**：当 decode 阶段需要新 block 但没有空闲 block 时，调度器从 running 队列尾部取出序列进行抢占——释放其所有 KV Cache block，将其状态改回 WAITING，放到 waiting 队列头部。使用 LIFO 策略是因为最后加入的序列生成的 token 最少，被抢占浪费的计算量最小。被抢占序列放到 waiting 头部是为了保证公平性，避免饥饿。

### 考点 4：nano-vllm 的抢占策略是 recompute，与 swap 策略有什么区别？

**标准回答**：Recompute 策略直接丢弃被抢占序列的 KV Cache，重新调度时需要重新做 prefill，计算浪费大但实现简单、不需要额外内存。Swap 策略将 KV Cache 从 GPU 交换到 CPU 内存，恢复时只需传回 GPU，避免重复计算，但需要额外的 CPU 内存和 PCIe 带宽，实现也更复杂。vLLM 同时支持两种策略。

### 考点 5：postprocess 的终止条件有哪些？如何处理 ignore_eos？

**标准回答**：严格来说，`postprocess()` 先做三件事：给新完成的 block 做哈希、把 `num_cached_tokens` 向前推进、清零 `num_scheduled_tokens`。如果当前是部分 prefill，且 prompt 还没算完，会直接 `continue`，这轮不会 append 新 token。只有完整 prefill 结束或 decode 时，才会 `append_token(token_id)`。终止条件有两个：(1) 生成了 EOS 且 `ignore_eos=False`，(2) completion token 数达到 `max_tokens`。`max_tokens` 是硬上限，`ignore_eos` 只能屏蔽 EOS 终止，不能突破长度上限。

### 考点 6：如果让你改进 nano-vllm 的调度器，你会从哪些方面入手？

**参考思路**：
1. 实现 chunked prefill，平衡 TTFT 和 TPOT
2. 添加 swap 策略减少抢占浪费
3. 支持优先级调度（priority queue）
4. 支持 prefix-aware 调度，共享前缀的请求一起调度以最大化缓存命中
5. 支持投机解码的两阶段调度
6. 添加公平性保障（基于等待时间的优先级提升）

### 考点 7：FlashAttention 和 chunked prefill 有什么区别与联系？

**标准回答**：两者都带“分块”色彩，但分的根本不是同一层东西。`FlashAttention` 的分块是**kernel 内部的 tiling**：经典 attention 会材料化完整的 \(S \times S\) 注意力矩阵，HBM 读写很重；FlashAttention 把 Q/K/V 在 kernel 内部分 tile，在 SRAM 上完成 online softmax 规约，避免完整材料化注意力矩阵，所以它解决的是“**单次 attention 前向内部怎么算得更省 IO、更快**”。在 nano-vllm 里，prefill 走 `flash_attn_varlen_func`，输入是多条变长序列拼接后的张量和 `cu_seqlens_*`；decode 走 `flash_attn_with_kvcache`，输入是当前短 Q、历史 `k_cache/v_cache` 和 `cache_seqlens`。  

`chunked prefill` 的分块则是**调度层的 chunking**：它不是改 attention kernel 的内部算法，而是把一个很长的 prefill 任务，在调度层拆成多个 step 去跑。也就是原本要一次性处理完整 prompt，现在变成“这轮先算前 1024 个 token，下轮再算后 1024 个 token”之类。它解决的是“**长 prompt 什么时候算、一次算多少、会不会把 decode 卡住**”。

所以二者最本质的区别是：

1. `FlashAttention` 分的是**一次前向内部的计算 tile**，这个 tile 对调度器通常是**不可见**的。
2. `chunked prefill` 分的是**一条长请求在时间维度上的执行轮次**，这个 chunk 对调度器是**可见**的，会直接体现在 `num_scheduled_tokens`、`waiting/running` 和 step 次数上。
3. `FlashAttention` 的 tiling 发生在**一次 kernel 调用内部**；`chunked prefill` 的 chunking 发生在**多次 `schedule -> run -> postprocess` 之间**。

二者的联系是：**chunked prefill 拆出来的每个 chunk，内部仍然可以继续使用 FlashAttention**。也就是说，chunked prefill 决定“这次拿 1024 个 token 来算”，而 FlashAttention 决定“这 1024 个 token 在 attention 内部如何分 tile、高效完成”。一句话记忆就是：**FlashAttention 优化单次 chunk 的计算成本，chunked prefill 优化多个 chunk 之间的调度时机；它们是上下层配合关系，不是互斥关系。**

**延伸补充**：
1. **没有 FlashAttention，只有 chunked prefill**：长 prompt 虽然不会整段阻塞 decode，但每个 chunk 自身的 attention 仍可能很慢，因为 kernel 里还是可能有很重的 HBM IO。
2. **没有 chunked prefill，只有 FlashAttention**：单次 prefill 会更快，但超长 prompt 仍可能独占一个 step，造成 head-of-line blocking。
3. **FlashAttention 不只服务 prefill**：decode 也有带 KV cache 的 FlashAttention 路径；而 chunked prefill 只作用在 prefill 阶段。
4. **可以顺手区分三种“block / chunk”**：
   - `FlashAttention tile`：kernel 内部为了降 IO 的计算分块。
   - `chunked prefill chunk`：调度层把长 prompt 拆成多轮执行的逻辑分段。
   - `PagedAttention block / block_table`：KV Cache 在显存中的物理分页单位，用于存储和寻址。
5. **一个很实用的口述例子**：假设 prompt 长度 4096，调度器决定按 1024 做 chunked prefill，那么这条请求会经历 4 轮 prefill step；而在每一轮处理这 1024 个 token 时，FlashAttention 内部还会继续做更细粒度的 tiling。前者决定“**分几轮跑**”，后者决定“**每轮内部怎么算**”。

### 考点 8：为什么 prefill 和 decode 不在同一步混合执行？

**标准回答**：Prefill 和 decode 的输入组织方式不同。Prefill 往往是一段连续 token，decode 每条序列一次只处理最后一个 token，二者的张量准备和上下文组织都不一样。当前 nano-vllm 在单个 `schedule()` 结果里仍坚持二选一，这能显著简化 `ModelRunner` 和上下文构造逻辑。要注意补一句：当前实现虽然不做 prefill/decode 混合 batch，但已经支持一个轻量版的“按 step 切分 prefill”。

---

## 十五、小结

| 知识点 | 核心理解 |
|--------|---------|
| 调度器角色 | 引擎的"大脑"，决定每一步谁参与计算 |
| 双队列模型 | waiting（等待） + running（运行），用 deque 实现 |
| Prefill 优先 | 新请求优先处理，降低 TTFT |
| Decode 轮转 | FIFO 依次处理 running 中的序列 |
| 抢占机制 | LIFO 策略 + recompute 策略，被抢占者回到 waiting 头部 |
| postprocess | 先更新缓存进度，再在合适时机追加 token、终止检查、资源回收 |
| 资源管理 | 调度器不直接管理 block，委托给 BlockManager |

**下一课预告**：我们将从调度器的视角上升到系统层面，深入理解**连续批处理（Continuous Batching）**——它是调度器和 ModelRunner 配合实现的核心优化策略。

---

> **学习建议**：在纸上模拟一个有 3-4 个请求的调度场景，画出每一步 waiting 和 running 队列的变化，以及 block 的分配和释放过程。这对理解调度器至关重要。
