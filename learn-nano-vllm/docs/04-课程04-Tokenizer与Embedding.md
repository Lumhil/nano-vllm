# 课程 04：Tokenizer 与 Embedding

> 先把字符串变成 token id（Tokenizer 工作流），再看 **词表并行**：每张卡只存 `V/tp_size` 行嵌入，用 mask + all_reduce 拼出完整向量；最后理解 **ParallelLMHead** 在 Prefill 只取每序列最后一个位置——与自回归「预测下一 token」一致。

## 本课目标

1. 描述 **Tokenizer 工作流**：normalize → 词表映射 → special tokens →（可选）chat 模板。
2. 理解 **VocabParallelEmbedding** 的 **切分方式**、**mask 含义**、**all_reduce 必要性**。
3. 解释 **ParallelLMHead.forward** 中 `cu_seqlens_q` 与 **只取 last token** 的原因。
4. 面试中能对比 **Embedding 前向** 与 **LMHead 前向** 在 TP 下的差异（reduce vs gather）。

## 核心概念

### Tokenizer 工作流（以 HuggingFace 为例）

nano-vllm 的 **`example.py`** 在引擎外使用 **`AutoTokenizer`**，典型步骤：

1. **加载**：`from_pretrained(model_dir)` 读 `tokenizer.json` / `vocab` 等。
2. **对话格式**：`apply_chat_template(messages, ...)` 生成 **带角色标记** 的字符串，便于指令模型理解 **user/assistant** 边界。
3. **编码**：引擎侧或脚本侧 `encode` 得到 **token id 序列**（整数张量），作为 **`embedding` 层输入**。

**面试常问点**：Tokenizer **不属于** `nanovllm` 核心包，但 **与词表大小 V、embed 权重形状** 强相关；**词表并行** 正是按 **V 的维度** 切分。

### 词表并行（Vocab Parallelism）

先把这几件事对齐，不然后面很容易混：

- **序列（sequence）**：一次请求对应的一条 token 链。它包含用户输入的 prompt，也包含模型后续生成的 token。
- **batch**：一次前向里同时处理的多条序列。比如 3 个请求一起送进模型，就是一个 batch 里有 3 条序列。
- **hidden / hidden state**：模型在某一层、某一位置算出来的内部表示向量。它不是最终输出文本，而是“模型目前如何理解这个位置”的中间表示，长度通常就是隐藏维 `D`。

输入 token id 进入模型后，第一步通常是查 **嵌入矩阵** `E`。`E` 的形状可记为 `(V, D)`：

- `V` 是 **词表大小**（vocab size）
- `D` 是 **隐藏维**（hidden size）

`E` 的第 `i` 行就对应 **词表中第 `i` 个 token** 的向量，所以这里“权重”和“词表”天然绑定：

- 词表决定一共有多少个 token，也就是矩阵要有多少行
- 每一行权重都代表一个 token 的 embedding
- token id 本质上就是去 `E` 里按行索引

当 `E` 太大时，就可以按 **词表维度** 切分，也就是做 **词表并行**：

- 第 `r` 张卡只保存全局词表中的一段行区间 `[v_start, v_end)`，也就是大约 `V/tp_size` 行
- 前向时，如果某个 token id 落在本卡负责的区间，就用本地行查表
- 如果某个 token id 不在本卡区间，本卡对这个位置的贡献就是 0
- 所有卡算完后，对结果做一次 `all_reduce(sum)`，就能把完整 embedding 合并出来

这里之所以可以“求和”，是因为 **每个 token 的正确 embedding 只会在一张卡上是非零的**，其余卡对应位置都是 0。  
所以求和后的结果，等价于“把正确那一行取出来”。

**直觉**：每张卡只保存并计算自己那部分词表；对某个 token 来说，真正有用的 embedding 只在一张卡上，其余卡补 0，最后一加就是完整向量。

### LMHead 与 Embedding 权重共享

很多自回归模型会做 **weight tying**，也就是：

- 输入侧的 embedding 用矩阵 `E`
- 输出侧的 LMHead 也复用这张矩阵，只是换一种用法

输入侧的逻辑是：

- token id `i` -> 取 `E[i]` 这一行，得到该 token 的 embedding

输出侧的逻辑是：

- 某个位置的 hidden 向量记为 `h`
- 用 `logits = h @ E^T` 计算对整个词表的打分

这说明 `E` 不只是“输入查表用的权重”，它的每一行同时也代表“这个 token 在输出层对应的打分方向”。  
因此：

- **词表中的每个 token** 对应 `E` 的一行
- **输入时** 按 token id 取那一行
- **输出时** 用 hidden 和每一行做点积，得到“下一个 token 是它的分数”

这就是“权重和词表怎么联系上”的核心。

`ParallelLMHead` 继承 `VocabParallelEmbedding`，就是因为它们都围绕同一张按词表切分的矩阵工作；但两者的前向不同：

- **Embedding**：做的是查表
- **LMHead**：做的是线性投影，把 hidden 变成 logits

再看 prefill / decode：

- **Prefill**：一个 batch 里每条序列会一次性送入多枚 token，所以模型会为这些位置都算出 hidden。  
  但在推理里，我们只需要 **每条序列最后一个位置的 hidden** 去预测“下一个 token”。
- **Decode**：通常每条序列这一步只新增 1 个 token，因此天然只有“当前这一个位置”的 hidden，直接拿来预测下一 token 即可。

### `cu_seqlens_q` 是什么（与本课相关）

Prefill 时，一个 batch 里的多条序列长度往往不同，所以实现里常把它们 **展平** 成一个连续 token 数组来计算。  
`cu_seqlens_q` 就是这个展平数组的 **前缀和边界表**，长度为 `batch + 1`。

例如，若 batch 内 3 条序列的 query 长度分别是 `[3, 2, 4]`，那么：

```text
cu_seqlens_q = [0, 3, 5, 9]
```

它表示：

- 第 0 条序列在展平数组中的范围是 `[0, 3)`
- 第 1 条序列在展平数组中的范围是 `[3, 5)`
- 第 2 条序列在展平数组中的范围是 `[5, 9)`

因此：

```text
cu_seqlens_q[1:] - 1 = [2, 4, 8]
```

这正好就是 **每条序列最后一个位置** 在展平数组中的下标。

在 `ParallelLMHead.forward` 里，`x` 是展平后的 hidden，形状可以理解成 `(total_query_tokens, D)`。  
Prefill 阶段先用 `cu_seqlens_q[1:] - 1` 取出每条序列最后一个 hidden，再据此计算 logits，含义就是：

- 每条序列前面的 hidden 只是“铺上下文”
- 真正要拿来预测下一 token 的，是最后一个位置的 hidden

本课里先把 `q` 记成“本轮需要参与计算的 query token”。在没有 prefix cache 时，它通常就是本轮输入的这些 token；有 prefix cache 时，`q` 可能只覆盖新增部分。

## 源码解析（带完整源码和逐行注释）

下列代码与仓库 `nanovllm/layers/embed_head.py` 一致（含 `weight_loader` 便于理解权重加载）。

```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
```

### VocabParallelEmbedding 逐段注释

| 代码片段 | 解释 |
|----------|------|
| `dist.get_rank()` / `get_world_size()` | 当前 **TP 组** 内 rank 与 **并行度 tp_size** |
| `num_embeddings % self.tp_size == 0` | 词表行数必须 **整除**，否则无法均分 |
| `num_embeddings_per_partition` | 每卡 **本地词表行数** `V/tp_size` |
| `vocab_start_idx` / `vocab_end_idx` | 本卡负责的 **全局 token id 区间** |
| `self.weight` 形状 `(V/tp, D)` | 只存 **本分片** 的嵌入表 |
| `weight_loader` | 从 **完整 HF 权重** 按行切 **`narrow`** 再 `copy_`，与 TP rank 对齐 |
| `mask = (x >= ...) & (x < ...)` | 标记 **哪些位置属于本卡词表** |
| `x = mask * (x - self.vocab_start_idx)` | 将 **全局 id** 转为 **本地行号**；不属于本卡的 id 被置 0（与 mask 配合） |
| `F.embedding(x, self.weight)` | 标准查表；越界 id 行为依赖 mask 与后续乘法 |
| `mask.unsqueeze(1) * y` | 非本卡词表位置 **嵌入置零**，避免脏值进入规约 |
| `dist.all_reduce(y)` | **求和** 合并各卡贡献，得到 **完整 D 维向量** |

### ParallelLMHead 逐段注释

| 代码片段 | 解释 |
|----------|------|
| `assert not bias` | 输出层 **无 bias**，与 Qwen 类实现一致，简化并行 |
| `get_context()` | 取 **全局推理上下文**（prefill/decode、cu_seqlens 等） |
| `if context.is_prefill` | **Prefill**：序列并行展开，`hidden` 形状对应 **所有位置** |
| `last_indices = context.cu_seqlens_q[1:] - 1` | 每条序列 **最后一个 token** 在展平 `hidden` 里的索引 |
| `x = x[last_indices].contiguous()` | 只保留 **last hidden**，形状 `(batch, D)`，准备算 **下一 token logits** |
| `F.linear(x, self.weight)` | 与 embedding 同权重的 **线性层**：`logits = x @ W^T`（形状细节以布局为准） |
| `tp_size > 1` 时 `gather` + `cat` | **词表维切分** 下，每卡只持有 **部分 vocab 列**；需在 **logits 最后一维** 拼接成全词表 logits（**gather 到 rank0** 是常见模式） |

**注意**：多卡时 **rank 非 0** 可能返回 `None`，由引擎保证只在需要处消费 logits；以你阅读的 `sampler`/`engine` 为准。

## 图解（用文字/ASCII 描述）

**词表并行 Embedding（tp=2 示意）**：

```
全局 token id:  0 ... V/2-1  |  V/2 ... V-1
                 ----卡0----    ----卡1----

token 落在卡0区间 -> 卡0算向量，卡1置零 -> all_reduce 相加 -> 完整向量
```

**Prefill 时 LMHead 取 last**：

```
batch 内 3 条序列，展平后 hidden 下标:
  seq0: [0,1,2]
  seq1: [3,4]
  seq2: [5,6,7,8]

cu_seqlens_q 类似 [0,3,5,9]
last_indices = [2,4,8]  -> 只取这三处 hidden 做 logits
```

## 面试考点

- **词表并行 vs 行并行/列并行**：这里并行的是 **嵌入矩阵的行（vocab 维）**。
- **为什么用 mask + all_reduce**：每张卡只负责部分 id，**其余必须为 0** 再规约。
- **LMHead 在 prefill 只取 last**：对齐 **因果 LM 的预测位置**（预测 **下一个** token）。
- **TP>1 时 logits 需 gather/cat**：每张卡 **部分 vocab logits**，拼接成全词表再采样。

## 常见面试题

1. **若没有 `mask * y` 直接 all_reduce 会怎样？**  
   答：非本分片 id 可能产生 **错误非零嵌入**，规约后 **污染结果**。

2. **Decode 阶段 LMHead 还需要 `cu_seqlens_q` 吗？**  
   答：通常 **每序列一步**，`is_prefill` 为 False 时 **不走路径**；以 `context` 为准。

3. **weight tying 时如何加载权重？**  
   答：`weight_loader` 对 **同一份 checkpoint 行切分** 到各卡，**embedding 与 lm_head 共享 Parameter**（若模型实现如此）。

4. **Tokenizer 词表大小与 `num_embeddings` 不一致会怎样？**  
   答：配置/权重不匹配会 **load 失败** 或 **越界**；需与 HF `config.vocab_size` 对齐。

## 小结

- **Tokenizer** 在引擎外把文本变为 **id**；**词表大小** 驱动嵌入形状。
- **VocabParallelEmbedding** 用 **区间 mask + 本地行号 + all_reduce** 实现 **无重复全表存储** 的嵌入。
- **ParallelLMHead** 在 **prefill** 用 **`cu_seqlens_q`** 定位 **每序列最后位置**，与 **自回归目标** 对齐；多卡时对 **logits 维做 gather/拼接**。

## 下一课预告

下一课 **《05-课程05-Attention机制与FlashAttention》** 将拆解 **`store_kvcache` Triton 内核**、`flash_attn_varlen_func` 与 `flash_attn_with_kvcache` 两分支，以及 **prefix cache（`block_tables`）** 下如何从 **KV cache** 读 K、V。
