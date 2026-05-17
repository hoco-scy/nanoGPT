# Day-3

## 理解Self-Attention

### 阅读源码`model.py`

CausalSelfAttention 类


## Q&A

### Q: train.py 320行 `logits, loss = model(X, Y)` 中的 `model` 是函数吗？为什么找不到定义？

**A: `model` 不是函数，是 `GPT` 类的实例（对象）。调用 `model(X, Y)` 触发的是 PyTorch `nn.Module.__call__`，内部调用 `forward` 方法。**

**1. model 在哪创建的？**

`train.py:172`：

```python
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
```

`model` 是 `GPT(gptconf)` 创建出来的对象，`GPT` 定义在 `model.py` 中，继承自 `nn.Module`。

**2. 为什么可以像函数一样调用？**

`model(X, Y)` 看起来像函数调用，实际上是 Python 的**可调用对象**语法。当你 `model(X, Y)` 时，Python 会调用 `nn.Module.__call__` 方法，这个方法内部会调用你定义的 `forward` 方法。这是 PyTorch 的核心设计模式。

**3. 真正执行的代码在哪？**

`model.py:170-193` 的 `GPT.forward` 方法：

```python
def forward(self, idx, targets=None):
    # ... token embedding + position embedding + transformer blocks ...

    if targets is not None:
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    else:
        logits = self.lm_head(x[:, [-1], :])
        loss = None

    return logits, loss
```

注意参数：`X` 对应 `idx`（输入 token），`Y` 对应 `targets`（目标 token）。当 `targets` 不为 `None` 时，除了输出 `logits`，还会计算交叉熵 `loss`。

**4. 关键总结**

| 概念 | 说明 |
|------|------|
| `model` | `GPT` 类的实例，继承自 `nn.Module` |
| `model(X, Y)` | 调用 `nn.Module.__call__` → 内部调用 `forward` |
| `forward` | `model.py:170`，定义了前向传播和 loss 计算 |

这是 PyTorch 的标准模式：**把模型当函数调用**。`nn.Module.__call__` 除了调用 `forward`，还会自动处理 hooks、梯度追踪等逻辑，所以永远不要直接调用 `model.forward()`，而是用 `model()`。

### Q: 为什么不能直接调用 model.forward()？

**A: `nn.Module.__call__` 在调用 `forward` 前后会执行额外逻辑，直接调 `forward()` 会跳过这些，导致混合精度失效、torch.compile 失效等问题。**

`__call__` 额外做的事情：

**1. 激活 `autocast` 混合精度**

`train.py:319-320` 用 `with ctx:` 包裹模型调用，`autocast` 通过 forward pre-hook 生效。直接调 `forward()` 时，autocast 不会自动传递到每一层，模型可能悄悄跑在 fp32 上，慢一倍但你不知道。

**2. `torch.compile` 完全失效**

`train.py:223` 做了 `model = torch.compile(model)`。编译后的对象只有 `model(X, Y)` 走编译图，`model.forward(X, Y)` 绕过编译层，等于白装。

**3. 梯度追踪行为不同**

`__call__` 模式下 PyTorch 正确建立计算图；直接调 `forward()` 在某些 edge case 下 autograd 行为不一致，属于 undefined behavior。

**4. Hooks 被跳过（通用场景）**

虽然 nanoGPT 没用 hooks，但很多训练框架注册 forward hooks 做梯度监控、LoRA 注入等。直接调 `forward()` 会静默跳过。

**总结：** 简单场景下直接调 `forward()` 碰巧能工作，但在用了 `autocast` + `torch.compile` + `DDP` 的场景会出真实 bug。养成用 `model()` 的习惯。

### Q: GPTConfig 中各参数的含义？为什么断言 n_embd % n_head == 0？

**A: n_embd 是每个token的向量维度，n_head 是注意力头数。多头注意力要把 n_embd 维向量均匀切分给每个头，所以必须整除。**

**1. GPTConfig 参数 (`model.py:108-116`)**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `block_size` | 1024 | 上下文窗口长度，一次能看多少个token |
| `vocab_size` | 50304 | 词表大小 |
| `n_layer` | 12 | Transformer Block 层数 |
| `n_head` | 12 | 注意力头数 |
| `n_embd` | 768 | 嵌入维度，每个token用多少维向量表示 |
| `dropout` | 0.0 | dropout 概率 |
| `bias` | True | Linear/LayerNorm 是否用偏置 |

**2. 层级关系**

```
GPT
 └─ n_layer 个 Block
     └─ 每个 Block = CausalSelfAttention + MLP
         └─ CausalSelfAttention 把 n_embd 维向量拆成 n_head 份并行计算
```

**3. 为什么 n_embd 必须是 n_head 的整数倍？**

`model.py:57-59` 做了 `view` 操作把向量切分：

```python
head_size = C // self.n_head  # 768 / 12 = 64

k = k.view(B, T, self.n_head, C // self.n_head)  # (B, T, 12, 64)
```

如果 n_embd=768, n_head=13，`768 / 13 = 59.07...`，`view` 无法均匀切分，直接报错。

**4. 多头注意力的直觉**

```
一个768维向量 → 拆成12个64维子向量 → 每个头独立做注意力 → 拼回768维
```

每个头可以学到不同类型的注意力模式（语法关系、指代关系、位置邻近等），这就是"多头"的意义：并行地从不同角度理解上下文。
