# Day-2

## 阅读train.py源码

对 `train.py` 的完整拆解，标注了与源码的行号对应关系。

---

### 1. 文件头部 (L1-17)

纯注释，说明三种运行方式：

- 单 GPU 调试：`python train.py`
- 单节点多 GPU (DDP)：`torchrun --standalone --nproc_per_node=4 train.py`
- 多节点多 GPU (DDP)：需要指定 `master_addr` / `master_port`

---

### 2. 导入 (L19-32)

```python
import os, sys, time, math, pickle, json
from contextlib import nullcontext    # CPU时不使用autocast的占位符
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import GPTConfig, GPT      # 核心：模型定义
```

---

### 3. 默认配置 (L34-77)

这是"Poor Man's Configurator"的上半部分。所有超参数以**全局变量**形式定义：

| 类别 | 变量 | 默认值 | 含义 |
|------|------|--------|------|
| **I/O** | `out_dir` | `'out'` | 输出目录 |
| | `eval_interval` | `2000` | 每N步做一次验证 |
| | `eval_iters` | `200` | 验证时取多少batch估算loss |
| | `eval_only` | `False` | 若True，验证一次就退出 |
| | `init_from` | `'scratch'` | `'scratch'`/`'resume'`/`'gpt2'` |
| **数据** | `dataset` | `'openwebtext'` | 数据集名 |
| | `gradient_accumulation_steps` | `40` | 模拟大batch，实际batch=40×12×1024 |
| | `batch_size` | `12` | micro-batch大小 |
| | `block_size` | `1024` | 序列长度(上下文窗口) |
| **模型** | `n_layer/n_head/n_embd` | `12/12/768` | GPT-2 124M的配置 |
| | `dropout` | `0.0` | 预训练不用dropout |
| **优化器** | `learning_rate` | `6e-4` | 最大学习率 |
| | `max_iters` | `600000` | 总训练步数 |
| | `weight_decay` | `0.1` | 权重衰减 |
| | `beta1/beta2` | `0.9/0.95` | AdamW动量参数 |
| | `grad_clip` | `1.0` | 梯度裁剪阈值 |
| **LR调度** | `warmup_iters` | `2000` | 线性预热步数 |
| | `min_lr` | `6e-5` | 最小学习率 ≈ lr/10 |
| **系统** | `device` | `'cuda'` | 计算设备 |
| | `dtype` | `'bfloat16'` | 混合精度类型 |
| | `compile` | `True` | 是否用torch.compile加速 |

---

### 4. 配置覆盖机制 (L78-80)

```python
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
```

三步走：

1. 收集所有可配置的全局变量名（int/float/bool/str类型）
2. 执行 `configurator.py`，它会处理命令行参数（`--key=value`）和配置文件（如 `config/train_gpt2.py`），覆盖对应全局变量
3. 把最终值存入 `config` 字典用于日志

---

### 5. DDP 初始化 (L83-104)

```python
ddp = int(os.environ.get('RANK', -1)) != -1  # 有RANK环境变量就是DDP模式
```

**DDP模式**下：

- `init_process_group(backend)` — 初始化分布式进程组
- 从环境变量读取 `RANK`（全局rank）、`LOCAL_RANK`（本机rank）、`WORLD_SIZE`（总进程数）
- 设备绑定到 `cuda:{local_rank}`
- 只有 `rank==0` 是 master_process，负责日志和存档
- `gradient_accumulation_steps` 按world_size均分（因为每个进程都在算梯度）

**单GPU模式**下：master_process=True, world_size=1

关键公式 (L103)：

```
tokens_per_iter = gradient_accumulation_steps × world_size × batch_size × block_size
```

默认配置：40 × 1 × 12 × 1024 = **491,520 tokens/iter**

---

### 6. 日志与环境设置 (L106-127)

**Tee类** (L109-120)：同时输出到终端和日志文件（`train.log`），核心是重写 `write()` 方法。

**随机种子** (L121)：`1337 + seed_offset`，DDP时每个进程种子不同，保证数据采样不同。

**TF32加速** (L122-123)：允许Ampere+GPU使用TF32格式（比FP32快，精度够用）。

**混合精度上下文** (L124-127)：

```python
ctx = nullcontext()                       # CPU时不做特殊处理
ctx = torch.amp.autocast(dtype=ptdtype)   # CUDA时用autocast自动混合精度
```

---

### 7. 数据加载 `get_batch()` (L129-146)

```python
def get_batch(split):
    data = np.memmap('train.bin'/'val.bin', dtype=np.uint16)  # 内存映射，不吃RAM
    ix = torch.randint(len(data) - block_size, (batch_size,))  # 随机选batch_size个起始位置
    x = stack([data[i:i+block_size] for i in ix])              # 输入：连续block_size个token
    y = stack([data[i+1:i+1+block_size] for i in ix])          # 标签：向右移一位
    x, y = x.pin_memory().to(device, non_blocking=True)        # 锁页内存→异步传输到GPU
```

核心设计：

- `np.memmap`：文件映射到内存，OS按需加载页面，不一次读完17GB
- 每次调用都重新创建memmap（避免内存泄漏，有StackOverflow链接说明）
- `pin_memory` + `non_blocking=True`：CPU→GPU传输与GPU计算重叠

---

### 8. 模型初始化 (L148-208)

三种模式：

#### `init_from='scratch'` (L164-172)

```python
model_args['vocab_size'] = meta_vocab_size or 50304  # 50304是256的整数倍，利于GPU
model = GPT(GPTConfig(**model_args))
```

- 优先从 `meta.pkl` 读 vocab_size（字符级数据集会带）
- 否则用 50304（GPT-2原版50257向上取整到256的倍数，利于GPU对齐）

#### `init_from='resume'` (L173-195)

```python
checkpoint = torch.load('ckpt.pt')
model = GPT(GPTConfig(**model_args))
# 处理torch.compile可能添加的前缀
for k in state_dict:
    if k.startswith('_orig_mod.'):  
        # 去掉这个前缀 
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
iter_num = checkpoint['iter_num']           # 恢复迭代数
best_val_loss = checkpoint['best_val_loss']  # 恢复最佳验证loss
```

#### `init_from='gpt2*'` (L196-203)

```python
model = GPT.from_pretrained('gpt2')  # 加载OpenAI官方权重
```

用于微调场景。`from_pretrained` 在 model.py 中定义，处理了 Conv1D→Linear 的转置。

**block_size裁剪** (L205-207)：如果指定的 block_size 比模型原版小，做"模型手术"裁剪位置编码。

---

### 9. 优化器与编译 (L210-228)

```python
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# float16时需要GradScaler防止下溢；bfloat16不需要

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# model.py中的实现：分decay/no-decay两组参数，CUDA上用fused AdamW

if compile:
    model = torch.compile(model)          # PyTorch 2.0 编译加速
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])  # 包装为分布式模型
```

---

### 10. `estimate_loss()` (L230-243) — 验证评估

```python
@torch.no_grad()
def estimate_loss():
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):          # 跑200个batch
            X, Y = get_batch(split)
            with ctx:                         # autocast上下文
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
```

- `@torch.no_grad()` — 不计算梯度，省内存
- 用200个batch的平均loss来估算，比单batch稳定得多
- train和val都评估，用于检测过拟合

---

### 11. `get_lr()` (L246-257) — 学习率调度

三段式cosine调度：

```
lr
│   /\
│  /  \
│ /    ‾‾‾‾‾‾‾
│/              ‾‾‾‾‾
└──────────────────── it
  warmup  decay   min_lr
```

1. **线性预热** (`it < warmup_iters`)：从0线性增长到 `learning_rate`
2. **余弦衰减** (`warmup_iters < it < lr_decay_iters`)：cos曲线从 `learning_rate` 降到 `min_lr`
3. **保持最小** (`it > lr_decay_iters`)：固定 `min_lr`

公式：`lr = min_lr + 0.5*(1+cos(pi*decay_ratio)) * (learning_rate - min_lr)`

---

### 12. 训练主循环 (L264-354)

整体流程：

```
while True:
    ① get_lr() → 设置学习率
    ② estimate_loss() → 验证 + 存档 (每eval_interval步)
    ③ 梯度累积的前向-反向传播
    ④ 梯度裁剪
    ⑤ 优化器步进
    ⑥ 日志：loss、耗时、MFU
    iter_num++
    if iter_num > max_iters: break
```

**关键细节**：

- **梯度累积** (L312-325)：for循环里每个micro_step算一个micro-batch的梯度，loss除以步数使得最终梯度等效于大batch
- **DDP梯度同步优化** (L313-318)：只在最后一个micro_step同步梯度，减少通信次数。直接操作 `require_backward_grad_sync` 变量，比用 `model.no_sync()` 上下文管理器更简洁
- **异步预取** (L323)：在GPU做forward时，CPU同时准备下一个batch
- **GradScaler** (L325)：float16时需要，先放大loss防下溢，再在optimizer.step前缩回来
- **MFU** (L345)：Model FLOPs Utilization，衡量GPU利用率，用PaLM公式估算，A100为基准

---

### 13. 收尾 (L356-375)

```python
# 保存loss历史为JSON（用于后续画图）
json.dump({'train': [...], 'val': [...]}, 'loss.json')

# DDP清理
if ddp:
    destroy_process_group()

# 恢复stdout，关闭日志文件
sys.stdout = sys.stdout.stream
```

---

### 整体数据流

```
配置文件/CLI → configurator.py覆盖全局变量
                    ↓
            模型初始化(scratch/resume/gpt2)
                    ↓
    ┌─── 训练循环 ◄──────────────────────┐
    │  get_lr() → 设置学习率              │
    │  estimate_loss() → 验证+存档        │
    │  get_batch() → 数据加载             │
    │  model(X,Y) → forward               │
    │  loss.backward() → backward          │
    │  clip_grad → scaler.step → 更新权重  │
    │  log: loss, time, MFU               │
    └────────── iter_num++ ──────────────┘
                    ↓ (达到max_iters)
            保存loss.json，清理退出
```

## 自己理解的数据流

## Q&A

### Q: 为什么 train.py 中直接把整数 numpy 丢给 model，而没有用 meta.pkl 做字符映射？

**A: 因为模型不需要知道字符是什么，它只处理数字。**

完整数据流：

```
训练时 (train.py):
  字符 → 整数 (prepare.py做好) → numpy bin文件 → model (直接用整数训练)

采样时 (sample.py):
  model输出整数 → 用meta.pkl反向映射 → 字符文本
```

关键机制：

1. **`nn.Embedding` 是数字查找表** (`model.py:127`)：`wte = nn.Embedding(vocab_size, n_embd)` 把整数索引映射成向量，不需要知道索引代表什么
2. **训练时只用整数** (`train.py:139-140`)：numpy bin文件存的是uint16整数，直接转int64给模型
3. **`meta.pkl` 只在采样时使用** (`sample.py:56-68`)：`stoi`/`itos` 用于字符↔整数的编解码

设计原因：**关注点分离** — prepare.py负责编码，model.py只学数字模式，sample.py负责解码。这样model可以处理任何tokenization方案（字符级、BPE、WordPiece）。

### Q: 所有现代LLM都用nn.Embedding吗？GPT-3不是把token映射为768维向量吗？这个查找表是怎么来的？

**A: 所有现代LLM都用nn.Embedding，这不是简化。向量是随机初始化后通过训练学到的。**

**1. nn.Embedding就是一个可训练的矩阵**

```python
nn.Embedding(50257, 768)  # 等价于一个50257行×768列的矩阵
# 查找操作 = 取出第token_id行
weight[token_id]  # 返回一个768维向量
```

**2. 向量不是预先定义的，是训练出来的**

- 初始化：随机正态分布（mean=0, std=0.02）— 见 `model.py:167-168`
- 训练中：通过反向传播更新矩阵的每一行
- 训练后：语义相近的token，向量也相近（分布假说）

**3. 各模型的Embedding规模**

| 模型 | vocab_size | n_embd | 矩阵大小 |
|------|-----------|--------|----------|
| GPT-2 | 50257 | 768 | 38M参数 |
| GPT-3 | 50257 | 12288 | 617M参数 |

GPT-3的n_embd是12288，不是768。768是GPT-2 small的配置。

**4. 为什么训练后向量有意义**

训练目标是预测下一个token。模型发现"猫"和"狗"常出现在相似上下文中，所以它们的向量会逐渐靠近。这就是分布假说：词的含义由上下文决定。

```
训练前：Embedding矩阵是随机噪声
训练后：vec("国王") - vec("男人") + vec("女人") ≈ vec("女王")
```

### Q: 50257是世界上所有字符的数量吗？Shakespeare只有65个字符，是否浪费参数？

**A: 50257不是字符数，是BPE子词数量。Shakespeare确实只训练65个embedding向量，模型会自动适配实际vocab_size。**

**1. 50257是BPE词表，不是字符数**

GPT-2使用BPE (Byte Pair Encoding)，不是字符级：
- 字符级："unbelievable" → ['u','n','b','e','l','i','e','v','a','b','l','e'] (12个token)
- BPE："unbelievable" → ['un','believ','able'] (3个token)
- 50257是OpenAI训练出的高频子词组合数量

**2. nanoGPT会自动适配实际vocab_size**

`prepare.py:55-56` 把 `vocab_size: 65` 存入 `meta.pkl`
`train.py:168-170` 读取后用实际大小初始化模型

如果错误地用vocab_size=50304训练Shakespeare：
- 实际训练：65×768 = 49,920个参数
- 浪费：50239×768 = 38,585,600个参数（99.87%）
- 能跑，但浪费显存和存储

**3. 结论**

模型参数量会根据实际vocab_size动态调整，不会浪费。

### Q: 为什么x和y等长（都是block_size），而不是y比x多一个token？block_size=1024相当于一次训练1024条数据吗？

**A: 是的。通过因果掩码，block_size=1024的一次forward等价于同时训练1024个预测任务。**

**1. 因果掩码让每个位置独立训练**

```
x = [t0, t1, t2, t3]  →  模型输出4个位置的预测
y = [t1, t2, t3, t4]  →  4个位置各自的预测目标
```

通过下三角因果掩码（`model.py:49-50`）：
```
位置0: 只看[t0]              → 预测t1  (1条训练数据)
位置1: 只看[t0, t1]          → 预测t2  (1条训练数据)
位置2: 只看[t0, t1, t2]      → 预测t3  (1条训练数据)
位置3: 只看[t0, t1, t2, t3]  → 预测t4  (1条训练数据)
```

一次forward，4个预测结果，一次反向传播更新权重。

**2. block_size=1024的实际含义**

一个序列产生1024个预测，batch_size=12就是12个序列，所以：
```
每个微步的训练样本数 = batch_size × block_size = 12 × 1024 = 12,288 个预测
```

这就是为什么大block_size能大幅提高训练效率——同样的forward pass，训练了更多的预测任务。

**3. 如果让x比y短一个token**

每次batch只训练一个预测位置，效率极低（慢block_size倍）。

**4. 核心思想**

x和y等长不是"少了一个目标"，而是"一次训练了block_size个预测任务"。因果掩码保证每个位置只能看到过去的信息，这就是Transformer并行训练的效率所在。

### Q: gradient_accumulation_steps=40，但每个micro_step都调用了backward()，这不是每步都在更新梯度吗？

**A: backward()只是累加梯度，optimizer.step()才更新权重。PyTorch的梯度默认累加，不清零就不重置。**

**1. 关键：梯度累加 vs 梯度清零**

```python
for micro_step in range(40):
    loss = model(X, Y) / 40       # loss除以40，保证累加后梯度正确
    loss.backward()               # 梯度累加，不清零！
optimizer.step()                  # 用累加的梯度更新权重
optimizer.zero_grad(set_to_none=True)  # 这里才清零
```

**2. 梯度变化过程**

```
micro_step=0: grad = g0
micro_step=1: grad = g0 + g1
micro_step=2: grad = g0 + g1 + g2
...
micro_step=39: grad = g0 + g1 + ... + g39

optimizer.step() → 用这40个梯度的和更新权重
optimizer.zero_grad() → 清零，下一轮重新开始
```

**3. 为什么要这样做？用时间换空间**

```
方案A: 一次forward 480个样本 → 需要超大显存
方案B: 分40次forward 12个样本 → 显存只需1/40

两种方案的梯度等价！
```

loss除以40是因为梯度会累加40次，不除的话最终梯度会是正常值的40倍。

### Q: 模拟大batch有什么优点？梯度不是及时更新更好吗？

**A: 大batch梯度更稳定、训练更可靠，GPU利用率更高。小batch更新频繁但噪声大，容易训练崩溃。**

**1. 梯度噪声问题**

```
小batch (12样本): 梯度 = 真实梯度 + 大噪声 → "醉汉走路"
大batch (480样本): 梯度 = 真实梯度 + 小噪声 → "清醒走路"
```

LLM训练很不稳定，小batch的梯度噪声容易导致loss spike，训练直接崩溃。

**2. GPU并行效率**

A100算力~312 TFLOPS，小batch只用一小部分算力，大部分空转。大batch更充分利用并行能力。

**3. 实际规模**

GPT-3: ~3.2M tokens/step
nanoGPT默认: 40×12×1024 = 491,520 tokens/step

**4. 总结**

| | 小batch | 大batch |
|--|---------|---------|
| 更新频率 | 高 | 低 |
| 梯度噪声 | 大 | 小 |
| 训练稳定性 | 差 | 好 |
| GPU利用率 | 低 | 高 |
| 显存需求 | 低 | 高 |

梯度累积是"用时间换空间"：显存不够装大batch，就分多次小batch算，累加梯度模拟大batch效果。

