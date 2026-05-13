
# Day-1

## 首次跑通工作流
### 训练

#### 执行命令
```bash
python train.py config/train_shakespeare_char.py --compile=False
```
发现配置max_iter=5000, 在个人电脑上运行大概耗时10min，并且自动化套件似乎没有自动收集loss曲线，自己手动添加套件

#### 运行结果
暂未保存

### 推理
#### 执行命令
```bash
python sample.py --out_dir=out-shakespeare-char
```
#### 运行结果
```
Overriding: out_dir = out-shakespeare-char
number of parameters: 10.65M
Loading meta from data\shakespeare_char\meta.pkl...


ANGELO:
And come, my lord,
Will I stay before.

ISABELLA:
Sir, I say, it is not a brother distrossess.

ANGELO:
Do not more penitently, first it is the winter
With every contraction. What says your presence?

ISABELLA:
For all the marner,
You speak so low you were sent for this desire
In world than your countrymen? you will to make him short
A poor great resolve of three-pile arm
To the flower of Polixenes, Prince, Sir, and Pompey,
Your same daughters, he may in the cause.

CAPULET:
Ay, a comfo
---------------

Men pardon me, you shall have hang your good manner.

GLOUCESTER:
Resolve you, sir, let me desire you from your honour,
I make something a stronger to be your tongue.

PRINCE:
You meet, sir, sir, sir, that Richard: there loves
The heart-slaughter, and the heart beards as yours.

WARWICK:
So shame with these lasts, this arms so I say.

GLOUCESTER:
Where she is?

GLOUCESTER:
She shall show the submission that we have show'd
That prives me made a dishonour'd by the chamber;
But if you should have e
---------------

MARIANA:
I would fly, I have no word.
My lord, I do dispatch; I am not your suit:
But yet so much so well were not the story ground;
And yet I call your army womanIng
And all in the particular and keep
You more than your side, for not my hands.

DUKE VINCENTIO:
He would not have less'd of grace
To the world and from his murderer.
Ah, he should this stay as lost thou hast quickly
But Henry's fault, Cristing Froth, the babe,
Only was his neighbour to her enemy.

DUKE VINCENTIO:
My lord, whose sons
---------------

The senators will know
What he were prized the action of her
Of the seen your brother's praise, and well prevented
My helps with the stood trumpets are and all the arm.
I am sorry for my heart: and I'll stand thee to the brave,
That bears not so many be corrupted, to our knee,
As look the king in safety and all to ours.

NORTHUMBERLAND:
No, gentle Lord Hastings, and Saint George Guild,
Hath such made a resemblance by the Edward
And stop partices brief the nor father mates me on the bride!

KING 
---------------

The more belly, the bloody of Troy
For in reason and noble fashion and unmust have
A daughter not on his song a world whole.
If my liege, and many wish, both his wife;
And she was a heir of death at my his course;
But now I have done them done.

PAULINA:
Ay, commanded them not that we have something done.

ISABELLA:
O, and turn that had ne'er dismiss'd the state matcher,
And the blood of great parting for her own soul,
But which he had affer'd it for your jest decrees,
That ever strength and for
---------------


MENENIUS:
Have been so to acquaint, their
approaching of a wife, they are now for the people, but I
have done so attended the thing shoes of the court
does and slept for our pastimes of weight will inter need;
therefore is, if thou hast been a good condemn
to the duke and sense, in thy tongue's head
and so the soldier, in a kingly feebles, till we are none
the proof in the sentency of that winds of the least.

CORIOLANUS:
Have you so?

BRUTUS:
We drawn'd him, and report to serve it
To Rome it?

---------------

She would be married to have fill'd the war, and had not to it
more than all the place of a shrewd: yea, I have done there
not the senator of death, but righting to have ourself
sound and rage to the sight of despite, they swear the house
of this head, have been or no fair man, so weed the world
spite of the rest which mates of them; and their
father tenrous to the walls, we show their faces, their own
service, that tender to purfear their loves, the care gavexped
ones, that we did slipt a souls
---------------

ladies, be readying to me, or with no; and there were
I say the while I have there no more,
That he bores and love my neck wind with mine.

FRIAR LAURENCE:
The gather duke down ye hold his bodies,
She had words doing to beat a serious grave dog soundly dog!
That hear me tell me, Henry's dear mourners
A little time so a weague from here:
The noble eyes ransom with the death-bosom'd to do
Her troubled daughter!

QUEEN MARGARET:
Petructioner, Claudio, do me from off?

WARWICK:
The world she cannot 
---------------


LEONTES:
The realm be from the grave. But, come old,
I know not the common will what I said; so slug
To be impatient. The sorrow of men.

PAULINA:
Not then, my lord,
But they have some exiled.

FLORIZEL:
No, I have spoken in my soul rest
That many is out made, and together
Than my number then must be daily.

DUKE VINCENTIO:
You should forget the harden.

Provost:
An it is not your grace.

JULIET:
Your honour daughter.

DUCHESS OF YORK:
I am sorry.

DUKE OF AUMERLE:
Come, my mountain! how sir?


---------------

How cheer your sorrow?

GLOUCESTER:

CLARENCE:
Ha?

DUKE VINCENTIO:
Norfolk, how can you have gave them not?

LUCIO:
Here comes your name?

ISABELLA:
My lord, my lord, I spoke her to do it.

LUCIO:
Why should you read an army storm of such love?

QUEEN:
I have like off of all noble company.

DUKE VINCENTIO:
Why, then we first the news?

ISABELLA:
You have a voiced with his bones,
And every committed some to a substitute.

DUCHESS OF YORK:
We will ashoot for your subjects.

DUKE OF YORK:
Why, Cla
---------------
```

## 对比实验

*sample运行结果存在各自文件夹中的sample.txt中*

### 运行100次

1. 运行
```bash
python train.py config/train_shakespeare_char.py --compile=False  --max_iters=100 --lr_decay_iters=100 --out_dir=out-shakespeare-char-100 --eval_interval=5 --eval_iters=4 --warmup_iters=10
```

2. 绘制曲线
```bash
python plot_loss.py --loss_file=out-shakespeare-char-100/loss.json --out_file=out-shakespeare-char-100/loss_curve.png
```

3. 运行对比
```bash
python sample.py --out_dir=out-shakespeare-char-100
```


### 运行500次

1. 运行
```bash
python train.py config/train_shakespeare_char.py --compile=False  --max_iters=500 --lr_decay_iters=500 --out_dir=out-shakespeare-char-500 --eval_interval=25 --eval_iters=20
```

2. 绘制曲线
```bash
python plot_loss.py --loss_file=out-shakespeare-char-500/loss.json --out_file=out-shakespeare-char-500/loss_curve.png
```

3. 运行对比
```bash
python sample.py --out_dir=out-shakespeare-char-500
```

### 运行5000次

```bash
python train.py config/train_shakespeare_char.py --compile=False --max_iters=5000 --lr_decay_iters=5000 --out_dir=out-shakespeare-char-5000 
```

2. 绘制曲线
```bash
python plot_loss.py --loss_file=out-shakespeare-char-5000/loss.json --out_file=out-shakespeare-char-5000/loss_curve.png
```
3. 运行对比
```bash
python sample.py --out_dir=out-shakespeare-char-5000
```

## Q&A

1. 提示词：
默认情况下，提示词就是一个换行符 \n，非常短。这意味着模型基本上是从"空白"开始自由生成。
也可以通过命令行覆盖它：
python sample.py --start="ROMEO:"          # 传入一段文字
python sample.py --start="FILE:prompt.txt" # 从文件读取

2. 三次训练结果对比评估：

| 训练步数 | 最终train loss | 最终val loss | 生成质量 |
|---------|---------------|-------------|---------|
| 100     | ~2.49         | ~2.47       | 完全不可读，随机字符 |
| 500     | ~1.69         | ~1.79       | 有单词雏形，但不连贯 |
| 5000    | ~0.81         | ~1.72       | 基本可读，有对话结构 |

具体分析：

(1) 100步（loss ~2.5）：输出完全是乱码，没有任何可识别的英文单词。
示例："I trid owind t son, be matisere obe t eravegrth my delatange"
结论：模型只学会了字符频率分布，还没学到字母组合规律。

(2) 500步（loss ~1.7）：开始出现零星可识别的单词和人名，但句子不连贯。
示例："He madies my be dong magraves me?" "I make me to their meent being"
结论：模型学会了常见字母组合和基本拼写，但语法和语义还差很远。

(3) 5000步（loss ~0.81）：输出已经基本可读，有莎士比亚风格的对话结构。
示例："KING RICHARD III: The law of England's blood: and that the gate / My Lord Hastings, he that did him entreat your face"
结论：模型学会了人名格式、对话排版、基本语法和部分词汇搭配，但仍有语法错误和语义不连贯之处。

注意：5000步时val loss（1.72）明显高于2000步时的val loss（1.48），说明模型出现了过拟合——训练loss持续下降，但验证loss反而上升。如果要继续训练，应该加入正则化或早停策略。

3. Token：字符级 tokenizer 把每个字符映射成什么？

见 `data/shakespeare_char/prepare.py`：
```python
chars = sorted(list(set(data)))  # 提取所有不重复字符，排序
stoi = { ch:i for i,ch in enumerate(chars) }  # 字符→整数
itos = { i:ch for i,ch in enumerate(chars) }  # 整数→字符
```
莎士比亚数据集共有 65 个不同字符（26个小写 + 26个大写 + 空格 + 标点），每个字符被映射为 0~64 的整数。
例如：`'H'→20, 'e'→43, 'l'→50, 'l'→50, 'o'→53`。
这些整数再通过 `nn.Embedding(65, n_embd)` 被查表转换为向量（如 384 维），送入 Transformer。

4. Loss：CrossEntropy loss 在衡量什么？

见 `model.py:187`：
```python
logits = self.lm_head(x)  # (b, t, vocab_size) — 每个位置对65个字符的预测分数
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
```
模型的任务是**预测下一个字符**。给定 `x = "HELL"`，目标 `y = "ELLO"`。
- 在位置 0，输入 `H`，要求预测 `E`
- 在位置 1，输入 `HE`，要求预测 `L`
- 在位置 2，输入 `HEL`，要求预测 `L`
- 在位置 3，输入 `HELL`，要求预测 `O`

CrossEntropy 衡量的是：模型给正确字符分配的概率有多高。
- loss = 4.28（初始值）→ 模型对 65 个字符几乎均匀分配概率，完全在瞎猜
- loss = 2.5 → 模型开始偏好某些字符
- loss = 1.0 → 模型相当准确，但仍有错误
- loss = 0 → 完美预测（实际不可能达到）

直观理解：loss 从 4.28 降到 0.81，意味着模型预测正确字符的概率从 ~1.4% 提升到了 ~45%。

5. 为什么"像语言"：模型并没有学"语法"，它学的是 token 之间的统计规律

模型只做一件事：给定前面的字符序列，预测下一个字符的**概率分布**。
- 训练时，它看到大量 `"KING"` 后面跟 `" "`、`"\n"` 后面跟大写字母、`","` 后面跟空格等模式
- 这些模式完全是从数据中**统计出来的**，没有人告诉它"英语语法"或"莎士比亚风格"
- 它学到的只是：在某个上下文下，哪个字符出现的概率更高

100步时 loss~2.5，输出乱码，因为模型只学到了单字符频率（如 'e' 比 'z' 常见）。
500步时 loss~1.7，出现单词雏形，因为模型学到了常见字母组合（如 'th'、'he'、'ing'）。
5000步时 loss~0.81，看起来像莎士比亚，因为模型学到了更长范围的统计规律（如人名后跟冒号、对话格式、常见短语搭配）。

所以"像语言"只是统计规律的副产品——当统计规律足够丰富时，看起来就像理解了语言。这正是大语言模型的核心原理。

