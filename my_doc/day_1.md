
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