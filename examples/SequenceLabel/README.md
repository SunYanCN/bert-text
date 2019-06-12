## 命名实体识别任务描述：
命名实体识别(Named Entity Recognition, NER)，主要任务是识别出文本中属于特定类别的词语并加以归类。
目前较为常见的处理方法是，将NER任务抽象为序列标注问题。

例：<br> 
首先定义一个标签集合：{B-PER，I-PER，I-PER，O}<br> 
给定文本序列：
>周杰伦的教育经历是怎么样的？

若当前任务为识别文本中的人名，则对应的标签序列为：
>B-PER I-PER I-PER O O O O O O O O O O O

归类时，将BII的组合归为同一类：人名（PER）

## MSRA NER Dataset

语言：简体中文
编码：utf-8

**包含专名**：

    |标签| LOC | ORG | PER |
    |----|-----|-----|-----|
    |含义|地名 |组织名|人名|

**训练集**：

    |  句数  |  字符数  |  LOC数  |  ORG数  |  PER数  |
    |--------|----------|---------|---------|---------|
    |  45000 | 2171573  |  36860  |  20584  |  17615  |

**测试集**：

    |  句数  |  字符数  |  LOC数  |  ORG数  |  PER数  |
    |--------|----------|---------|---------|---------|
    |  3442  |  172601  |  2886   |  1331   |  1973   |


**标注格式**：

	[字符]	[标签]	# 分隔符为"\t"

	其中标签采用BIO规则，即非专名为"O",专名首部字符为"B-[专名标签]"，专名中部字符为"I-[专名标签]"

	例如：

		历	B-LOC
		博	I-LOC
		、	O
		古	B-ORG
		研	I-ORG
		所	I-ORG

## 数据来源

1. [SUDA-HLT/NewStudents](https://github.com/SUDA-HLT/NewStudents)

2. [paddlehub](https://github.com/PaddlePaddle/PaddleHub)

3. [Data URL](https://paddlehub-dataset.bj.bcebos.com/msra_ner.tar.gz)

## 目前存在问题
鉴于我对这一任务不是很了解，目前还存在一些问题
- 代码问题
```python
UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
```
~~尚不清楚警告的来源，大约是来自tf.gather函数，还需要查找警告的原因和影响。~~ 已解决，不知道改了啥突然好了φ(≧ω≦*)♪
- 评价问题
使用的是`seqeval.metrics`，效果很差，但是正确率很高，尚不清楚是模型有问题还是评价函数有问题
