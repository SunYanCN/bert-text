# BERT+ TF Keras For NLP Tasks
**说明**：本项目处于开发阶段，暂时不可用

**简介**：以Tensorflow的Keras和Tensorflow hub的Bert预训练模型开发NLP的各种任务。

**项目特点**
- 下载稳定，较为典型的测试数据，附带详细介绍
- 数据接口封装，减少数据的处理工作
- 模型采用tf.keras完成，方便快捷不失灵活
- 方便的保存模型和部署

## 环境
- Tensorflow:1.13.1
- Tensorflow-hub

## Tasks
### 句子向量化
- [ ]  ~~但是存在一个重复加载计算图导致速度变慢的问题，体验不佳，后续将改进。~~ 
- [ ] 性能测试
- [ ] Batch输入
### 文本分类
- [x] 二分类任务

已完成。数据量不大的情况下可以不使用微调，否则参数量增大可能会过拟合。
- [ ] 多分类任务
BERT的多分类任务和二分类类似，只需要修改标签，然后sigmod换softmax,损失函数也换成多分类交叉熵即可。目前没有稳定的外链数据，所以没写example。
- [ ] 多标签任务

### 序列标注
- [ ] NER

开发中
### 阅读理解
- [ ] 斯坦福SQUAD类似的中文检索式阅读理解

未完成
## 模型保存和部署
- [x] h5转saved_model。
- [ ] 最好的方式是可以使用`tf.keras.experimental.export_saved_model`导出模型，然后直接使用TF serving部署。

## TF Data输入
- [] 多输入的TF Data写法
- [] GPU利用率的比较

## 其他说明
- 为什么用tf.keras而不是keras
tf.keras成为TF2.0的主要模式，由TF团队开发，支持更多TF的特性包括tf.data以及tf serving，TF2.0出了之后项目会迁移到TF2.0，而keras已经较长时间没有重大更新了，所以tf.keras是更好的选择。
- 由于网络原因，无法下载Tensorflow hub的BERT模型
这里有一份百度云的，[链接](https://pan.baidu.com/s/1Gm9Hcs4ysJGITKUoPZJxNg)， 提取码:4pcq，大小为364.1M，linux下载解压后拷贝到`/tmp/`，完成后的路径为`/tmp/tfhub_modules`
 - [ ] Tensorflow hub加载本地路径
 - [ ] Windows测试

## 参考项目
- [keras-bert](https://github.com/strongio/keras-bert)
