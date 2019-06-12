# BERT+ TF Keras For NLP Tasks
**说明**：本项目处于开发阶段，暂时不可用

**简介**：以Tensorflow的Keras和Tensorflow hub的Bert预训练模型开发NLP的各种任务。
## 环境
- Tensorflow:1.13.1
- Tensorflow-hub

## Tasks
### 文本分类
### 序列标注
### 阅读理解

## 模型保存和加载
最好的方式是可以使用`tf.keras.experimental.export_saved_model`导出模型，然后直接使用TF serving部署

