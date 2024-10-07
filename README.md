# SAPC-PLM
`基于结构信息感知预训练模型的专利分类方法研究` 的Python实现.

针对专利数据中的结构化信息，利用编码器－解码器网络架构，在预训练过程中融合专利长文本的结构特征，从而增强了预训练模型对专利技术特征的表征能力

## Dependencies
- Python >= 3.8
- transformers >= 4.38.0
- torch >= 1.7.1

## Usage
This project consists of two main steps: preprocessing the binary file, and then training the encoder for instruction sequence using DiffBCE. Below are the details on how to run each step:

### 数据集
预训练以及专利分类测试数据集下载链接. https://cail.oss-cn-qingdao.aliyuncs.com/cail2019/CAIL2019-SCM.zip

# 项目说明
`Bert-finetune`是各类Bert模型的微调代码
`Llama2-finetune`是各类Llama模型的微调代码
`Structure-PLM-ChatGLM`是给Llama做增量预训练的代码，包括本文的融合结构信息的预训练与传统的因果语言建模的预训练
