# Transformer-Pytorch-Simple

从零手写复现 *Attention Is All You Need* (Vaswani et al., 2017)，不使用 `torch.nn.Transformer`。

## 目标

- 完整实现论文的 encoder-decoder Transformer
- 覆盖训练与推理全流程
- 语言对：英语 ↔ 德语（Multi30k），与论文主结果 WMT 2014 En-De 的设定一致

## 进度

- [x] Scaled Dot-Product Attention
- [x] Self-Attention / Cross-Attention
- [ ] Multi-Head Attention
- [ ] 位置编码、Position-wise FFN、残差 + LayerNorm
- [ ] Encoder / Decoder 堆叠
- [ ] 训练脚本（Noam 学习率调度 + label smoothing）
- [ ] 推理（greedy / beam search）与 BLEU 评估

## 环境

- Python 3.13
- PyTorch 2.13 (CPU)

## 数据

使用 [Multi30k](https://github.com/multi30k/dataset) 英德平行语料（29000 / 1014 / 1000 句对）。

语料不入库，需自行下载并放置为：

```
data/raw/multi30k/
├── train.jsonl     # 每行 {"en": "...", "de": "..."}
├── val.jsonl
└── test.jsonl
```

## 文件

| 文件 | 说明 |
|---|---|
| `attention.py` | Scaled Dot-Product Attention、Self-Attention、Cross-Attention |
