测试结果如下：

- `Recall@5 = 0.02244`
- `Recall@10 = 0.03708`
- `NDCG@5 = 0.01314`
- `NDCG@10 = 0.01787`

对应配置为：

- 文本编码器：`BAAI/bge-base-en-v1.5`
- tokenizer：`Balanced K-Means`
- 语义 ID 结构：`(16, 32, 64, 128) + 第 5 位 de-dup token`
- 生成模型：`seq2seq encoder-decoder Transformer`

## 项目流程

```text
商品文本 -> 文本编码器 -> item embedding -> RQ-VAE 量化为 SID -> de-dup 消解冲突
-> 用户历史 SID 序列 -> encoder-decoder Transformer -> beam search -> Recall/NDCG 评估
```



