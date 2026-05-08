# BeautySID-Rec

这是一个基于 Amazon Beauty 数据集的生成式推荐实验项目，整体思路参考 GRID / TIGER 一类“语义 ID + 生成模型”的路线，当前仓库保留的是**最后一次跑通并取得当前最优结果**的版本。

## 当前最佳结果

直接使用保存好的 `best_model.pt` 跑 `test`，最新测试结果如下：

- `Recall@5 = 0.02244`
- `Recall@10 = 0.03708`
- `NDCG@5 = 0.01314`
- `NDCG@10 = 0.01787`

对应配置可以概括为：

- 文本编码器：`BAAI/bge-base-en-v1.5`
- tokenizer：`RQ-VAE`
- 语义 ID 结构：`(16, 32, 64, 128) + 第 5 位 de-dup token`
- 生成模型：`seq2seq encoder-decoder Transformer`

## 项目流程

```text
商品文本 -> 文本编码器 -> item embedding -> RQ-VAE 量化为 SID -> de-dup 消解冲突
-> 用户历史 SID 序列 -> encoder-decoder Transformer -> beam search -> Recall/NDCG 评估
```

## 目录结构

```text
configs/      配置文件
scripts/      运行入口
src/          核心实现
outputs/      checkpoint 与评估结果
PROJECT_SUMMARY.md   项目总结文档
```

## 运行环境

建议优先使用：

```powershell
E:\Anaconda\envs\torch_gpu\python.exe
```

安装依赖：

```powershell
E:\Anaconda\envs\torch_gpu\python.exe -m pip install -r requirements.txt
```

## 主要脚本

运行 tokenizer：

```powershell
E:\Anaconda\envs\torch_gpu\python.exe scripts\run_tokenizer.py --config configs\beauty_baseline.yaml
```

运行训练：

```powershell
E:\Anaconda\envs\torch_gpu\python.exe scripts\run_train.py --config configs\beauty_baseline.yaml
```

运行测试：

```powershell
E:\Anaconda\envs\torch_gpu\python.exe scripts\run_eval.py --config configs\beauty_baseline.yaml --split test
```

## 当前版本说明

这次提交主要保留：

- 最后一版可跑通代码
- 最优配置文件
- `best_model.pt`
- `test_metrics.json`
- 项目总结文档

没有把完整原始数据与中间处理结果一起放进仓库，以避免仓库体积过大；后续如需完整复现，可根据代码重新下载与生成。
