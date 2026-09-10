# 小型上海话phone CTC训练入口

> 2026-09-11最终状态：v4新增6000步早停，未改善v3最佳；当前采用v3第6750步。下文启动与续训记录为历史配置。启动器包含当时目录和截止日期，重新运行前必须配置本次路径与截止时间。

本脚本只训练无调phone预测模型，结果供后续人工验收与对齐研究；**CTC发射位置不等于完整phone/音节边界**。不读test选参，不把读音数字后缀变成预测目标，不覆盖重要录音索引。

## 模型和依赖

默认骨干：[TencentGameMate/chinese-wav2vec2-base](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base)，固定revision `3991242c806928916fff4a8c0e4f76acf661b743`。模型卡声明MIT、用WenetSpeech L约1万小时音频自监督预训练；12层、hidden 768，约95M参数。首次替换为45个项目phone＋blank的CTC头，初始配置先训头100步再开放最后2层；当前v4复用已训练46维头与最后6层，具体配置见下文。中文预训练不等于已适配上海话，约19分钟短词数据也不保证自然会话泛化。

只需下载 `pytorch_model.bin`（380,261,837 bytes）、`config.json`、`preprocessor_config.json`、`README.md`，不需要1.14GB fairseq checkpoint。权重SHA256：`be2da40c9e7ae26bfc904a3ed79ebb9e8f060bec6dba85d6a6ae86114bc38901`。模型仓库没有tokenizer；本项目直接用冻结的 `vocab.json` phone ID训练。

云端先使用兼容GPU的PyTorch环境，其他依赖见同目录 `requirements_phone_ctc.txt`。无fairseq、Whisper、datasets、torchaudio或云API依赖。需要Torch≥2.6读取旧 `.bin` 权重；默认Transformers固定4.57.6。日志记录实际版本。

最初600步、最后2层的试验配置保留作历史示例；当前训练器硬参数上限30000步/7200秒，实际v4新增上限21750步/3600秒。云端RTX4090 24GB已运行最后6层、batch8×累积2，启动快照GPU占用2466MiB，不代表全过程峰值。实际预算还受云端下载/准备/计费影响，租机方必须另有总20元限额和停机机制；本脚本**不管理实例计费或关机**。

## 输入

`--train` / `--dev`：JSONL，每行至少包含：

```json
{"id":"0-01","audio_path":"corpus_16k/0-01.wav","phones":["tɕ","y"],"phone_ids":[1,2],"group_id":"example","split":"train"}
```

上例ID仅为结构示意，真实ID由 `dataset/vocab.json` 决定。`audio_path` 相对 `--audio-root`（默认manifest父目录的父目录），音频必须为16kHz单声道。脚本独立归一化每条推理输入，不改WAV。严格检查phone ID、未知phone、blank误入目标、train/dev组重叠、重复ID和CTC最小可行帧数；只加载train/dev，test不传入脚本。

## 当前v4权重续训（2026-09-10）

v3已完成8000步、713.39秒，最佳6750步组平均PER 4.37283%、普通PER 4.71545%，末步分别5.73018%、6.01626%。训练loss继续下降并未保证开发集改善，v4因此从最佳模型以十分之一学习率继续试验。`export_phone_checkpoint.py`严格导出v3 best.pt至云端`/workspace/am_project/models/v3_best_step006750/`；不重新初始化46维phone头。新优化器、调度器和RNG重置，这是权重warm-start；原配置的严格`--resume`不能用于改变max_steps或LR调度，且v3完成时LR已衰减为0。

`launch_cloud_continuation_v4.sh`使用最后6层、头部3e-5/编码器3e-6、batch8×累积2、warmup100，新增最多21750步；父有效路径1500+6750=8250步，累计上限30000步，不计v1及被放弃的v3尾1250步。`--initial-eval`在第0步评估并保存基线best，每500步评估/保存，patience12、min_delta 0.0005。内墙钟3600秒、外timeout3720秒+kill120秒，并设2026-09-11 00:50上海硬截止；平台按用户设定01:00关机，单价1.9元/小时、总预算20元。

实际于23:37:38上海启动，23:40:47核验training2258/21750步、loss有限；初始dev精确复现v3最佳指标。云运行`/workspace/am_project/runs/phone_ctc_v4_20260910/`，本地镜像`runs/aligner_v1/phone_ctc_v4_20260910/`。训练由nohup独立运行，`watch_cloud_training.py`镜像最新状态与指标，`show_training_progress.ps1`显示本轮step/目标、loss、latest PER与best group PER；关闭可见窗口不停止云训练。test保持封存，边界质量仍待验收。

## 初始运行示例（历史配置）

以下以已经上传的包根 `/workspace/aligner_v1` 为示例，实际云路径由部署端填写：

```bash
python train_phone_ctc.py \
  --train /workspace/aligner_v1/dataset/train.jsonl \
  --dev /workspace/aligner_v1/dataset/dev.jsonl \
  --vocab /workspace/aligner_v1/dataset/vocab.json \
  --audio-root /workspace/aligner_v1 \
  --output /workspace/runs/phone_ctc_v1 \
  --device cuda --max-steps 600 --max-wall-seconds 7200 \
  --batch-size 8 --grad-accum 2 --unfreeze-last 2 --head-only-steps 100
```

头部学习率3e-4、编码器1e-5、50步学习率warmup、AdamW、clip grad=1.0；关闭SpecAugment时间掩码以避免极短词无可用帧。每50步checkpoint，dev同时输出总体phone edit distance/参考phone数和按读音组平均的PER；**best仅依据dev group macro PER**，不使用test。

CPU流程检查：同样命令改用新输出目录并添加 `--smoke-test`。它使用极小随机初始化网络、前4条train/dev、只跑1步，完全不下载预训练权重；生成的模型不能用于语音学分析。正式训练前可核验数据接口、梯度、日志和保存路径。

断线后台由云端用tmux/nohup或作业管理器启动，本脚本持续写状态。SIGTERM/SIGINT请求在当前训练步后保存并停止。续跑使用相同数据与配置，增加 `--resume /workspace/runs/phone_ctc_v1/last.pt`；只加载本脚本生成的可信checkpoint。步数和累计运行时间随checkpoint恢复，不借重启清零预算。参数检查之外，保存与正在执行的一个forward/evaluation可能有收尾开销，云端仍需独立设置截止停机。

## 输出

- `status.json`：starting/training/finished/failed、步骤、停止原因、当前时间与未验收标记；原子替换。
- `metrics.jsonl`：逐步loss、梯度norm、学习率与dev PER，不把softmax称为准确率。
- `last.pt` / `best.pt` / `final.pt`：完整模型、优化器、LR/scaler、RNG、shuffle位置、数据哈希，可恢复；临时文件写完后原子替换。
- `final_model_stepNNNNNN/`：末步HF可加载safetensors、config、feature extractor和phone词表；`final_model.json`指向它。**这是末步模型，验证最优模型为 `best.pt`**。
- `best_dev_predictions.json`：最佳验证轮次逐条无调phone参考与预测；`last_dev_metrics.json`、`run_config.json`、`vocab.json`保留统计和配置。

`finished`只表示训练正常停止（达到步数、早停、预算或接到停止信号），不表示音素边界已人工验收。未通过训练结构/有限值检查时写failed，不静默丢弃文件或把CTC无限损失归零。

## 本地流程QA（2026-09-10）

在既有 `qwen3-asr` 环境使用真实train/dev JSONL和tiny随机模型验证：1步CPU训练、梯度裁剪、4条dev总体/分组PER、周期last/best/final保存及safetensors导出均通过。重复同seed的smoke得到相同loss；从last续跑保持step=1，已存在final目录不会碰撞。另用仅用于QA的“已用满7200秒”checkpoint检验恢复预算，step=0直接以wall_time_limit收尾，没有新增训练步骤。final safetensors逐张量与last checkpoint一致，RNG与数据哈希均存在。

初次QA产物在 `generated/aligner_v1/training_smoke_cpu_final/` 与 `training_smoke_deadline/`。这些是**随机微型模型和模拟deadline fixture**，不上传为预训练/正式模型，也不把smoke的PER解释为上海话识别性能。该次CPU QA未下载95M权重；后续云端已核验真实骨干兼容性。v4新增的initial-eval→第0步best→一步训练→checkpoint/export烟测通过；实际云端第0步评估精确复现父模型最佳PER。当前训练脚本本地/云端SHA-256一致：`bd8de9e97512e383d2f569fc280218ee6a384f610e6ebc8c2f99cebf62cf0b8f`。
