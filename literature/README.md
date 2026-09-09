# 文献迁移说明

当前文献迁移只服务于上海话低层变调界限／AP边界预测。教学与实验转化见[学生学习文档](../docs/05_学生学习文档.md)及[变调界限预测验证卡](../docs/06_三套理论验证卡.md)：两份文档区分原论文主张、为教学所作的演绎，以及必须由本项目数据验证的推广。修订前Word和Markdown快照保存在本目录 `archive/`，当前文档位于 `../docs/`。

本目录的实验方案已查阅 `jieba/docs/GENERATIVE_PROSODY_THEORY_REVIEW_20260908.md`、`SENTENCE_SANDHI_FOCUS_REVIEW_20260908.md`、`TONE_RESEARCH_QUESTIONS_PROTOCOL_20260907.md` 和 `TONE_TOP20_CONSTRUCTIONS_20260908.md`。本轮只迁移其中能直接支持低层分域、词法边缘、音步长度、词汇化和数据设计的部分；旧标签只作候选线索，不能自动视为现行AP边界。原笔记仍保留在原位置，不在此目录复制整篇。

## 迁移后的共同结论

1. Selkirk与Shen提出的低层实词边缘可转化为AP边界候选特征；其高层MaP分析不进入当前实验。
2. Duanmu关于非焦点条件下音步、长度、语速和词汇化的讨论可转化为边界假设，但不能直接当作上海话全句规则。
3. Roberts的AP表示和上海话整句数据用于帮助定义、讨论和核查低层边界；`ip`、`IP`、Sh_ToBI高层事件和句末语调仅作归档背景。
4. 焦点和停顿不作为当前标签或操纵变量。明显依赖纠正／对比、强情绪或不流利停顿的材料进入 `excluded_or_future`，不把它们硬编码为普通负例。
5. 结构知识的收益必须通过L0至L4增量比较、陌生句族和陌生说话人测试来证明，不能用开发集命中或模型自己制造的标签循环论证。

## 文献表

详见上游笔记及 `jieba/generated/prosody_theory_20260908/source_manifest.json`。本轮重点来源包括 Roberts 2020、Selkirk & Shen 1990、Duanmu 2005、Sun & Chen 2015、Selkirk 2009公开稿、Xu & Prom-on 2014及郑秋豫等2007。Kratzer & Selkirk 2007原文尚未取得，暂不据其全文制定规则。
