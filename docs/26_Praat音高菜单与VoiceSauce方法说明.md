# Praat音高菜单与VoiceSauce方法说明

更新：2026-09-14。问题：截图各项含义；VoiceSauce具体采用什么F0算法；能否排除黎里论文的边界测量偏差。截图为用户的 `D:/Users/jim/Pictures/捕获.PNG`。官方网页和所读源码存于 `generated/praat_voicesauce_20260914/`；本轮没有运行或改动声学提取。

## 截图逐项

| 菜单文字 | 含义 |
|---|---|
| Show pitch | 显示／隐藏音高曲线；勾选表示显示。这里主要显示声学F0估计。 |
| Pitch methods and settings | “音高分析方法与设置”，是分组标题。 |
| How to choose a pitch analysis method | 打开官方方法选择帮助。 |
| Pitch analysis method is filtered autocorrelation | 选择滤波自相关。 |
| Pitch settings (filtered autocorrelation)… | 打开滤波自相关的参数窗口。 |
| Pitch analysis method is raw cross-correlation | 选择原始互相关；截图中此项勾选，当前采用它。 |
| Pitch settings (raw cross-correlation)… | 打开原始互相关的参数窗口。 |
| Pitch analysis method is raw autocorrelation | 选择原始自相关。 |
| Pitch settings (raw autocorrelation)… | 打开原始自相关的参数窗口。 |
| Pitch analysis method is filtered cross-correlation | 选择滤波互相关。 |
| Pitch settings (filtered cross-correlation)… | 打开滤波互相关的参数窗口。 |

四条“method is”是互斥的方法选项；不是四个接续执行的处理步骤。截图没有显示floor、ceiling／top、步长等实际参数。这个Praat界面的选择不能改变项目里已生成的HTML散点。

## 四种方法的区别

自相关和这里的互相关都比较同一声音与其时间错开版本的相似程度，寻找重复周期；这里的互相关不是拿两名发音人比较。Praat的自相关对加窗引入的影响作补偿；互相关使用forward cross-correlation，局部分析时间尺度较短。

raw表示没有额外的前置低通，不表示不加窗、不作归一化或候选路径选择。filtered表示先对声音信号作Gaussian低通，再估计周期，不是把算出的F0曲线磨平。

| 方法 | 官方用途与限制 |
|---|---|
| filtered autocorrelation | 自2023年起优先推荐用于语调／声带振动频率。减少部分共振峰过渡、倍频错误，仍可能误判。 |
| raw cross-correlation | 优先用于voice analysis，如配合后续脉冲分析计算jitter、shimmer；F0轨迹本身不是这些指标。短时间尺度也不保证onset准确。 |
| raw autocorrelation | 传统语调分析方法，适合考察原始周期性。 |
| filtered cross-correlation | 低通后用互相关。手册明确尚无已知场景优于其余三种，不应因多了filtered就认为最好。 |

来源：22号[69]。raw CC手册所称分析窗为1/floor；不要把它直接当成包括全部延迟比较的完整信号支撑。步长与分析窗是两回事。

## VoiceSauce是软件，F0后端可以不同

UCLA官方Parameters Measured的F0节列出四种输出：

| 输出 | 后端 | 本轮核实的细节 |
|---|---|---|
| strF0 | STRAIGHT | 默认用于定位谐波、计算谐波幅度。v1.28（2016-12-23）起换为TANDEM-STRAIGHT中的XSX；此前是Multicue／NDF，并有失败回退。 |
| sF0 | Snack | v1.37的func_SnackPitch.m明确调用`-method esps`，并传入窗口、步长、上下限。 |
| pF0 | Praat | v1.37调用脚本接受ac或cc，使用旧To Pitch (ac)/(cc)，对应原始自／互相关；另有倍频跳变处理、平滑、插值开关。不能用2023年新增的默认filtered AC倒推2020年包的行为。 |
| shrF0 | Sun的SHR法 | 以次谐波与谐波的关系选择F0候选，涉及倍周期；所读shrp.m包含分帧及阈值设置。 |

Settings还支持Other外部算法。软件默认用于谐波分析的strF0，不等于论文一定选strF0作最终F0统计。

STRAIGHT也不能仅凭名称当作Praat四种方法之一。本轮核实v1.37的包装函数依次调用候选提取、轨迹跟踪、有声判断；核心候选提取为p-code，本轮未读其内部实现，不能宣称已复现XSX的全部数学细节。

官方所述1ms是输出间隔。源码`params.framePeriod = 1`证实这一点，不证明只读取1ms声音；v1.37包装函数还对足够长的F0输出做9点移动平均。手册中谐波幅度的默认“三周期窗”、Snack的窗口与此F0后端不能混为一谈。Settings中的Straight Max duration是长文件分块长度，也不是分析窗长。

## 对黎里研究的解释与有限检验

Shi、Bai、Chen（2024）p.240交代VoiceSauce自动提取，但未交代后端、版本、F0参数及边界纠错协议。软件可追查，研究所用配置仍不确定。不能认定论文用了当前默认XSX，也不能把其结果解释为已排除窗口效应。

统一软件与显著统计差异都不能排除与声母类别共同变化的系统偏差；不同起声、噪声、周期规则度可能改变估计。反过来，存在偏差可能也不等于论文结果就是伪影。H1*−H2*的星号是谐波幅度校正，不是F0边界校正。

若继续验证，先限于现有“早”和两处“正”，不扩展招募：H测量假设是起始升高主要依赖窗口／算法；H声学假设是多个相邻真实周期本身变长，支持实际F0下降。固定同一音频上下文和范围，对比方法与合理floor设置，同时人工标记最初5–10个可辨周期、记录歧义。检验首段相对后段的差值、持续时间是否随参数明显改变，是否与周期时长变化一致。混合结果按不确定记录，不以算法多数表决当真值；三个例子不能验证黎里总体结论。该建议未在本轮执行，既有三例诊断见25号。

## 来源和阅读边界

- Praat官方方法手册：[方法选择](https://praat.org/manual/how_to_choose_a_pitch_analysis_method.html)、[filtered CC](https://praat.org/manual/pitch_analysis_by_filtered_cross-correlation.html)，其余页面见22号[69]。filtered CC页的Availability仍写raw CC，与该页算法段不一致；本说明依据Purpose、Algorithm及Usage，不照抄该处疑似复制遗留。
- VoiceSauce：[主页](https://www.phonetics.ucla.edu/voicesauce/)、[F0说明](https://www.phonetics.ucla.edu/voicesauce/documentation/parameters.html#f0)、[设置](https://www.phonetics.ucla.edu/voicesauce/documentation/settings.html)、[官方v1.37包](https://www.phonetics.ucla.edu/voicesauce/current/VoiceSauce.zip)。已读相关官方文档和包装源码，未运行软件；不算读过所引算法论文全文。手册存在历史版本文字差异，版本判断优先明确更新说明和所查源码。


## 2026-09-14补充：能否在起声处缩窗

可以减少跨界混合，但不能无限缩窗。假设60ms前无稳定周期，目标是65ms：若只用60–65ms的5ms，100Hz声音尚不足一个10ms周期，不能作可靠的、无强先验的独立周期估计。可将窗口向右移以取得后续稳定周期，但60–90ms这类区间测得的是一段局部估计，不应伪装成65ms的瞬时真值。窗口长度、权重或偏移变化也会改变时间定位和估计偏差。本轮未找到可直接认定为标准、且对所有起声适用的“按有声边界裁窗修正公式”。

更直接的既有方法是逐周期测量：定位两个相邻的同相位脉冲，以间隔的倒数表示该周期频率。若脉冲在60、70ms，间隔10ms对应100Hz，标在65ms；这依然是一个周期的平均，不是无时间跨度的瞬时测量。声学波形的任意峰不等于声门闭合，需跨周期辨认同类结构；不规则发声可能没有唯一可信F0。EGG闭合事件可作辅助参照，但也需要识别可靠事件。

Praat的Sound & Pitch: To PointProcess (cc)依据已有Pitch有声区间，从区间中部向两侧逐周期匹配；会按相关强度拒绝或补充边界脉冲。这证实逐周期定位是现有实现，不证实其自动起声点就是真值；其依赖初始Pitch，因此不能作为完全独立验证。PointProcess: To PitchTier将相邻事件间隔倒数放在两事件中点。官方手册该页正文的“more than maximumInterval”与设置描述冲突；本轮源码核实实际条件为interval <= maximumPeriod。

项目建议（尚未执行）：保留现有长窗结果；仅在三个指定字的起声区增加逐周期人工复核。H1：长窗混入前段是主要原因，预计边界避让或逐周期结果中首段升高显著减弱；H2：周期本身在变长，预计逐周期仍显示下降。每例标最初5–10个可辨周期及歧义，固定同一音频与合理搜索范围，对照首段升高幅度、持续时间和参数敏感性；混合结果保留不确定，不以更平滑为更准确。只标记测量支撑区間和可信起点，不把真实微扰扣除。试行预算约30–60分钟，波形难辨则停止该例并记缺测，不扩展逐构式任务。三例不足推断总体准确率或黎里研究结果。

新增核查来源（并入22号[69]，不另增重复手册条目）：
- https://praat.org/manual/Sound___Pitch__To_PointProcess__cc_.html （Algorithm全文）
- https://praat.org/manual/Sound__To_PointProcess__periodic__cc____.html （说明先raw AC，再脉冲匹配）
- https://praat.org/manual/PointProcess__To_PitchTier___.html （Settings、Algorithm）
- https://praat.org/manual/Voice_2__Jitter.html （周期测量用途；通常用于持续元音，不能直接声称已验证起声可靠性）
- https://github.com/praat/praat/blob/master/fon/PitchTier_to_PointProcess.cpp （PointProcess_to_PitchTier函数，2026-09-14取得的master快照已缓存，非固定发布版本）
