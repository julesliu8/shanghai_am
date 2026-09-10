# Praat 与 Python 音频处理操作笔记

记录日期：2026-09-10。用于录音11的基频测量、五度标记试验和人工听校；软件操作与命令已经在本机核验。公式与归一化方法见 [10_五度标记与F0归一化方法笔记.md](10_五度标记与F0归一化方法笔记.md)。本笔记不把实测 F0、停顿或标点变成项目的 AP 金标准。

## 1. 文件位置与工具版本

- 原始录音：`data/raw/natural/recording_11_20260909/`，保留不改写。
- 已按 Audition 工程时间线重建的音轨1：`generated/recording_11_20260909/track_1_16k.wav`。这是完整轨道；不能只读 `轨道 1_004.wav` 然后把它的第0秒当成全录音第0秒。该原文件在全局时间线从约9.427秒开始。
- 正式句段索引：`data/metadata/recording_11_segments.csv`。句段编号及音频定位以此为准，试验表另存，不覆盖索引中已有的人工状态。
- 本轮可重建测量、图形与报告：`generated/recording_11_20260909/tone_pilot/`；官方手册缓存：其下 `references/`。
- 可供人工修改的试验标注应放 `data/annotations/draft/recording_11_20260909/` 下独立批次；不要把唯一的人工 TextGrid 修改只留在 `generated/`。

| 工具 | 本机位置／版本 | 说明 |
|---|---|---|
| 独立 Praat | `C:\tools\Praat.exe`；6.6.30，2026-06-30 | 命令行 `--version` 已核验；支持新 filtered autocorrelation |
| Python 环境 | `C:\tools\anaconda3\envs\qwen3-asr\python.exe` | 已存在，无须为了本笔记另建环境 |
| Parselmouth | 0.4.7 | Python 中直接调用嵌入的 Praat 算法 |
| Parselmouth 内置 Praat | 6.1.38 | **与独立 Praat 6.6.30 不同**；本机 `to_pitch_ac` 是 raw autocorrelation |

版本查看方式（PowerShell）：

```powershell
& 'C:\tools\Praat.exe' --utf8 --version
conda run --no-capture-output -n qwen3-asr python -c "import parselmouth; print(parselmouth.__version__, parselmouth.PRAAT_VERSION)"
```

如果 `conda` 没有进入 PATH，可直接使用上表中的 Python 绝对路径。`--utf8` 很有用：Windows Praat 默认向管道输出无 BOM 的 UTF-16LE，若接收程序按 UTF-8 解码，会看到空字符或乱码。

## 2. 先分清基频提取方法

Praat 官方手册明确区分以下方法。[1–3]

| 方法 | 适用情形 | 本次可用实现 |
|---|---|---|
| Raw autocorrelation，原始自相关 | 传统语调研究方法，也是原始周期性分析方法；可作为可复现基线 | Parselmouth `Sound.to_pitch_ac`；独立 Praat `To Pitch (ac)` 或新名称 |
| Filtered autocorrelation，滤波自相关 | 自2023年起官方优先推荐用于语调与声带振动频率；先作低通滤波，通常减少倍频／半频错误 | 独立 Praat 6.6.30；本机 Parselmouth 6.1.38 不支持 |
| Raw cross-correlation，原始互相关 | 官方优先用于嗓音分析，例如病理嗓音研究 | 不作为本轮声调试验的默认方法 |

不能因为 Python 包和桌面程序都叫 Praat，就认定它们算法、默认参数或结果相同。本机实测在 Parselmouth 调用 `To Pitch (filtered autocorrelation)` 会返回命令不可用；不要静默退回 raw 后仍写 filtered。

Filtered 法的 `pitch top` 同时影响滤波与候选范围，**不等于 raw 法的 `pitch ceiling`**。官方默认 filtered 是 floor=50 Hz、top=800 Hz，raw 是 floor=75 Hz、ceiling=600 Hz。若 raw 使用60–400 Hz，不能把 filtered 的 top 也直接填400 Hz；滤波会提前压低较高频率的成分。

低通滤波也可能改变清浊判定，尤其是弱声、带声塞音闭塞段；没有一种提取方法能免除波形、语谱图与听感复核。若切换算法，需重新提取整轨、重新拟合归一化范围并记录版本，不能混用两套阈值。

## 3. 参数怎么理解与记录

下表列 raw autocorrelation 的常用参数；“本机示例”是验证代码所用数值，不是已经人工确认适合整段录音的最优值。[1]

| 参数 | 含义 | 官方 raw 默认值 | 本机示例 |
|---|---|---:|---:|
| `time_step` | 相邻测量帧的时间间隔，秒 | 0／自动 | 0.01 s |
| `pitch_floor` | 最低候选基频，也决定窗长 | 75 Hz | 60 Hz |
| `pitch_ceiling` | 最高候选基频 | 600 Hz | 400 Hz |
| `max_number_of_candidates` | 每帧候选数量设置，含清音候选 | 15 | 15 |
| `very_accurate` | 使用更长的高斯窗 | False | False |
| `silence_threshold` | 相对整段最大振幅的静音阈值 | 0.03 | 0.03 |
| `voicing_threshold` | 有声与无声判断的周期性阈值 | 0.45 | 0.45 |
| `octave_cost` | 候选选择中偏好较高频率的代价项 | 0.01 | 0.01 |
| `octave_jump_cost` | 抑制突跳的路径代价项 | 0.35 | 0.35 |
| `voiced_unvoiced_cost` | 抑制过多清浊切换的路径代价项 | 0.14 | 0.14 |

普通 raw 分析窗的物理长度约为 `3 / pitch_floor`：floor=60 Hz 时约50 ms。时间步长10 ms不表示每帧只看10 ms声音，也不意味着测量或字边界有10 ms精度。自动步长约为 `0.75 / pitch_floor`。开启 `very_accurate` 后使用物理长度 `6 / pitch_floor` 的高斯窗，会增加对边缘和极短音节的影响。

参数选择应先看试段的低音、高音、弱声、句末和重叠处。下限太高可能丢掉低声／误识倍频；下限过低可能增加半频、嘎裂声和噪声候选。过宽上限也会引入高次谐波。不要只依据性别给所有说话人套同一固定范围。

整轨分块时还要注意：静音阈值相对的是传入 Sound 的最大振幅，而非整份工程的统一参照。重叠分块提取能控制内存，但块长、重叠量、边缘裁切和幅度门限都应记入运行报告，否则换块长可能改变清浊判断。任何短块都应先保留足够上下文，再裁去边缘帧；不要把跨静音和跨说话人的断裂强行连成一条曲线。

## 4. 用桌面 Praat 人工检查

以下菜单来自当前官方手册；旧版有些命令仍叫 `To Pitch (ac)`。[1–4]

1. 在 Objects 窗口选择 **Open → Read from file…**，打开音轨1或一个试验片段。长文件也可使用 **Open long sound file…**；若需要创建整轨 Pitch 对象，宜用脚本处理或将片段读成普通 Sound。
2. 选中 Sound，点击 **View & Edit**。在编辑器的 **Pitch** 菜单打开 **Show pitch**，在 **Pitch settings…** 检查算法、范围和步长。分析范围与画图显示范围是不同设置，不要只改坐标轴便以为重新提取了基频。
3. 放大到短语或音节，听播放区，同时看波形、语谱图、基频线。突然翻倍／减半、静音中的长平线、重叠人声、句末不规则振动都应加入复核记录。
4. 若需要保存明确的 Pitch 对象，回到 Objects，选中 Sound，通过 **Analyse periodicity** 菜单执行所需 `To Pitch…` 命令；在对象列表中选中 Pitch 再进入其编辑／查询窗口。仅在 Sound 编辑器显示了曲线，不代表已经保存了可追溯的提取结果。
5. 改候选或重提取后保存独立版本，并注明改动的时间区间、原因与操作者。不把“图上看起来连续”当成正确性的充分条件。

音轨与说话人是两件事：单个麦克风轨道可能混入另一人的声音。音轨1的整体分位数只能先称“轨道参考范围”；要解释成某一说话人的调域，仍需确认目标说话人的区间。

## 5. 用 Parselmouth 提取、查询并处理缺测

下面 API 已在本机 Parselmouth 0.4.7／Praat 6.1.38 用180 Hz合成纯音验证；实际长录音宜使用本轮批处理脚本，避免一次创建过多中间数组。[5]

```python
import numpy as np
import parselmouth
from parselmouth.praat import call

sound = parselmouth.Sound("generated/recording_11_20260909/track_1_16k.wav")
pitch = sound.to_pitch_ac(
    time_step=0.01, pitch_floor=60, pitch_ceiling=400,
    max_number_of_candidates=15, very_accurate=False,
    silence_threshold=0.03, voicing_threshold=0.45,
    octave_cost=0.01, octave_jump_cost=0.35,
    voiced_unvoiced_cost=0.14,
)
time_s = pitch.xs()  # 真正的帧中心；不能用 np.arange(n) * 0.01 代替
f0_hz = pitch.selected_array["frequency"].copy()
strength = pitch.selected_array["strength"].copy()
valid = np.isfinite(f0_hz) & (f0_hz > 0)
f0_hz[~valid] = np.nan

if valid.any():
    q05, q50, q95 = np.quantile(f0_hz[valid], [0.05, 0.50, 0.95])
    praat_median = call(pitch, "Get quantile", 0, 0, 0.50, "Hertz")
    log_f0 = np.full(f0_hz.shape, np.nan)
    log_f0[valid] = np.log2(f0_hz[valid])
else:
    raise ValueError("没有可用有声帧，不能计算调域或五度值")
```

这里 NumPy 的分位数是示例；正式分析还应先套用强度、周期性、轨道串音及异常值筛选。Praat 的 `Get quantile` 直接查询其 Pitch 对象中的有声候选，没有自动继承你对 NumPy 数组另做的质量掩码；不能把二者当成同一套过滤后的统计。不同软件的分位数插值约定也可能造成小差别，所以应固定实际使用的实现与版本。

Praat 查询中的 `0, 0` 表示整个 Pitch 对象时间范围，`0.50` 表示中位数。本机180 Hz合成纯音的 Parselmouth中位数约180.00035 Hz；这只证明命令调用和量纲正常，不证明真实上海话的提取准确率。

Parselmouth 数组通常把未选中有声候选的帧表示为0 Hz；Praat其他查询可能返回 undefined／NaN。**0不是“极低声调”**：0、负值、NaN及无穷值不能进入对数或归一化，应保留为空值／缺测状态。`strength` 是候选周期性指标，不是校准过的正确概率。不要把它写成“95%准确率”。

如果只想看图，可以画有声点或用 NaN 断线；跨长清音段插值会制造实际不存在的声调运动。字内覆盖不足应输出 `insufficient_voicing` 一类明确状态，而不是强制填成111或最低调。

## 6. 用独立 Praat 批处理

下列两个提取命令和分位数查询均已在本机 Praat 6.6.30运行成功；参数顺序不可相互替换。

```praat
# 例：生成180 Hz测试音。正式处理改为 Read from file: "完整路径.wav"
s = Create Sound from formula: "test180", 1, 0, 2, 16000, "0.2 * sin(2 * pi * 180 * x)"

# 新版 filtered：time step, floor, top, candidates, very accurate,
# attenuation at top, silence, voicing, octave, octave jump, voiced/unvoiced
p = To Pitch (filtered autocorrelation): 0.01, 60, 800, 15, "no", 0.03, 0.09, 0.50, 0.055, 0.35, 0.14
q = Get quantile: 0, 0, 0.50, "Hertz"
writeInfoLine: "filtered_median_Hz=", q

selectObject: s
# 兼容旧版本的 raw 命令：ceiling 位于最后
p2 = To Pitch (ac): 0.01, 60, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 400
q2 = Get quantile: 0, 0, 0.50, "Hertz"
appendInfoLine: "raw_median_Hz=", q2
```

本机上述 filtered 和 raw 的中位数分别为180.00002347 Hz和180.00002347 Hz。纯音两法接近是正常现象，不能由此推断自然录音中的两法也等价。

将脚本保存为 `.praat` 后在 PowerShell 中运行（路径为待替换示例）：

```powershell
& 'C:\tools\Praat.exe' --utf8 --no-pref-files --run 'D:\path\analysis.praat'
```

`--run` 在无 GUI 模式执行，脚本里不能调用 `View & Edit`；`--open` 用于在GUI中打开文件或脚本；`--send` 用于让GUI执行脚本。批处理选 `--run`，不要依赖省略开关时不稳定的自动判断。`--no-pref-files` 避免批处理读写个人GUI偏好。Python 调外部 Praat 时优先用 `subprocess.run([exe, "--utf8", "--no-pref-files", "--run", script], check=True)` 传参数列表，路径带空格也无需自己拼引号。[6]

### 本轮真实音轨的 filtered 对照

另存的 [track1_filtered_ac.praat](../scripts/track1_filtered_ac.praat) 已实际处理完整音轨1，作为 raw 主分析的敏感性检查。该脚本接受输入WAV和输出Pitch两个参数，固定 `dt=0.01 s, floor=50 Hz, top=600 Hz`，其余采用filtered默认参数。这里 top=600是本次对照设置，区别于上方演示的top=800。重跑命令应在 `AM project/` 目录执行：

```powershell
$pitchScript = (Resolve-Path -LiteralPath 'scripts/track1_filtered_ac.praat').Path
$sourceWav = (Resolve-Path -LiteralPath 'generated/recording_11_20260909/track_1_16k.wav').Path
$outputPitch = Join-Path (Get-Location).Path 'generated/recording_11_20260909/tone_pilot/track1_filtered_ac.Pitch'
& 'C:\tools\Praat.exe' --utf8 --no-pref-files --run $pitchScript $sourceWav $outputPitch
```

`tone_pilot/track1_filtered_ac.Pitch` 是保存的Pitch对象；`filtered_comparison.tsv` 逐帧保存绝对时间、Hz、周期性和清浊状态，无声Hz为空；`filtered_comparison_summary.json` 保存参数、统计与五个试段的对照。二进制Pitch已由本机Parselmouth成功重新读取。

| 指标 | 本轮filtered结果 |
|---|---:|
| 完整时域 | 0–3609.3794375 s |
| 帧数 | 360,932 |
| 有声帧数／比例 | 137,500／38.10% |
| 未经主流程质量后筛的Q05／中位数／Q95 | 87.0198／108.6677／156.7011 Hz |

用最近帧心（差不超过5.1 ms）匹配两法，在“raw主筛可靠且filtered有声”的131,499帧上，F0差的绝对值中位数为2.98 cents，95%分位数为21.26 cents；绝对差超过100 cents的比例约0.0479%。计算为 `1200 * log2(F0_filtered / F0_raw)`；100 cents等于一个半音。试段0008、0078、0240、0351、0579在同样比较条件下均未出现超过100 cents的差值。

两法的清浊覆盖并不相同，上述统计也排除了主流程认为不可靠的raw帧，因此只能说明保留帧的结果大体一致。它不是逐字人工准确率；整体分位数也不能直接替换主流程质量筛选后的归一化阈值。帧中心差和边缘误差仍需听校。

## 7. TextGrid 与汉字时间对齐

TextGrid 保存时间标注，本身不保存声音。官方建议从 Sound／LongSound 创建 TextGrid，以自动获得相同起止时间，然后同时选中声音和 TextGrid，进入 **View & Edit**。[4,7]

本项目正式核心轨道为 `syllable word AP misc`。本轮可以另加带 `draft` 或 `measured` 名称的试验轨道存放汉字／五度候选，并在 `misc` 写明“机器对齐，待听校”；不能直接填入正式 AP 标签，也不能假称每个汉字都已被证实对应一个音节。

在GUI中创建时，**Annotate → To TextGrid…** 的 Tier names 填 `syllable word AP misc`，Point tiers 留空则全部是区间轨道。以音节为单位的标签应该标在区间中；不要把音节时长误作一个瞬间点。

以下 Python 调用已验证：

```python
from parselmouth.praat import call
tg = call(sound, "To TextGrid", "syllable word AP misc", "")
call(tg, "Insert boundary", 1, 0.4)  # 第1轨，0.4秒；须在声音时间域内部
call(tg, "Set interval text", 1, 1, "示例")
tg.save("example.TextGrid", parselmouth.Data.FileFormat.TEXT)
```

这只是操作示例，不应把示例标签写到实际录音中。保存为文本 TextGrid 可供后续审查与版本比较；重新打开时要把对应声音一起选中。中文内容的 Python 脚本应以 UTF-8 保存；不要把 PowerShell管道编码错误造成的乱码当成 TextGrid不支持中文。

**人工修改请另存。** 当前 `scripts/analyze_track1_tones.py export` 从对齐JSON重新生成TextGrid，会覆盖原导出的同名文件，不会读回在Praat中修改的边界或标签，也不会据此更新逐字CSV和五度结果。人工听校后的TextGrid请另存到 `data/annotations/draft/recording_11_20260909/tone_pilot/manual_review/`，保留原句段编号，并记录修改日期和操作者。该目录用于保存人工版本；当前尚无将人工修改导回JSON、重算F0和五度结果的程序。

五度标记与汉字的匹配流程：

1. 给选定句段的用户修订文字做强制对齐，获得字／词时间区间；保留原始输出和未对齐、零时长、重叠等失败状态。
2. 对每个汉字候选区间 `[a, b)`，选取全轨 F0 中满足 `a <= time_s < b` 的有效帧。标点不占声学区间；多字词如果没有字级时间，不能平均拆分后假称已对齐。
3. 按统一的全轨参考范围转换为五度连续值与离散级别。可在区间内部作起、中、末位置的局部稳健统计，输出三点候选如352；若有效帧不足或有长断裂，保留空值和原因。
4. 将字、字区间、局部F0、五度候选、覆盖率和质量状态一起写入试验表，在 TextGrid中按相同边界显示。
5. 人工按“波形／语谱图边界 → 听字音 → 检查F0候选 → 确认读数”的顺序复核。强制对齐能给出合理外观的错误结果，尤其在上海话、语气词、重叠和稿音不符处。

句段切片前后有50 ms上下文时，不能直接使用CSV的原句开始时间作切片零点。正确公式是：

`全轨时间 = 切片物理开始时间 + 切片内时间`。

取到全轨F0后映射回切片则相减。若加载的声音本身保留非零 `xmin`，还要核对这一偏移；`pitch.xs()` 只在其自身 Sound 时间坐标内正确。全局时间、切片时间与 TextGrid 时间轴三者必须明确记录。

## 8. 实际复核清单与当前限制

- 对比音轨1与另一轨的能量和试听，确认被测的是目标声源，而不是串音或整段混音。
- 检查低音、高音、嘎裂、弱声和两人重叠；保留缺测与算法分歧，不为了图形漂亮补齐。
- 对每字记录有声覆盖率、有效帧数、边缘缺失和对齐状态。三点数字是实测调形摘要，不能直接替代本调调类、音系调值或变调域判断。
- 同一批次固定提取器、软件版本、采样率、分块策略、质量筛选、分位数实现和归一化阈值；改参数后生成新批次或明确版本。
- 当前已验证工具与命令可用；真实音轨的测量结果、范围与试段质量以本轮运行报告为准。说话人确认和逐字听校仍是后续工作。

## 9. 来源与缓存

以下为本轮实际读取的官方文档，访问日期2026-09-10；网页缓存位于 `generated/recording_11_20260909/tone_pilot/references/`。

1. Praat manual: [Pitch analysis by raw autocorrelation](https://www.fon.hum.uva.nl/praat/manual/pitch_analysis_by_raw_autocorrelation.html)。缓存 `pitch_analysis_by_raw_autocorrelation.html`。
2. Praat manual: [Pitch analysis by filtered autocorrelation](https://www.fon.hum.uva.nl/praat/manual/pitch_analysis_by_filtered_autocorrelation.html)。缓存 `pitch_analysis_by_filtered_autocorrelation.html`。
3. Praat manual: [How to choose a pitch analysis method](https://www.fon.hum.uva.nl/praat/manual/how_to_choose_a_pitch_analysis_method.html)。缓存 `how_to_choose_a_pitch_analysis_method.html`。
4. Praat manual: [Intro 7. Annotation](https://www.fon.hum.uva.nl/praat/manual/Intro_7__Annotation.html)。缓存 `praat_intro_annotation.html`。
5. Parselmouth: [API reference](https://parselmouth.readthedocs.io/en/stable/api_reference.html)，尤其 `Sound.to_pitch_ac`、`PRAAT_VERSION`、`parselmouth.praat.call`。缓存 `parselmouth_api.html`。
6. Praat manual: [Scripting 6.9. Calling from the command line](https://www.fon.hum.uva.nl/praat/manual/Scripting_6_9__Calling_from_the_command_line.html)。缓存 `praat_script_cli.html`。
7. Praat manual: [TextGrid](https://www.fon.hum.uva.nl/praat/manual/TextGrid.html)、[TextGrid file formats](https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html)、[TextGridEditor](https://www.fon.hum.uva.nl/praat/manual/TextGridEditor.html)。缓存 `praat_textgrid.html`、`praat_textgrid_file_formats.html`、`TextGridEditor.html`。

独立 `Pitch: Get quantile` 网页地址曾返回404，按项目约定通过 `http://127.0.0.1:7897` 重试仍为404；本笔记对该命令的说明依据本机实际调用核验，不引用不存在的网页。旧式分页面 Parselmouth URL也返回404，已改用实际可访问的官方统一 API 页面。

本轮5段试验音频实际采用前后各250 ms上下文，保存在 `tone_pilot/*_alignment.json` 的 `clip_start_global_s`，与早前593段索引的50 ms切片不同。试验TextGrid含四个核心层和两个附加层：`tone5_auto`存自动三点五度，`alignment_issues`点层保留零时长字；`syllable`也只是字区间候选，`word`和`AP`未预填。完整重跑入口为 `scripts/analyze_track1_tones.py` 的 extract、pilots、filtered、export、report 五个阶段，结果见 12号试验报告（本地材料：`12_音轨1五度标记试验报告.md`）。
