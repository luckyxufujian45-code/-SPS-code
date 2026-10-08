# 单颗粒 ICP-MS 事件识别

## Windows 桌面软件

`dist/ICPMS_Supervised.exe` 和 `dist/ICPMS_Unsupervised.exe` 为独立运行的 64 位 Windows 软件，无需 Python。选择纯离子背景、样品、结果目录即可运行，支持多样品处理。

用户已确认 `20ppt-*` 是 100 nm 银纳米颗粒标准。桌面软件支持选择标准文件并填写已知粒径，按 `d = d_ref × (C / median(C_ref))^(1/3)` 输出等效球径。两种模型采用相同标准选择规则：局部稳健 z > 8、相邻点合并、至少 5 个事件，取净计数中位数。此校准依赖相同颗粒组成、密度、仪器灵敏度和近似线性的质量响应。不同标准文件的响应不同，选择与样品采集条件对应的标准。

桌面输出包括全程信号与红色事件标记图、局部放大图、粒径分布 PNG/CSV、颗粒候选数量和粒径统计 CSV/JSON、逐事件和逐点 CSV。没有选择颗粒标准时，输出净积分信号分布，不生成粒径。

详见 `软件使用说明.txt`。开发者可安装 `requirements-build.txt` 后用 `build_apps.ps1` 重新打包。测试生成的合成校准参数仅用于流程测试，不是实验校准结果。

仓库包含源码、依赖和使用说明。原始实验数据、结果目录、虚拟环境和 exe/ZIP 不纳入源码提交。批处理脚本需要自行将实验 CSV 放入本地数据目录；`--demo` 和自动化测试不需要实验数据。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe -m unittest test_spicpms.py test_desktop_app.py
& .\build_apps.ps1
```

两个可运行模型：随机森林监督分类、Isolation Forest 无监督背景异常检测。输入原始时间序列，不使用曲线图片。无真实标注时，随机森林用背景中注入的模拟脉冲训练；这是模拟标签监督学习，不能作为真实识别准确率的证据。无监督异常也不能直接证明为纳米颗粒。

## 运行

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe spicpms.py --demo --out results_demo
.\.venv\Scripts\python.exe spicpms.py --background "纯离子背景.csv" --sample "样品.csv" --unit counts --out results_real
.\.venv\Scripts\python.exe run_batch.py
```

兼容本项目仪器 CSV（`sep=;`，Number;Time 107Ag;107Ag）和两列 time,signal CSV。时间单位必须为秒，强度不自动转换。用户已确认本项目为 **3 ms、cps**，标准曲线文件夹下所有 CSV 均为纯离子溶液。单文件运行本项目数据请用 `--unit cps`。

批处理读取所有非标准曲线 CSV，按信号中位数的对数距离选最近的离子标准溶液，复用对应模型并输出 `results_real/summary.csv` 和逐样品结果。选择只是探索性规则：不同浓度、基质和背景噪声可能不匹配，尤其标准曲线浓度跨度较大，须用匹配的纯离子对照与独立标注验证。不得把批处理事件数当作已确认颗粒数。

真实标签可另存为三列 `time,signal,label` CSV，label=0 是背景、1 是颗粒覆盖的采样点，使用 `--labels 标注.csv`。标注文件必须属于独立训练采集，样品文件应来自独立测试采集。仅有背景时，背景前 2/3 用于训练，后 1/3 用于阈值校准；建议采集更多独立空白批次验证误检率。

## 输出与解释

- `supervised_events.csv`、`unsupervised_events.csv`：事件起止、峰值、背景扣除后的信号和、分数。
- `point_predictions.csv`：原始数据、局部背景、两个模型的逐点结果。
- `detections.png`：最强信号附近的一段原始曲线与检测点。
- `report.json`：时间间隔、阈值、背景校准逐点误检率、事件数；demo 另有模拟逐点指标。

`net_signal_sum` 是扣除局部背景后的正信号求和。`net_counts_estimate` 对 cps 输入计算净信号求和 × 时间间隔，对 counts 输入直接保留净计数。这里假定连续采集、每个 cps 值对应完整采样区间的平均速率；如有死时间、跳点或导出重采样，需另行校正。这些值不是已标定的颗粒质量或粒径。

`--alpha` 默认 0.001，是背景校准的逐点误检目标，不是颗粒事件误检率。阈值采用严格大于处理离散分数和并列值，实际误检率可能更低。随机森林分数不是经过概率校准的颗粒可信度，Isolation Forest 分数是异常分数。

局部中位数与 MAD 归一化用于降低缓慢漂移影响。两个模型使用相邻五点特征；正向信号异常才允许被标记。相邻阳性点合并为事件，不自动跨越阴性点或拆分重叠事件。背景与样品噪声分布差异、快速背景变化、仪器尖峰和重叠颗粒都可能导致错误。3 ms 采样下可能只有一个点代表一个事件，无法重建未采到的脉冲形状。

模拟脉冲宽度为 1–3 个采样点、振幅按对数均匀分布取背景标准尺度的 2–300 倍，是探索性假设，用于覆盖弱峰与强峰。正式验证需独立真实标注，按批次评估事件召回率、每分钟误检数、合并/拆分错误和净信号误差；demo 的准确率不能替代这些指标。
