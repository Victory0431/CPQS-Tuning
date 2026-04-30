# GSM8K Full 回退排查

最后更新：2026-04-30 14:55 CST

## 目标

本次排查不继续推进 `Random-K / CNN Top-K / CNN Bottom-K`，而是先回答一个更关键的问题：

- 为什么 `GSM8K Full seed 1` 会从 `Base = 0.9310` 回退到 `0.8271`？

## 当前正式结果

| Method | GSM8K |
| --- | ---: |
| Base | 0.9310 |
| Full seed 1 | 0.8271 |

对应文件：

- [Base run_scores.csv](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/gsm8k_base_full/run_scores.csv)
- [Full run_scores.csv](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/gsm8k_full_seed1/run_scores.csv)

## 核心统计

- `Base` 正确数：`1228 / 1319`
- `Full` 正确数：`1091 / 1319`
- `Full` 错误数：`228`
- 其中：
  - `Base 正确, Full 错误`：`163`

这说明：

- 回退不是少量边缘样本波动
- 而是有一批原本 `Base` 能做对的题，在 `Full` 后变错了

## 首轮 ablation 结论

这轮先人工抽看了 `Full` 错误样本，并统计了最容易混淆的几类“假回退”：

- 没有最终答案标记：`0`
- 空预测：`0`
- 疑似输出截断：`0`
- 混入 `<think>`：`0`
- 依赖 `<answer>` 标签抽取：`0`

结论：

- 当前没有证据表明这是评测脚本抽取失败
- 也没有证据表明这是生成长度截断
- 也不像是模板跑偏到 thinking / 非标准输出
- 当前最像的情况是：
  - 模型确实按模板答了
  - 但数学推理质量下降了

## 错误类型观察

从 `Base 正确, Full 错误` 的前几条样例看，错误主要是：

### 1. 百分比 / 比例理解错误

例子：

- 房屋翻修题
- 下载进度题
- 月度下载量递减题

典型表现：

- 把 “increase by 150%” 当成只增加 `15%`
- 把 “reduced by 30%” 当成直接减 `30`
- 把“重启后从头下载”错误压缩成一次局部补算

### 2. 多步约束链条被简化或错连

例子：

- Melanie 吸尘器题
- John 开车题
- Doubtfire kitten 题
- 四个学校人数题

典型表现：

- 把题中的前后关系错误拼接
- 把“每队一个教练”简化成“每校一个教练”
- 把“剩余的一半”改成直接加减现成数字

### 3. 单位 / 总量概念漂移

例子：

- Claire 鸡蛋 dozen 题
- Lloyd 鸡蛋 farm 题
- John 跑步 speed 题

典型表现：

- 把“每周总量”误算成“某一天”或反过来
- 漏掉 dozen 的换算
- 把距离、时间、速度关系直接错配

### 4. 分数与混合液体问题容易退化

例子：

- Orange / pineapple drink 混合题

典型表现：

- 前面步骤看起来很多
- 但最后把“水量”和“总液量”混成一个量

## 样例对照

下面这些题目都属于：

- `Base` 做对
- `Full` 做错

### 1. 房屋翻修利润题

- gold：`70000`
- base：`70000`
- full：`19500`

观察：

- `Full` 把 “增加 150%” 错算成了 `0.15`

### 2. Carla 下载题

- gold：`160`
- base：`160`
- full：`60`

观察：

- `Full` 忽略了“从头重新下载”的总时长结构

### 3. 新程序下载量题

- gold：`366`
- base：`366`
- full：`390`

观察：

- `Full` 把“减少 30%”错成直接减 `30`

### 4. Lloyd 卖鸡蛋题

- gold：`294`
- base：`294`
- full：`3528`

观察：

- `Full` 完全漏掉了“per dozen”的换算

## 目前最可信的判断

在当前这版协议下，`Full` 的回退更像是：

- 模型被 `GSM8K train` 的监督格式牵引后
- 输出风格更像“短链条、直接套公式”
- 但多步数学约束和单位处理能力反而变差

也就是说：

- 这不像“评测坏了”
- 更像“训练后数学能力真的退化了”

## 这意味着什么

这对后续实验顺序的影响很大：

- 如果 `Full` 已经显著低于 `Base`
- 那么直接继续做 `Random-K / CNN Top-K / CNN Bottom-K`
- 很可能只是在一个已经退化的训练协议上继续比较子集

这样就很难把结论归因到 `CPQS` 方法本身。

## 建议的下一步

优先级建议如下：

1. 先不要继续 `CNN` 实验
2. 先做训练协议排查 ablation，例如：
   - 输出格式是否必须要求完整 CoT
   - 是否要缩短或调整 instruction 模板
   - 是否要改模型到更贴论文的 `Qwen2.5-7B-Instruct`
   - 是否要重新考虑 `GSM8K` 这条训练任务本身
3. 只有在 `Full >= Base` 或至少接近 `Base` 之后，再进入：
   - `Random-K`
   - `CNN Top-K`
   - `CNN Bottom-K`

## 相关文件

- `Base` 全量预测：
  - [gsm8k_predictions.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/gsm8k_base_full/gsm8k_predictions.json)
- `Full` 全量预测：
  - [gsm8k_predictions.json](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/eval/gsm8k_full_seed1/gsm8k_predictions.json)
- `Full` 正式日志：
  - [gsm8k_full_eval_seed1.log](/home/qjh/llm_learning/CPQS_lab/CPQS-Tuning/repro_outputs/logs/gsm8k_full_eval_seed1.log)
