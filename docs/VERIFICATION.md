# 本地验证记录

2026-09-16，使用 `D:/mambaforge/envs/cutile/python.exe`。

环境：Python 3.11.14，XGBoost 3.2.0，NumPy 2.3.5，pandas 2.3.3，scikit-learn 1.6.1，setuptools 80.9.0。

| 检查 | 结果 |
|---|---|
| 从 pyproject 构建 wheel | PASS，`dist/fprc-0.1.0-py3-none-any.whl` |
| wheel 安装到独立 `.install-test`，无原仓库模块路径 | PASS |
| 从安装产物运行全部 11 项行为测试 | PASS |
| CPU 合成案例 | 8,400 行训练，2,160 行未来预测，全部有限 |
| CUDA/4090 合成案例 | 8,400 行训练，2,160 行未来预测，全部有限 |
| 合成案例历史 OOF | 6,384 行有效；2,016 行 warm-up 保留 NaN |
| Pure Gain `B=Fmean+2*local_mean` 与 `S=B+closure` | PASS |
| CPU/CUDA 各自保存再加载 | 预测一致 |
| 早停边界二次 purge 审计 | 31 个使用早停的 CUDA 拟合全部通过 |
| 原代码 SHA256 与建包前对比 | 六个来源文件未变化 |
| CellStats 副本与原文件 SHA256 | 逐字相同 |

11 项测试覆盖：gain 代数与局部模型复用、closure 的 OOF 标签路由、未来标签扰动、严格模型时间边界、打乱输入顺序、字段置换、外部 PIT 田/包内田混用、序列化、输入错误、horizon/MultiIndex、旧三田因子名称与 Fold 同值分箱。

CUDA 预测 NumPy 输入时 XGBoost 会提示转 DMatrix 的设备不匹配警告；训练仍运行在 CUDA，示例预测完整且公式检查通过。CPU/GPU 算法产生的预测不要求逐位相等，保存加载检查分别在各自设备上进行。

验证的是安装、实现和时间协议；没有在新包上宣称复现旧账本收益或进行新的金融绩效试验。
