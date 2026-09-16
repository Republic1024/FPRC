# FPRC — Field-Preserving Residual Closure

**Preserve the fields. Learn their residuals. Close the aggregate.**

Paper Link: https://arxiv.org/abs/2608.07349

一个面向横截面因子研究的结构化学习框架：以可审计的二维因子田为先验，让局部 XGBoost 学习各自的残差，再由共享 XGBoost 对聚合后的新鲜残差完成闭合。**Pure Gain** 保留同一结构，仅将局部修正的总增益从 1 改为 2。

FPRC 不是一组只能在研究脚本中运行的公式。本项目提供独立可安装的 Python 包、时间前推交叉拟合、训练边界审计、分量预测、模型持久化、CPU/CUDA 示例和冻结实验结果快照。

| 项目 | 内容 |
|---|---|
| 方法 | FPRC / FPRC–Pure Gain；任意数量的田，始终等权聚合 |
| 学习器 | XGBoost；CPU 默认，CUDA 可选 |
| 接口 | `fit` / `predict` / `predict_components` / `save` / `load` |
| 防泄漏设计 | 标签成熟时间、前推 OOF、历史验证早停、fresh-residual closure |
| 验证状态 | 11 项自动测试通过；CPU/CUDA 示例及独立 wheel 安装验证通过 |
| 发布状态 | 研究软件 v0.1.0；可本地 pip 安装，尚未发布 PyPI；[MIT License](LICENSE) |

[方法与公式](#方法与完整公式) · [实验成绩](#冻结实验成绩) · [快速开始](#快速开始) · [时间安全](#时间安全与审计) · [复现与贡献](#复现与贡献)

## 为什么是 Field-Preserving

直接把所有因子输入一个模型，会让先验结构与残差学习混在一起。FPRC 将这两件事显式分开：

- **可解释先验**：每块田对应明确的 AB 坐标和历史条件收益统计。
- **明确的残差归属**：第 m 个专家学习 `r-F_m`，不偷换成 `r-mean(F)`。
- **精确表征旁路**：原始田的均值保留在 backbone，不要求有限模型重新学习它。
- **聚合后闭合**：共享模型针对当前 backbone 的样本外误差重新出题，而不是复用过期残差。
- **可控的机制改动**：Pure Gain 只改变修正增益，不增加局部模型，也不改变本田 anchor。

这里的 *Closure* 指最终共享残差学习步骤，不是声称有限模型能够零误差预测收益。结构带来的归纳偏置是否有用，需要样本外交易账本检验。

## 方法与完整公式

### 1. 从两个因子构造一块田

设 `r_{s,t}` 是股票 s 在信号日期 t 对应的未来收益标签；第 m 块田的父因子为 `A_m,B_m`。默认使用每日截面排名，将两个坐标分别分成 K=10 档：

$$
a_{m,s,t}=\operatorname{bin}_K(A_{m,s,t}),\qquad
b_{m,s,t}=\operatorname{bin}_K(B_{m,s,t}),
$$

$$
c_{m,s,t}=K a_{m,s,t}+b_{m,s,t},
\qquad a,b\in\{0,\ldots,K-1\}.
$$

这就是 **Fold**：保留二维坐标的组合身份，得到 K² 个状态，而非把两个因子相加后压成一维。实现使用日内 rank midpoint `(rank-0.5)/n_valid` 分箱。

每个格点只累计在 t 之前已经成熟的历史标签。设时间衰减后的格点有效计数、均值和方差为 `n_c, μ_c, v_c`，全局历史均值为 `μ_0`：

$$
\mu_c^{EB}
=\frac{n_c}{n_c+n_0}\mu_c
+\frac{n_0}{n_c+n_0}\mu_0,
$$

$$
T_c=\frac{\mu_c-\mu_0}
{\sqrt{\max(v_c,10^{-12})/\max(n_c,1)}},
\qquad
\operatorname{conf}_c=
\mathbf1\{n_c\ge n_{\min}\}
\frac{1}{1+\exp(2-|T_c|)},
$$

$$
\boxed{F_m(s,t)=\mu_{c_{m,s,t}}^{EB}\operatorname{conf}_{c_{m,s,t}}.}
$$

默认 `n_0=500`、`n_min=100`、半衰期 252 个信号日期。该 confidence 是原研究实现的启发式置信权重，不是校准后的显著性概率。空历史的可用格点为 0，缺失 AB 为 NaN。底层 CellStats 从研究代码原样复制，见[来源与哈希](docs/PROVENANCE.md)。

### 2. 明确局部输入与共享输入

最简配置只有各田 AB。原三田实验额外使用共享 P 与 Q：

$$
X_{\mathrm{parents}}=\bigcup_{m=1}^{M}\{A_m,B_m\},
\qquad
X_m=(\operatorname{Rank}_t A_m,\operatorname{Rank}_t B_m,
P_\star^\perp,\mathbf Q^\perp).
$$

`Rank_t` 表示每日截面 percentile rank。共享 P 用全部父坐标做截面线性净化：

$$
P_\star^\perp=
\operatorname{Rank}_t\!\left[
\operatorname{Rank}_t P-
\operatorname{OLS}_t(\operatorname{Rank}_t P\mid
\operatorname{Rank}_t X_{\mathrm{parents}})
\right].
$$

Q 的净化条件包含所有父坐标，以及各田单独净化出的局部 P：

$$
P_{m,\mathrm{local}}^\perp
=\operatorname{Rank}_t[P-\operatorname{OLS}_t(P\mid A_m,B_m)],
$$

$$
Q_\ell^\perp=
\operatorname{Rank}_t\!\left[
\operatorname{Rank}_t Q_\ell-
\operatorname{OLS}_t\!\left(
\operatorname{Rank}_t Q_\ell
\mid \operatorname{Rank}_t X_{\mathrm{parents}},
\{P_{m,\mathrm{local}}^\perp\}_{m=1}^{M}
\right)\right].
$$

OLS 包含截距；这些步骤不使用收益标签。没有 P 时，Q 只对父坐标净化。这里是线性去相关，不宣称统计独立。默认共享闭合器**只输入所有去重后的 AB parents 的截面排名**，不会自动重读 P/Q 或所有候选因子。

### 3. FPRC：本田修正 → 等权聚合 → 共享闭合

令 `g_m(X_m;e)` 表示“以 X_m 为输入、以 e 为训练标签”的模型，而不是推理时把未来标签输入模型。

各专家的训练目标与输出：

$$
e_m=r-F_m,\qquad
u_m=g_m(X_m;e_m),\qquad
H_m=F_m+u_m.
$$

第一层 backbone：

$$
\bar F=\frac1M\sum_m F_m,\qquad
\bar u=\frac1M\sum_m u_m,
$$

$$
\boxed{B_1=\frac1M\sum_m H_m=\bar F+\bar u.}
$$

共享闭合器必须以历史 OOF backbone 出题：

$$
e_{C,1}=r-B_1^{OOF},
\qquad
C_1=G_1(X_{\mathrm{parents}};e_{C,1}),
$$

$$
\boxed{S_{\mathrm{FPRC}}=\bar F+\bar u+C_1.}
$$

第一层各修各自的 `r-F_m`；第二层预测的是 **`r-B_1^{OOF}`**，不是直接预测真实平均田 `bar F`。

### 4. FPRC–Pure Gain：只加倍局部修正

统一引入固定增益：

$$
\gamma=
\begin{cases}
1,&\text{FPRC},\\
2,&\text{FPRC–Pure Gain}.
\end{cases}
$$

$$
\boxed{B_\gamma=\bar F+\gamma\bar u,}
\qquad
e_{C,\gamma}=r-B_\gamma^{OOF},
$$

$$
\boxed{
S_\gamma=
\bar F+\gamma\frac1M\sum_m g_m(X_m;r-F_m)
+G_\gamma(X_{\mathrm{parents}};r-B_\gamma^{OOF}).
}
$$

因此 Pure Gain 的完整形式为：

$$
\boxed{
S_{\mathrm{PG}}=
\frac1M\sum_m F_m
+\frac2M\sum_m g_m(X_m;r-F_m)
+G_2\!\left(
X_{\mathrm{parents}};
r-\left[\frac1M\sum_m F_m+\frac2M\sum_m u_m^{OOF}\right]
\right).
}
$$

**三个不变、一个重训：**

- 本田标签仍是 `r-F_m`，不是 `2(r-F_m)`。
- 精确田旁路仍只有一份，不是 `2F_m`。
- 局部模型数量仍是 M，不是训练两套专家取平均。
- B 改变后，closure 必须针对新的 `r-B_gamma^OOF` 重新拟合。

所有跨田聚合均为算术平均。`gamma` 是整体修正增益，不是田权重。两个版本部署时均为 **M 个局部模型 + 1 个闭合器**，另加轻量格点查表；训练成本还包括 OOF 拟合。

代数上 `B_gamma+(r-B_gamma)=r` 恒成立，但预测器只能估计后半项。**代数闭合不等于样本外误差消失，也不保证 gamma=2 优于 gamma=1。**

## 冻结实验成绩

以下数字来自已有冻结实验，**不是本次重写 README 后新跑的收益实验**。完整精度与审计快照见[基准证据说明](docs/benchmarks/README.md)；本包的通用训练接口与原研究运行协议存在明确差别，不能把这些数字当作安装后的默认验收值。

### 数据与统一账本

研究数据为本地 Tushare 缓存构建的 A 股价量、估值及财务因子面板。输入范围 **2018-01-02 至 2026-07-08，共 7,928,060 行**；严格共同评估集为 **2023-01-03 至 2026-07-01，共 3,666,920 个股票—日期观测、844 个信号日期**。

标签是五日后复权收益，减去构建标签时的当日横截面均值。原三田配置：

| 田 | A | B |
|---|---|---|
| Momentum–Reversal | `momentum_20d` | `reversal_5d` |
| Value–Momentum | `pb_rank` | `momentum_20d` |
| Breakout–Volume | `price_ma_ratio` | `volume_change_5d` |

共享 P 为 `turnover_rank`；Q 为 `volatility_20d`、`netprofit_yoy`、`ocf_to_profit`、`profit_stability`。名称不是实时交易建议；财务因子必须按公告可用时间对齐。

主任务为剔除预测分数最低 20% 的股票，其余等权，使用五个错位持有相位。账本使用买入 10 bps、卖出 15 bps 的平均费率 **12.5 bps × 换手**，不是逐笔买卖成交模拟。

### 主表：Gain 与跨田混合的冻结对照

| 方法 | 验证 ICIR | 超额 Sharpe | 累计净超额 | 相对 FPRC | 胜年¹ | 平均换手 |
|---|---:|---:|---:|---:|---:|---:|
| FPRC | 5.0132 | 2.0895 | 19.0977% | — | — | 12.1972% |
| **FPRC–Pure Gain** | **5.1931** | **2.2177** | **19.4699%** | **+37.22 bps** | **2/3** | **12.1378%** |
| Normalized Direct-Other | 4.9227 | 1.9986 | 18.5302% | −56.76 bps | 0/3 | 12.2654% |
| Direct-Other | 5.1413 | 2.1696 | 19.4410% | +34.33 bps | 2/3 | 12.2161% |
| Pure-Other | 4.8767 | 1.9164 | 17.9941% | −110.36 bps | 0/3 | 12.3309% |

后三项是冻结研究中的机制对照，**不是本包提供的模型选项**：Normalized Direct-Other 检查正常总增益下的跨田混合，Direct-Other 同时增加混合与总增益，Pure-Other 只让其他田答题。

¹ 2023 年各变体共用基线回退预测，所以胜年分母仅统计实际存在差异的 2024、2025、2026 三个年度窗口；2026 是不完整年度。若按四个窗口计，Pure Gain 是 **2 胜、1 负、1 平**。

**指标口径必须一起读：**

- 累计净超额按原账本 `sum(net_ret)/5` 累加；不是 CAGR，也不是复利账户累计收益。
- 超额 Sharpe 为 `mean(net_ret)/std(net_ret) × sqrt(252/5)`，基于五日超额标签；不是绝对账户收益 Sharpe。
- ICIR 基于每日 Spearman IC，在五个非重叠相位中分别年化后取平均。
- 换手是该账本按持有相位计算的平均调仓比例，不能直接解释为真实每日全账户换手。

### 逐年与其他交易映射

| 方法 | 2023 | 2024 | 2025 | 2026（部分年度） |
|---|---:|---:|---:|---:|
| FPRC | 3.4177% | 7.8881% | 5.7283% | 2.0636% |
| Pure Gain | 3.4177% | 7.4960% | 5.7605% | 2.7957% |

| 方法 | 主账本最大回撤 | Top-20 净超额 | Top–Bottom 净超额 | Top–Bottom Sharpe |
|---|---:|---:|---:|---:|
| FPRC | −2.7122% | 20.7691% | 110.4468% | 2.2837 |
| Pure Gain | −2.7537% | 28.6110% | 119.8114% | 2.6511 |

Top–Bottom 为多头 1 + 空头 1 的双腿账本，与主任务敞口不同；未模拟借券可得性和借券费，不能直接当成可执行收益承诺。回撤也是原超额账本口径。

### 证据边界

Pure Gain 的 +37.22 bps 点估计约由 **+35.96 bps 毛超额**与 **+1.25 bps 成本节省**构成。采用同日配对、21 日区块、10,000 次 bootstrap，净增益的 95% 区间为：

$$
\boxed{[-92.20,\ +172.13]\ \mathrm{bps}.}
$$

区间跨零，因此结论是：**一个不增加部署模型数量、点估计值得保留的简洁候选；不是已确认的统计胜利。** 2024 年回落、最大回撤未改善，也必须同时报告。多轮研究曾使用这段评估数据，未来晋升还需要未参与设计的数据或独立市场验证。

## 快速开始

在本项目目录内安装，无须将原研究仓库加入 `PYTHONPATH`：

```powershell
mamba activate cutile
cd D:/DesktopAgentProject/Claude/MOE-Grid-Digging/FPRC
python -m pip install .
```

目前未发布 PyPI，请使用本地目录或本地 wheel，不要假定 `pip install fprc` 对应本项目。

```python
from fprc import FPRC, FieldSpec

model = FPRC(
    fields=[
        FieldSpec("momentum_20d", "reversal_5d", "momentum_reversal"),
        FieldSpec("pb_rank", "momentum_20d", "value_momentum"),
        FieldSpec("price_ma_ratio", "volume_change_5d", "breakout_volume"),
    ],
    all_columns=[
        "momentum_20d", "reversal_5d", "pb_rank", "price_ma_ratio",
        "volume_change_5d", "turnover_rank", "volatility_20d",
        "netprofit_yoy", "ocf_to_profit", "profit_stability",
    ],
    apex="turnover_rank",
    q_features=["volatility_20d", "netprofit_yoy",
                "ocf_to_profit", "profit_stability"],
    pure_gain=True,             # False 切回 FPRC
    date_col="date",
    entity_col="symbol",
    label_end_col="label_end",
    device="cuda",              # 默认 cpu
)

model.fit(train_df, target="r", as_of="2026-07-09")
signal = model.predict(future_df)
components = model.predict_components(future_df)

model.save("fprc_pure_gain.joblib")
restored = FPRC.load("fprc_pure_gain.joblib")
```

`train_df` 是每行一个股票—日期的长表，至少包含 date、symbol、因子列、r 和标签结束时间 label_end。所有训练信号日期必须早于 as_of；未来数据日期必须不早于 as_of，预测无须提供 r。输出保持原索引及行顺序。

使用**同一天完整的目标股票截面**做排名，不能按单只股票逐行调用。包不自动合并多张源表，请自行按 date/symbol 合并并确保财务公告时间正确；重复键会报错。也支持同名 MultiIndex。

只用 AB、无需 P/Q 的最简配置：

```python
model = FPRC(
    fields=[("a", "b"), ("c", "d"), ("e", "f")],
    pure_gain=True,
    label_end_col="label_end",
).fit(train_df, target="r")
```

### 接口约定

| 参数/属性 | 含义 |
|---|---|
| `fields` | `FieldSpec` 或 AB 元组列表；支持任意 M，全程等权 |
| `all_columns` | 允许使用的因子清单；可由配置推导，不等于 closure 全量输入 |
| `target` / `y` | 标签列名，或与输入对齐的 Series / 向量 |
| `apex` / `q_features` | 显式指定的可选 P/Q，不自动搜索 |
| `FieldSpec.local_features` | 仅该田额外局部输入，做日内排名，不自动净化 |
| `closure_features` | 默认全部去重 AB parents；更改意味着改变架构 |
| `FieldSpec.score_column` | 可提供外部 PIT 田分数，替代包内建田 |
| `oof_predictions_` | 历史前推样本外分量与预测；warm-up 保留 NaN |
| `training_audit_` | 模型训练/验证边界、最大 label_end、树数与样本量 |
| `unused_columns_` | allowlist 中未实际使用的因子 |

外部田分数在训练和预测表中均须提供。包无法仅凭一列数字证明其无泄漏，调用方必须保证 PIT；score 列不能同时作为普通因子输入。P/AB 缺失按必要输入缺失处理；Q 及可选额外因子 NaN 由 XGBoost 原生处理。不会用 `nanmean` 悄悄改变专家权重。

## 时间安全与审计

1. **建田**：仅消费 `label_end < 当前信号日期` 的成熟历史标签。
2. **局部 OOF**：默认按年扩展训练，块内预测只使用块开始前成熟的标签；可改按季度/月。
3. **共享 OOF**：闭合器用更早时间块的 OOF backbone 构造 `r-B`，而非局部模型的训练内预测。
4. **早停**：每个训练历史的尾部 20% 日期作为验证段，验证边界再次 purge；选树数后用部署前完整成熟历史重拟合。
5. **部署**：最终局部模型使用成熟历史重拟合，closure 的训练标签仍来自 OOF backbone。

`predict(train_df)` 会拒绝训练期日期；历史评估使用 `oof_predictions_`。推荐明确提供真实 label_end。缺省的 `label_horizon=5` 按面板中后五个不同日期推算，不能替代不规则标签的真实成熟时间。

`predict` 使用冻结快照，不消费未来标签、不逐日更新格点统计。需随成熟数据定期重新 `fit`；一次多日期预测不等同于在线逐日更新回测。

**原实验与软件接口的区别**：本包早停采用历史日期比例，旧实验使用固定历史 dev 年；重拟合时间表、数据清洗、外部田与交易账本也会影响结果。方法一致不代表数值逐位一致，不能保证默认安装就复现 2.0895 / 2.2177。

## 复现与贡献

```powershell
python examples/synthetic_panel.py --device cpu
python examples/synthetic_panel.py --device cuda
python -m unittest discover -s tests -v
python -m pip wheel . --no-deps -w dist
```

合成示例自生成数据，完整覆盖训练、预测、分量对账、保存/加载，不依赖私有数据。默认学习器参数面向大面板，小样本示例明确使用较小配置；示例不是收益证明。

| 资源 | 用途 |
|---|---|
| [真实面板示例](examples/old_three_fields.py) | 原三田字段配置和本地数据调用 |
| [合成面板示例](examples/synthetic_panel.py) | 无私有数据即可运行端到端流程 |
| [验证记录](docs/VERIFICATION.md) | 测试、CPU/CUDA、独立安装验证范围 |
| [来源记录](docs/PROVENANCE.md) | 复制来源、哈希与原代码保护 |
| [基准证据](docs/benchmarks/README.md) | 主表、逐年、区块推断与审计原始结果 |
| [实现](src/fprc/model.py) | 可审阅的训练、OOF 和闭合逻辑 |

项目不依赖运行时导入原研究脚本；不捆绑受数据授权约束的原始金融数据库，也不提供订单执行。joblib 文件可能执行代码，只加载可信来源，并保持依赖版本一致。

欢迎优先贡献：可复现的泄漏边界测试、不同市场的同协议验证、内存/训练效率改进、文档与示例。报告结果时请同时提供数据区间、标签与成熟时间、田/输入配置、训练协议、成本账本、逐年差异与配对不确定性，不只提交最优 Sharpe。

## License

FPRC is licensed under the [MIT License](LICENSE). Copyright (c) 2026 Republic1024.

项目代码及随附文档采用 MIT 许可证。第三方依赖保留各自的许可证；未随项目分发的金融数据库不在本项目许可证授权范围内。代码来源记录见 [PROVENANCE.md](docs/PROVENANCE.md)。
