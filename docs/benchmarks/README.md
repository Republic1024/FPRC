# Frozen benchmark evidence

本目录保存 README 所引用的冻结实验结果快照，不包含原始金融数据库，也不表示本次软件封装重新运行了收益实验。

来源：研究仓库 `FieldAltasVision/out/residual_frame_gain_mixing/`，驱动脚本 `va_residual_frame_gain_mixing.py`，交易评估基于 `va_farr_fixed_anchor_relay.py`。文件按原样复制，保留全精度：

- [main.csv](residual_frame_gain_mixing/main.csv)：各变体主指标与其他交易映射。
- [yearly.csv](residual_frame_gain_mixing/yearly.csv)：逐年账本。
- [inference.csv](residual_frame_gain_mixing/inference.csv)：同日配对、21 日区块、10,000 次 bootstrap 推断。
- [audit.json](residual_frame_gain_mixing/audit.json)：公式、共同样本、2023 回退与参数审计。

方法 ID：`base` 为 FPRC；`gain2` 为 Pure Gain；`norm_do` 为 Normalized Direct-Other；`direct_other` 和 `pure_other` 对应同名机制对照。软件包仅实现前两者。

主任务是 bottom-20 剔除后的等权组合；标签为五日截面超额收益。`net_pct` 按原账本 sum(net_ret)/5 累加，不是 CAGR；Sharpe 为 mean/std × sqrt(252/5)，不是绝对收益 Sharpe。CSV 的换手比例乘 100 后成为 README 中的百分比；相对收益百分点乘 100 后成为 bps。

所有变体在 2023 年共用基线回退；实际变体胜年统计应以 2024–2026 为分母，2026 为部分年度。Pure Gain 的增益区间跨零，不支持统计确认的晋升。详细数据区间、成本与证据边界见项目 README。

这些快照可核对表格，不能单独重现模型训练或每日区块抽样；完整重现实验还需要原始授权面板、冻结预测、每日账本与原研究运行环境。本包通用 OOF/早停协议不宣称与旧实验逐位一致。
