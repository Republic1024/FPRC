# 来源与实现边界

所有新增及修改文件都在 `FPRC/`，原研究文件仅被读取。

## 原样复制

`src/fprc/_cellstats.py` ← `../src/wuspace/l1_field.py`，复制时 SHA256：

`289258E178BE695CA3E8261B976B89D8B507BF72F1B60D0BC1E7AC1D6475D054`

CellStats/EB/confidence 直接使用这份副本，运行时不导入 `wuspace` 或 `FieldAltasVision`。

## 公式与特征来源（只读）

| 文件 | SHA256 |
|---|---|
| FieldAltasVision/va_residual_frame_gain_mixing.py | 31E6831B342D87399D32217222909859FEDD2CE60CE58D6BDF7366733AAE6BD6 |
| FieldAltasVision/va_c_fprc.py | 6C01B8ADA8485103687DFCAE3E7D7B3AA48CEEA990292798E8A1B57BC936A36B |
| FieldAltasVision/va_stagewise_lfsi.py | C0F443077EF36B92268C5CFF4AF08B6A6193AB2D0DEAD69E7286ADC8F2F51C50 |
| FieldAltasVision/va_multifield_fare.py | E11DC7EBA40012ADC5A2BA1A273EC0E4649EE5DA7295F26BC3AA694407411959 |
| FieldAltasVision/va_hierarchical_residual_ladder.py | 71C2283F52F6308FE7D73DB122FAD68FBC4A0342AC852B6215531DE54E6F2206 |

Pure Gain 固定公式 `Fmean + 2*mean(local)`，最终 parents-only residual closure。共享 P 对全 parents rank 净化；Q 对 parents rank + 每田的 raw-AB 净化 P rank 净化。显式提供 apex/q_features 才启用对应坐标。

## 通用化设计

- 单独类配置、fit、predict；字段映射和局部所有权可检查。
- 默认 XGBoost 大面板参数来自 `va_cube_gbdt_same_features.py`；device 默认为 CPU，可显式用 CUDA。
- 训练边界使用显式 label_end 严格小于时间边界；缺省 horizon 基于观察日期推导。
- 早停使用训练期尾部日期比例，并 purge；旧实验则使用固定上一日历年 dev。不会宣称逐位复现原模型。
- 任意 M 等权田；不存在以表现优化田权重的步骤。
- 预测使用已拟合田的冻结统计快照；外部 PIT score_column 可由调用方的生产建田器提供。
- 不携带旧回测数据、已选训练参数、模型或旧预测缓存。
