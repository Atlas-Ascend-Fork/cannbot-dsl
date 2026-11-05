# 比较与判断

比较接口的返回形式分为以下三类。选择接口时，先确认需要保留每个分量的结果，还是只需要整体判断。

| 返回形式 | 接口示例 | 两个分量的比较结果分别为真、假时 |
| --- | --- | --- |
| 逐分量数值 | `veq` | 返回短向量`(1.0, 0.0)`。 |
| 整体布尔值 | `vbeq` | 返回`False`，因为并非所有分量都满足条件。 |
| 整数位图 | `veq_mask` | 返回`65535`，表示`.x`分量比较成立、`.y`分量比较不成立；完整取值见 [veq_mask返回值说明](./veq-mask.md#返回值说明)。 |

`vbeq_mask`等`vb*_mask`接口是对应布尔接口的兼容别名，仍返回布尔值。它们与`veq_mask`等整数位图接口不同。

- [eq](./eq.md)
- [veq](./veq.md)
- [veq_mask](./veq-mask.md)
- [vequ](./vequ.md)
- [vequ_mask](./vequ-mask.md)
- [vbeq](./vbeq.md)
- [vbeq_mask](./vbeq-mask.md)
- [ne](./ne.md)
- [vne](./vne.md)
- [vne_mask](./vne-mask.md)
- [vneu](./vneu.md)
- [vneu_mask](./vneu-mask.md)
- [vbne](./vbne.md)
- [vbne_mask](./vbne-mask.md)
- [le](./le.md)
- [vle](./vle.md)
- [vle_mask](./vle-mask.md)
- [vleu](./vleu.md)
- [vleu_mask](./vleu-mask.md)
- [vble](./vble.md)
- [vble_mask](./vble-mask.md)
- [ge](./ge.md)
- [vge](./vge.md)
- [vge_mask](./vge-mask.md)
- [vgeu](./vgeu.md)
- [vgeu_mask](./vgeu-mask.md)
- [vbge](./vbge.md)
- [vbge_mask](./vbge-mask.md)
- [lt](./lt.md)
- [vlt](./vlt.md)
- [vlt_mask](./vlt-mask.md)
- [vltu](./vltu.md)
- [vltu_mask](./vltu-mask.md)
- [vblt](./vblt.md)
- [vblt_mask](./vblt-mask.md)
- [gt](./gt.md)
- [vgt](./vgt.md)
- [vgt_mask](./vgt-mask.md)
- [vgtu](./vgtu.md)
- [vgtu_mask](./vgtu-mask.md)
- [vbgt](./vbgt.md)
- [vbgt_mask](./vbgt-mask.md)
- [isfinite](./isfinite.md)
- [isnan](./isnan.md)
- [visnan](./visnan.md)
- [isinf](./isinf.md)
