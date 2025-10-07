"""凝视指标计算（最小版）。

输入：
- gaze_series: List[Tuple[t_ms, u, v, valid]]，其中 (u, v) 为归一化坐标 [0,1]；
  t_ms 单调递增（毫秒）；valid=False 表示该帧无效（丢失或置信度低）。
- ROI：以归一化坐标定义的矩形区域。

输出：
- 停留时长（毫秒）；首次注视延迟（毫秒，不存在返回 None）；丢失率等。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any


Gaze = Tuple[float, float, float, bool]  # (t_ms, u, v, valid)


@dataclass
class ROI:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, u: float, v: float) -> bool:
        return (self.x0 <= u <= self.x1) and (self.y0 <= v <= self.y1)


def compute_dwell_time(gaze_series: List[Gaze], roi: ROI) -> float:
    """计算在 ROI 内的累计停留时长（毫秒）。"""

    total = 0.0
    for i in range(1, len(gaze_series)):
        t0, u0, v0, ok0 = gaze_series[i - 1]
        t1, u1, v1, ok1 = gaze_series[i]
        dt = max(0.0, t1 - t0)
        if ok0 and roi.contains(u0, v0):
            total += dt
    return total


def compute_first_fixation_latency(
    gaze_series: List[Gaze], roi: ROI, min_fix_ms: float = 100.0, max_latency_ms: float = 4000.0
) -> Optional[float]:
    """首次注视延迟：进入 ROI 后连续驻留至少 min_fix_ms 的最早时间。

    - 若超出 max_latency_ms 仍未满足，返回 None。
    - 计算从序列起点（gaze_series[0].t_ms）开始计时。
    """

    if not gaze_series:
        return None
    t_start = gaze_series[0][0]
    enter_t: Optional[float] = None
    acc = 0.0
    for i in range(1, len(gaze_series)):
        t0, u0, v0, ok0 = gaze_series[i - 1]
        t1, u1, v1, ok1 = gaze_series[i]
        dt = max(0.0, t1 - t0)
        within = ok0 and roi.contains(u0, v0)
        if within:
            if enter_t is None:
                enter_t = t0
                acc = 0.0
            acc += dt
            if acc >= min_fix_ms:
                return enter_t - t_start
        else:
            enter_t = None
            acc = 0.0
        if (t1 - t_start) > max_latency_ms:
            break
    return None


def compute_missing_rate(gaze_series: List[Gaze]) -> float:
    if not gaze_series:
        return 1.0
    invalid = sum(1 for (_, _, _, ok) in gaze_series if not ok)
    return invalid / float(len(gaze_series))


def compute_metrics_for_rois(
    gaze_series: List[Gaze], rois: List[ROI], min_fix_ms: float = 100.0
) -> Dict[str, Any]:
    """批量计算 ROI 指标。"""

    out: Dict[str, Any] = {"missing_rate": compute_missing_rate(gaze_series)}
    for r in rois:
        dwell = compute_dwell_time(gaze_series, r)
        lat = compute_first_fixation_latency(gaze_series, r, min_fix_ms=min_fix_ms)
        out[r.name] = {"dwell_ms": dwell, "first_fix_ms": lat}
    return out
