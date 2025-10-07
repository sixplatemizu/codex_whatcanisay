"""校准与映射（最小可行版）

提供将虹膜中心（或双眼中点）的图像像素坐标，映射到屏幕/预览的归一化坐标 [0,1]^2。

设计要点（简化假设）：
- 采用二维仿射映射（2x3 矩阵），最少需要 3 个样本点，建议 5–9 点；
- 源点：图像坐标（x, y，单位：像素）；
- 目标点：归一化坐标（u, v ∈ [0,1]，以 UI 预览区域为基准）；
- 预测：将任意图像点映射为 (u, v)，超界时裁剪到 [0,1]；
- 序列化：支持保存/加载矩阵参数。

后续可扩展：
- 使用多项式/薄板样条或根据头姿做自适应校正；
- 多分段/分区拟合，改善边缘误差。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List, Tuple, Optional, Dict, Any

import numpy as np


Point = Tuple[float, float]


@dataclass
class AffineCalibrationModel:
    """二维仿射映射模型：y = A @ [x, y, 1].

    - A 形状为 (2, 3)。
    - predict 输入/输出均为二维点。
    """

    A: np.ndarray  # (2, 3)

    def predict(self, pt: Point, clip: bool = True) -> Point:
        x, y = float(pt[0]), float(pt[1])
        vec = np.array([x, y, 1.0], dtype=np.float64)
        u, v = (self.A @ vec).tolist()
        if clip:
            u = min(max(u, 0.0), 1.0)
            v = min(max(v, 0.0), 1.0)
        return float(u), float(v)

    def to_dict(self) -> Dict[str, Any]:
        return {"A": self.A.tolist()}

    @staticmethod
    def from_dict(obj: Dict[str, Any]) -> "AffineCalibrationModel":
        A = np.array(obj["A"], dtype=np.float64)
        assert A.shape == (2, 3)
        return AffineCalibrationModel(A=A)


class AffineCalibrator:
    """仿射校准器。

    用法：
        cal = AffineCalibrator()
        cal.add_sample((x_img, y_img), (u, v))  # 多次
        model = cal.fit()
        u, v = model.predict((x_img, y_img))
    """

    def __init__(self) -> None:
        self._src: List[Point] = []
        self._dst: List[Point] = []

    def add_sample(self, src_xy: Point, dst_uv: Point) -> None:
        self._src.append((float(src_xy[0]), float(src_xy[1])))
        self._dst.append((float(dst_uv[0]), float(dst_uv[1])))

    @property
    def num_samples(self) -> int:
        return len(self._src)

    def clear(self) -> None:
        self._src.clear()
        self._dst.clear()

    def fit(self) -> AffineCalibrationModel:
        if len(self._src) < 3:
            raise ValueError("至少需要 3 个样本点进行仿射拟合")
        # 构造最小二乘问题：U = X @ W，其中 X=[x,y,1]，W^T=A
        X = np.array([[x, y, 1.0] for (x, y) in self._src], dtype=np.float64)  # (N,3)
        U = np.array(self._dst, dtype=np.float64)  # (N,2)
        # 解 W (3,2)
        W, *_ = np.linalg.lstsq(X, U, rcond=None)
        A = W.T  # (2,3)
        return AffineCalibrationModel(A=A)


def average_iris_center(iris_centers: Iterable[Point]) -> Optional[Point]:
    """将双眼虹膜中心取平均，得到 gaze 的简化代理点（图像像素坐标）。"""

    pts = list(iris_centers)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))
