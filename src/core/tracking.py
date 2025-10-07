"""
基于 MediaPipe FaceMesh 的人脸/眼部关键点检测与跟踪。

默认在降尺度帧上推理以提升吞吐与帧率；输出坐标映射回原始帧尺寸。
并尽量降低底层 TFLite/MediaPipe 的冗余日志噪声。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import os
import cv2
import numpy as np

# 常用关键点索引（基于 MediaPipe FaceMesh 拓展，含虹膜）
# 降低 TensorFlow Lite / absl 的日志等级
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all,1=INFO,2=WARNING,3=ERROR
os.environ.setdefault("GLOG_minloglevel", "2")
try:
    from absl import logging as absl_logging  # type: ignore

    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.use_absl_handler()
except Exception:
    pass

try:
    import mediapipe as mp
except Exception as e:  # pragma: no cover - 运行时缺失依赖时给出清晰提示
    mp = None  # type: ignore


# 常用关键点索引（基于 MediaPipe FaceMesh 拓展，含虹膜）
# 参考：官方示例与社区共识（可能因版本存在微小差异）
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS = [473, 474, 475, 476, 477]
EYE_CORNERS = [33, 133, 362, 263]  # 眼角（左外、左内、右内、右外）


@dataclass
class TrackingResult:
    points: List[Tuple[int, int]]  # 关键点像素坐标（对应输入帧尺寸）
    iris_centers: List[Tuple[int, int]]  # 左/右虹膜中心
    face_box: Optional[Tuple[int, int, int, int]] = None  # 可选的人脸包围盒（x,y,w,h）


class PointSmoother:
    """二维点的指数平滑器（低延迟、轻量级）。

    说明：
    - 使用 EMA（Exponential Moving Average）对 (x, y) 进行平滑。
    - `alpha` 越小，越平滑但滞后越明显；建议 0.3~0.8 之间调节。
    """

    def __init__(self, alpha: float = 0.6) -> None:
        self.alpha = float(alpha)
        self._prev: Optional[Tuple[int, int]] = None

    def reset(self) -> None:
        self._prev = None

    def filter(self, pt: Tuple[int, int]) -> Tuple[int, int]:
        if self._prev is None:
            self._prev = pt
            return pt
        ax = int(self.alpha * pt[0] + (1.0 - self.alpha) * self._prev[0])
        ay = int(self.alpha * pt[1] + (1.0 - self.alpha) * self._prev[1])
        self._prev = (ax, ay)
        return self._prev


class FaceTracker:
    def __init__(
        self,
        infer_width: int = 640,
        max_faces: int = 1,
        min_det_conf: float = 0.5,
        min_trk_conf: float = 0.5,
        smooth_iris: bool = True,
        smooth_alpha: float = 0.6,
    ) -> None:
        if mp is None:
            raise RuntimeError(
                "未安装 mediapipe，请执行：uv add mediapipe 或更新 pyproject 依赖后 uv sync"
            )
        self.infer_width = infer_width
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,  # 启用虹膜关键点
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_trk_conf,
        )
        self.smooth_iris = bool(smooth_iris)
        self._iris_smoothers: Optional[List[PointSmoother]] = None
        if self.smooth_iris:
            self._iris_smoothers = [
                PointSmoother(alpha=smooth_alpha),
                PointSmoother(alpha=smooth_alpha),
            ]

    def close(self) -> None:
        try:
            self.face_mesh.close()
        except Exception:
            pass

    def process(self, frame_bgr: np.ndarray) -> Optional[TrackingResult]:
        """在降尺度帧上推理，返回映射回原尺寸的关键点。"""

        h0, w0 = frame_bgr.shape[:2]
        # 推理尺寸：按宽度限制到 infer_width
        scale = 1.0
        if w0 > self.infer_width:
            scale = self.infer_width / float(w0)
            new_size = (int(w0 * scale), int(h0 * scale))
            infer_frame = cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)
        else:
            infer_frame = frame_bgr

        rgb = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        # 仅取第一张人脸
        lms = res.multi_face_landmarks[0].landmark

        def to_px(idx: int) -> Tuple[int, int]:
            # 归一化坐标 → 原始帧像素坐标
            x = int(lms[idx].x * w0)
            y = int(lms[idx].y * h0)
            return x, y

        points: List[Tuple[int, int]] = []
        for idx in EYE_CORNERS:
            if idx < len(lms):
                points.append(to_px(idx))

        # 虹膜中心（平均 5 点）
        iris_centers: List[Tuple[int, int]] = []
        for iris in (LEFT_IRIS, RIGHT_IRIS):
            xs, ys = [], []
            for idx in iris:
                if idx < len(lms):
                    x, y = to_px(idx)
                    xs.append(x)
                    ys.append(y)
            if xs and ys:
                cx = int(sum(xs) / len(xs))
                cy = int(sum(ys) / len(ys))
                iris_centers.append((cx, cy))

        # 可选：对虹膜中心做轻量平滑，减少抖动
        if self.smooth_iris and self._iris_smoothers is not None and len(iris_centers) == 2:
            iris_centers = [
                self._iris_smoothers[0].filter(iris_centers[0]),
                self._iris_smoothers[1].filter(iris_centers[1]),
            ]

        # 可选：估计包围盒（基于眼角与虹膜点的极值）
        all_pts = points + iris_centers
        face_box = None
        if all_pts:
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            # 适度扩展边界
            pad = max(8, int(0.05 * max(w0, h0)))
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(w0 - 1, x1 + pad)
            y1 = min(h0 - 1, y1 + pad)
            face_box = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

        return TrackingResult(points=points, iris_centers=iris_centers, face_box=face_box)

    @staticmethod
    def draw_overlays(
        frame_bgr: np.ndarray,
        tr: TrackingResult,
        color_points: Tuple[int, int, int] = (0, 255, 0),
        color_iris: Tuple[int, int, int] = (255, 0, 0),
        color_box: Tuple[int, int, int] = (0, 170, 255),
    ) -> None:
        """在帧上叠加可视化图层（原地修改）。"""
        for x, y in tr.points:
            cv2.circle(frame_bgr, (x, y), 2, color_points, thickness=-1, lineType=cv2.LINE_AA)
        for x, y in tr.iris_centers:
            cv2.circle(frame_bgr, (x, y), 3, color_iris, thickness=-1, lineType=cv2.LINE_AA)
        if tr.face_box is not None:
            x, y, w, h = tr.face_box
            cv2.rectangle(
                frame_bgr, (x, y), (x + w, y + h), color_box, thickness=1, lineType=cv2.LINE_AA
            )
