"""
摄像头服务模块

提供跨平台的摄像头设备枚举、打开/关闭、分辨率设置与逐帧读取能力。
优先满足桌面端原型需求，后续可扩展多路摄像头、硬件加速与视频文件输入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import os
import sys
import platform

# 尽量压制 OpenCV 的冗余日志
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2


@dataclass
class CameraInfo:
    """摄像头设备信息"""

    index: int
    name: str = "Unknown Camera"


class CameraService:
    """简化的摄像头采集封装。

    - 支持基础的设备枚举（尝试打开探测）。
    - 支持设置分辨率（不保证所有设备都成功）。
    - 提供单帧读取接口 `read()`。
    """

    def __init__(self) -> None:
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_index: Optional[int] = None

    @staticmethod
    def _backend_candidates() -> List[int]:
        """后端优先级：尽量只尝试一次，减少控制台噪声。"""
        is_win = sys.platform.startswith("win")
        cands: List[int] = []
        # 先让 OpenCV 自选一次（通常是 MSMF）；失败再回退到 DSHOW（Windows）
        if hasattr(cv2, "CAP_ANY"):
            cands.append(cv2.CAP_ANY)
        if is_win and hasattr(cv2, "CAP_DSHOW"):
            cands.append(cv2.CAP_DSHOW)
        return cands or [0]

    @staticmethod
    def list_devices(max_devices: int = 1) -> List[CameraInfo]:
        """保守返回默认摄像头，避免枚举导致的多后端 WARN。

        后续若需要精确枚举，再按平台能力增强。
        """
        return [CameraInfo(index=0, name="默认摄像头")]

    def open(self, index: int, resolution: Tuple[int, int] | None = None) -> bool:
        """打开指定索引的设备，可选设置分辨率。"""

        self.close()
        # 按候选后端依次尝试（最多两次），避免多后端多索引的噪声
        backends = self._backend_candidates()
        cap = None
        for be in backends:
            cap = cv2.VideoCapture(index) if (be == getattr(cv2, "CAP_ANY", 0)) else cv2.VideoCapture(index, be)
            if cap is not None and cap.isOpened():
                break
            if cap is not None:
                cap.release()
                cap = None
        if cap is None or not cap.isOpened():
            self.cap = None
            return False
        self.cap = cap
        self.current_index = index
        if resolution:
            w, h = resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
        return True

    def close(self) -> None:
        """关闭当前摄像头。"""

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.current_index = None

    def read(self) -> Tuple[bool, Optional["np.ndarray"]]:
        """读取一帧图像。

        返回 (ok, frame)。失败时 frame 为 None。
        """

        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        return True, frame
