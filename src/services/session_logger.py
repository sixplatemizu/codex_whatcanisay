"""会话日志记录（最小版）。

- 将关键事件、设备与版本信息、指标结果落盘为 JSON；
- 默认输出到 `sessions/` 目录，文件名包含时间戳；
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any


class SessionLogger:
    def __init__(self, out_dir: str = "sessions") -> None:
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.out_dir, f"session_{ts}.json")
        self.data: dict[str, Any] = {
            "created_at": ts,
            "device": {},
            "events": [],
            "metrics": {},
            "meta": {},
        }

    def set_device_info(self, **kwargs: Any) -> None:
        self.data["device"].update(kwargs)

    def add_event(self, etype: str, **payload: Any) -> None:
        evt = {"t": time.time(), "type": etype, "data": payload}
        self.data["events"].append(evt)

    def set_metrics(self, **metrics: Any) -> None:
        self.data["metrics"].update(metrics)

    def set_meta(self, **meta: Any) -> None:
        self.data["meta"].update(meta)

    def save(self) -> None:
        def default(o: Any) -> Any:
            if is_dataclass(o):
                return asdict(o)
            return o

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2, default=default)
