"""刺激播放与时间轴（最小骨架）。

职责：
- 读取 `assets/stimuli/<id>.json` 时间轴；
- 基于单调时钟驱动事件触发（segment enter/exit、dialogue turn、name_call）；
- 提供回调接口（on_event）给 UI 与日志层；

说明：
- 此版本不强依赖获取视频的精准播放位置，默认以点击播放时的 `time.perf_counter()` 为基准；
- 若后续需要更高精度，可切换到外部播放器（例如 python-vlc）并用其 `get_time()` 作为 `pos_ms`；
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import json
import os
import threading
import time


@dataclass
class Segment:
    name: str
    start_ms: float
    end_ms: float
    tags: Optional[List[str]] = None


@dataclass
class DialogueTurn:
    speaker: str  # "left" | "right"
    start_ms: float
    end_ms: float


@dataclass
class NameCall:
    t_ms: float
    label: str


@dataclass
class StimulusTimeline:
    id: str
    segments: List[Segment]
    dialogue_turns: List[DialogueTurn]
    name_calls: List[NameCall]
    sides: List[Dict[str, Any]]  # 例如 [{"t_ms":0, "person_side":"left","toy_side":"right"}]


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_timeline_from_json(path: str) -> StimulusTimeline:
    obj = _read_json(path)
    sid = obj.get("id") or os.path.splitext(os.path.basename(path))[0]
    def _to_segments() -> List[Segment]:
        out: List[Segment] = []
        for it in obj.get("segments", []) or []:
            out.append(
                Segment(
                    name=str(it.get("name", "segment")),
                    start_ms=float(it.get("start_ms", 0.0)),
                    end_ms=float(it.get("end_ms", 0.0)),
                    tags=list(it.get("tags") or []) or None,
                )
            )
        return out
    def _to_turns() -> List[DialogueTurn]:
        out: List[DialogueTurn] = []
        for it in obj.get("dialogue_turns", []) or []:
            out.append(
                DialogueTurn(
                    speaker=str(it.get("speaker", "left")),
                    start_ms=float(it.get("start_ms", 0.0)),
                    end_ms=float(it.get("end_ms", 0.0)),
                )
            )
        return out
    def _to_names() -> List[NameCall]:
        out: List[NameCall] = []
        for it in obj.get("name_calls", []) or []:
            out.append(NameCall(t_ms=float(it.get("t_ms", 0.0)), label=str(it.get("label", "name_call"))))
        return out
    sides = list(obj.get("sides", []) or [])
    return StimulusTimeline(id=sid, segments=_to_segments(), dialogue_turns=_to_turns(), name_calls=_to_names(), sides=sides)


class StimulusPlayer:
    """时间轴驱动器（不直接操控视频，仅负责事件时序）。"""

    def __init__(self, stimulus_id: str, timeline: StimulusTimeline) -> None:
        self.stimulus_id = stimulus_id
        self.timeline = timeline
        self.on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._t0_ms: float = 0.0
        self._last_pos_ms: float = 0.0
        self._fired: Dict[str, bool] = {}

    def _emit(self, etype: str, **payload: Any) -> None:
        cb = self.on_event
        if cb:
            try:
                cb(etype, {**payload, "stimulus_id": self.stimulus_id})
            except Exception:
                pass

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._t0_ms = time.perf_counter() * 1000.0
        self._last_pos_ms = 0.0
        self._fired.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._emit("stimulus_start", t_ms=self._t0_ms)

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop.set()
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        self._emit("stimulus_stop", t_ms=time.perf_counter() * 1000.0)

    def _run_loop(self) -> None:
        try:
            while not self._stop.is_set():
                now_ms = time.perf_counter() * 1000.0
                pos_ms = now_ms - self._t0_ms
                self._tick(self._last_pos_ms, pos_ms)
                self._last_pos_ms = pos_ms
                time.sleep(0.02)  # 50Hz
        except Exception:
            pass

    def _tick(self, last_ms: float, pos_ms: float) -> None:
        # segment enter/exit
        for seg in self.timeline.segments:
            key_enter = f"seg@{seg.name}:enter:{seg.start_ms}"
            key_exit = f"seg@{seg.name}:exit:{seg.end_ms}"
            if last_ms < seg.start_ms <= pos_ms and not self._fired.get(key_enter):
                self._fired[key_enter] = True
                self._emit("segment_enter", name=seg.name, t_ms=seg.start_ms, tags=seg.tags or [])
            if last_ms < seg.end_ms <= pos_ms and not self._fired.get(key_exit):
                self._fired[key_exit] = True
                self._emit("segment_exit", name=seg.name, t_ms=seg.end_ms, tags=seg.tags or [])

        # dialogue turns
        for dt in self.timeline.dialogue_turns:
            k1 = f"dlg@{dt.speaker}:start:{dt.start_ms}"
            k2 = f"dlg@{dt.speaker}:end:{dt.end_ms}"
            if last_ms < dt.start_ms <= pos_ms and not self._fired.get(k1):
                self._fired[k1] = True
                self._emit("dialogue_turn_start", speaker=dt.speaker, t_ms=dt.start_ms)
            if last_ms < dt.end_ms <= pos_ms and not self._fired.get(k2):
                self._fired[k2] = True
                self._emit("dialogue_turn_end", speaker=dt.speaker, t_ms=dt.end_ms)

        # name calls
        for nc in self.timeline.name_calls:
            k = f"name@{nc.label}:{nc.t_ms}"
            if last_ms < nc.t_ms <= pos_ms and not self._fired.get(k):
                self._fired[k] = True
                self._emit("name_call", label=nc.label, t_ms=nc.t_ms)

