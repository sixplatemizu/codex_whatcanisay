"""
Flet 桌面端原型

功能：
- 设备选择、分辨率选择
- 启动/停止摄像头
- 实时预览与 FPS 显示

说明：
- 以简洁、美观、易理解为目标；后续迭代加入人脸与眼动追踪。
"""

from __future__ import annotations

import base64
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import flet as ft
import numpy as np

# 确保以脚本运行时可导入到 src/services
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from services.camera import CameraService


@dataclass
class AppState:
    capturing: bool = False
    fps: float = 0.0
    selected_device: int = 0
    resolution: Tuple[int, int] = (1280, 720)
    requested_resolution: Tuple[int, int] = (1280, 720)
    device_idx: int = 0
    reported_resolution: bool = False
    _stop_event: Optional[threading.Event] = None


def frame_to_jpeg_base64(frame: np.ndarray, max_width: int = 960, quality: int = 80) -> str:
    """缩放并编码为 JPEG 的 base64。

    - 先将帧按宽度限制到 max_width（保持纵横比，采用 AREA 插值以提升缩小质量）
    - 再以给定质量 JPEG 编码，减小 UI 传输/渲染开销
    """

    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        new_size = (int(w * scale), int(h * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return ""
    return base64.b64encode(buf).decode("ascii")


def main(page: ft.Page) -> None:
    page.title = "Eye Track Prototype"
    page.window_width = 1100
    page.window_height = 740
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 16
    page.bgcolor = "#FAFAFA"  # 浅色背景，避免依赖 flet.colors

    cam = CameraService()
    state = AppState()

    # UI 组件
    device_options = [
        ft.dropdown.Option(f"{info.index}: {info.name}") for info in cam.list_devices(max_devices=4)
    ]
    device_dd = ft.Dropdown(
        label="摄像头",
        options=device_options,
        value=device_options[0].key if device_options else None,
        width=260,
    )

    res_dd = ft.Dropdown(
        label="分辨率",
        options=[
            ft.dropdown.Option("640x480"),
            ft.dropdown.Option("1280x720"),
            ft.dropdown.Option("1920x1080"),
        ],
        value="1280x720",
        width=160,
    )

    fps_text = ft.Text("FPS: 0.0", size=14, color="#616161")  # 实测帧率（UI 刷新统计）
    status_text = ft.Text("就绪", size=14, color="#616161")
    mirror_switch = ft.Switch(label="镜像预览", value=True)

    img = ft.Image(
        src_base64=None,
        fit=ft.ImageFit.CONTAIN,
        border_radius=8,
        expand=1,
    )

    start_btn = ft.ElevatedButton("启动", icon="play_arrow")
    stop_btn = ft.OutlinedButton("停止", icon="stop", disabled=True)

    def parse_device_value(val: str) -> int:
        try:
            return int(val.split(":")[0])
        except Exception:
            return 0

    def parse_resolution(val: str) -> Tuple[int, int]:
        try:
            w, h = val.lower().split("x")
            return int(w), int(h)
        except Exception:
            return 1280, 720

    def on_start_click(e: ft.ControlEvent) -> None:
        nonlocal state
        if state.capturing:
            return
        device_idx = parse_device_value(device_dd.value or "0: Camera")
        res = parse_resolution(res_dd.value or "1280x720")
        ok = cam.open(device_idx, res)
        if not ok:
            status_text.value = f"无法打开摄像头 {device_idx}"
            page.update()
            return
        state.capturing = True
        state._stop_event = threading.Event()
        state.device_idx = device_idx
        state.requested_resolution = res
        state.reported_resolution = False
        status_text.value = f"采集中（设备 {device_idx}，请求 {res[0]}x{res[1]}，实际/驱动 FPS 检测中…）"
        start_btn.disabled = True
        stop_btn.disabled = False
        page.update()
        page.run_thread(lambda: capture_loop())

    def on_stop_click(e: ft.ControlEvent) -> None:
        nonlocal state
        if not state.capturing:
            return
        if state._stop_event:
            state._stop_event.set()
        state.capturing = False
        cam.close()
        status_text.value = "已停止"
        start_btn.disabled = False
        stop_btn.disabled = True
        page.update()

    def capture_loop() -> None:
        nonlocal state
        frame_count = 0
        t0 = time.perf_counter()
        last_ui = 0.0
        try:
            while state.capturing and state._stop_event and not state._stop_event.is_set():
                ok, frame = cam.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                frame_count += 1
                now = time.perf_counter()
                dt = now - t0
                if dt > 0.25:
                    state.fps = frame_count / dt
                    t0 = now
                    frame_count = 0
                # 首帧或首次报告：显示实际分辨率与驱动回报 FPS
                if not state.reported_resolution:
                    h, w = frame.shape[:2]
                    rw, rh, r_fps = cam.get_reported_props()
                    status_text.value = (
                        f"采集中（设备 {state.device_idx}，请求 {state.requested_resolution[0]}x{state.requested_resolution[1]}，实际 {w}x{h}，驱动FPS {r_fps:.0f}）"
                    )
                    state.reported_resolution = True
                # 可选镜像（自拍预览更符合直觉）
                if mirror_switch.value:
                    frame = cv2.flip(frame, 1)
                # 编码与 UI 更新节流（~30 FPS）
                if now - last_ui >= 1.0 / 30.0:
                    img.src_base64 = frame_to_jpeg_base64(frame, max_width=960, quality=80)
                    fps_text.value = f"FPS: {state.fps:.1f}"
                    page.update()
                    last_ui = now
        finally:
            # 退出时确保资源释放与 UI 复位
            cam.close()

    start_btn.on_click = on_start_click
    stop_btn.on_click = on_stop_click

    # 布局
    appbar = ft.AppBar(title=ft.Text("眼动追踪原型"), center_title=False, bgcolor="#ECEFF1")
    page.appbar = appbar

    controls_row = ft.Row(
        controls=[device_dd, res_dd, mirror_switch, start_btn, stop_btn, fps_text, status_text],
        alignment=ft.MainAxisAlignment.START,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=12,
        wrap=False,
    )

    preview_card = ft.Card(
        content=ft.Container(content=img, padding=10, expand=1),
        elevation=2,
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
    )

    content_column = ft.Column(
        expand=1,
        spacing=8,
        controls=[
            controls_row,
            ft.Container(expand=1, content=preview_card),
        ],
    )

    page.add(content_column)

    def on_close(e: ft.ControlEvent) -> None:
        if state._stop_event:
            state._stop_event.set()
        cam.close()

    page.on_close = on_close


def cli() -> None:
    """命令行入口：启动 Flet 应用。"""

    ft.app(target=main)


if __name__ == "__main__":
    # 本地运行（方案一）：uv run flet run src/app
    # 本地运行（方案二）：uv run -m app.main
    cli()
