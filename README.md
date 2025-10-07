# Eye-Track Vibecoding

一个基于 Flet + OpenCV + MediaPipe 的轻量级眼动/人脸关键点原型项目，目标是快速验证“摄像头采集 → 人脸/虹膜关键点检测 → 叠加可视化 → UI 交互”的端侧流程，并为后续视线映射、校准与扩展交互打基础。

## 功能概览

- 摄像头接入与采集（`services/CameraService`）
  - 设备打开/关闭、分辨率与 FOURCC 设定（优先 MJPG，回退默认）。
  - 单帧读取接口、设备属性回报（宽/高/FPS）。
- 虹膜与人脸关键点追踪（`core/FaceTracker`）
  - 使用 MediaPipe Face Mesh，推理分辨率按宽度限制（默认 640）。
  - 计算眼角点与双眼虹膜中心，支持轻量指数平滑减少抖动。
  - 叠加可视化：关键点、小圆、推断的人脸框。
- Flet UI 原型（`app/main.py`）
  - 基础 UI（设备/分辨率选择，开始/停止），帧展示（JPEG base64）。
  - 采集与推理线程、停止事件、队列限流（`maxsize=1`），避免积压。
 - 刺激播放与时间轴（最小骨架）：`assets/stimuli/<id>.mp4` + `<id>.json`；播放时按单调时钟触发片段/对话/呼名事件并写入会话日志。
  - 推荐安装 `flet-video` 扩展以避免 Flet 内置 `Video` 的弃用告警：`uv add flet-video`（已内置兼容：若未安装则回退使用内置 `ft.Video`）。

## 尚未实现/规划中

- 视线映射与校准（gaze calibration/mapping）。
- 更完善的多设备枚举与后端切换策略（跨平台差异适配）。
- 更细致的异常处理与 UI 提示（摄像头权限/占用/失败回退）。
- 自动化测试覆盖（合成帧的单测、UI 逻辑的最小回归）。
- Release Drafter/语义化版本流（可选），持续自动化发布流程。

## 刺激与时间轴（使用说明）

1. 放置视频与时间轴：
   - 视频：`assets/stimuli/<id>.mp4`
   - 时间轴：`assets/stimuli/<id>.json`（可参考示例 `assets/stimuli/sample.json`）
2. 运行应用，选择刺激 ID，点击“播放刺激”。
3. 事件会记录到会话日志（`sessions/session_*.json`）。

时间轴 JSON 字段约定：
```
{
  "id": "bubbles",
  "segments": [ { "name": "salient_bubbles", "start_ms": 32500, "end_ms": 34500, "tags": ["salient", "social"] } ],
  "dialogue_turns": [ { "speaker": "left", "start_ms": 12000, "end_ms": 14000 } ],
  "name_calls": [ { "t_ms": 52000, "label": "name_call_1" } ],
  "sides": [ { "t_ms": 0, "person_side": "left", "toy_side": "right" } ]
}
```

## 目录结构（简要）

- `src/app/` Flet UI 与入口（`main.py`）。
- `src/core/` 追踪算法（人脸/虹膜、平滑与可视化）。
- `src/services/` 设备封装（摄像头）。
- `assets/` 模型与静态资源（如需）。
- `.github/workflows/` CI 工作流（发布与分支维护：`release_ops.yml`）。

## 本地开发

1. 安装依赖（基于 `pyproject.toml`）：
   - `uv sync`
2. 运行原型（任选其一）：
   - `uv run -m app.main`
   - `uv run eye-track`（先执行一次 `uv pip install -e .`）
3. 常用命令：
   - 代码检查：`uv run ruff check .`、格式化：`uv run black .`
   - 测试（若补充测试）：`uv run pytest -q`

## 发布与分支

- 默认分支：`master`。
- 长期开发分支：`develop`。
- 手动发布/创建 Release：Actions → `Release Ops`（支持 tag 存在则发布、缺失则创建）。

## 贡献指南（简要）

- 代码风格：Python 3.11+，`black` + `ruff`，中文注释与文档。
- 提交信息：建议使用 Conventional Commits（如 `feat(core): 新增虹膜平滑`）。
- 变更请附必要说明、验证方式与可能影响范围。

## 许可证

本仓库为原型项目，许可证与使用范围请参见仓库设置或后续补充说明。

