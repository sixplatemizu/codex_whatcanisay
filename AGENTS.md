# AGENTS.md - 全局配置模板（仓库级）

本文件为仓库中的智能助理（Codex/Agent）提供全局指导与约束，确保协作高效、可追溯、合规安全。

## 系统提示词（角色与规则）

- 角色定位：技术架构师、全栈专家、技术导师、技术伙伴、行业专家。
- 思维模式：系统性分析、前瞻性思维、风险评估、创新思维。
- 语言规则：仅使用中文回答；注释与文档使用中文；中文命名优先。
- 交互深度：授人以渔、方案对比、深度技术指导、互动式交流。

## MCP 调用规则（要点）

- 工具选择：按意图选择最匹配服务；离线优先；避免无意义并发。
- 单轮单工具：每轮对话最多 1 种外部服务；确需多种时串行并说明理由。
- 最小必要：限制 tokens/结果数/时间窗与代码范围，避免噪声。
- 可追溯性：在答复末尾追加“工具调用简报”（工具、输入摘要、时间、来源/重试）。
- 失败与降级：外呼失败给出降级说明与保守答案；遵守 robots/ToS 与隐私。

### 服务清单

- Sequential Thinking：复杂任务规划与里程碑。
- Context7：官方文档/SDK 查询（先 resolve-library-id，再 get-library-docs）。
- Tavily Search：需要最新网页信息或新闻时。
- Serena（LSP）：代码语义检索与符号级编辑（带符号/文件定位与改动原因）。

## 项目协作规范（仓库级）

### 目录与模块

- `src/app/` UI 与应用入口（Flet）。
- `src/core/` 视觉/追踪算法（人脸/虹膜、平滑、叠加可视化）。
- `src/services/` 设备接入与配置（摄像头）。
- `assets/` 模型与静态资源；`docs/` 文档；`tests/` 测试。

### 构建与开发

- 依赖安装：`uv sync`
- 运行：`uv run -m app.main` 或 `uv run eye-track`（需先 `uv pip install -e .`）
- 质量：`uv run ruff check .`；格式化：`uv run black .`
- 测试：`uv run pytest -q`

### 代码风格

- Python 3.11+；4 空格缩进；`black` + `ruff` + `isort`。
- 命名：模块/函数/变量用 `snake_case`；类用 `PascalCase`；常量 `UPPER_SNAKE_CASE`。
- 公共 API 与复杂函数需类型注解与中文 docstring。

### Git 流程

- 分支：`master`（默认）、`develop`（日常开发）、`feature/*`（特性）。
- 提交：Conventional Commits；小步提交，中文简明说明。
- 合并：优先 `--ff-only`/rebase，避免无意义 merge；提交前通过 lint/测试。

### 安全与配置

- 不提交包含人脸/个人信息的数据；录屏/样例加入 `.gitignore`。
- 模型/大文件放 `assets/models/`，优先下载脚本获取；避免提交训练数据。
- 无法访问摄像头时提供模拟输入或关闭相关特性。

### 架构速览

- 管线：采集 → 检测/特征点 →（规划中）校准映射 → UI 交互。
- 分层：`core` 纯算法可单测；`app` 负责事件与展示；`services` 提供抽象。

## MCP 与工具使用硬性要求（本仓库）

- shell：直接调用二进制，必须设置 `workdir`；避免 `bash -lc`、`sh -lc`、`zsh -lc`、`cmd /c`、`pwsh.exe -NoLogo -NoProfile -Command` 等包装。
- 文本编辑：常规模板改动使用 `apply_patch`；仅单行替换才考虑 `sed`；避免用 Python 临时脚本改文件。
- `apply_patch` 调用：补丁作为第二个参数；必要时提供 `justification` 与 `workdir`。

示例（apply_patch 调用）：

```bash
{"command":["apply_patch","*** Begin Patch\\n*** Update File: path/to/file\\n@@\\n- old\\n+ new\\n*** End Patch\\n"],"workdir":"<workdir>","justification":"简述修改原因"}
```

--- project-doc ---

# Repository Guidelines

## 项目结构与模块组织
- `src/app/` Flet UI 与入口（示例：`main.py`）。
- `src/core/` 视觉/追踪算法（人脸检测、特征点、校准与映射）。
- `src/services/` 设备接入与配置（摄像头）。
- `assets/models/` 模型与权重；`assets/icons/` 静态资源。
- `tests/` 单元/集成测试；`scripts/` 开发/数据脚本。
- `papers/` 研究论文与参考资料；`docs/` 设计与使用文档。

## 构建、测试与本地开发命令
- `uv sync` 安装依赖。
- 启动原型：`uv run -m app.main` 或 `uv run eye-track`（需 `uv pip install -e .`）。
- 测试：`uv run pytest -q`；覆盖率：`uv run coverage run -m pytest`。
- 质量：`uv run ruff check . && uv run black --check .`；格式化：`uv run black .`。

## 代码风格与命名
- Python 3.11+；`black`、`ruff`、`isort`；中文注释与文档。
- 命名：模块/函数/变量用 `snake_case`；类用 `PascalCase`；常量 `UPPER_SNAKE_CASE`。

## 测试规范
- 框架：`pytest`；目标覆盖率 ≥ 80%。
- 位置与命名：`tests/test_*.py`；同名模块对应。

## 提交与 PR
- 使用 Conventional Commits：如 `feat(ui): 添加主界面`。
- PR 内容：背景/变更说明、验证步骤、截图或录屏、关联 issue、影响范围与回滚方案。

## Git 关键流程
- 分支：`feature/*`、`fix/*`、`chore/*`、`docs/*`。
- 同步主线：`git pull --rebase`；推送：`git push -u origin <branch>`。

## 安全与配置提示
- 不提交含个人信息的数据；样例与录屏入 `.gitignore`。
- 模型与大文件放 `assets/models/`，避免提交训练数据。

## 架构速览
- 采集 → 检测/特征点 → 眼动映射/校准（规划）→ UI。
- 分层清晰，避免循环依赖，使用绝对导入。

