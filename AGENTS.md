# Repository Guidelines

## 项目结构与模块组织

- `src/app/` Flet UI 与入口（示例：`main.py`）。
- `src/core/` 视觉/追踪算法（人脸检测、特征点、校准与映射）。
- `src/services/` 设备接入与配置（摄像头、日志、存储）。
- `assets/models/` 模型与权重；`assets/icons/` 静态资源。
- `tests/` 单元/集成测试；`scripts/` 开发/数据脚本。
- `papers/` 研究论文与参考资料；`docs/` 设计与使用文档。

## 构建、测试与本地开发命令

- `uv sync` 同步安装依赖（基于 `pyproject.toml`）。
- 启动原型（三选一，推荐前两种）：
  - `uv run -m app.main`
  - `uv run eye-track`（先执行一次 `uv pip install -e .`）
  - 若遇 Flet 自动安装依赖报错（pip 不存在），先执行：`uv sync`（已固定 `flet[all]==0.28.3`）或手动：`uv pip install 'flet[all]==0.28.3' -U`
- `uv run pytest -q` 运行测试；`uv run coverage run -m pytest` 统计覆盖率。
- `uv add flet opencv-python mediapipe` 添加依赖示例；`uv remove <pkg>` 移除。
- `uv run ruff check . && uv run black --check .` 静态检查与格式校验；`uv run black .` 一键格式化。

## 代码风格与命名约定

- Python 3.11+；缩进 4 空格；使用 `black` 格式化、`ruff` 作为 linter，`isort` 排序导入。
- 命名：模块/文件与函数/变量使用 `snake_case`；类使用 `PascalCase`；常量 `UPPER_SNAKE_CASE`。
- 公共 API 与复杂函数需类型注解与中文 docstring；避免循环依赖，优先 `src/` 布局的绝对导入。

## 测试规范

- 框架：`pytest`；目标覆盖率 ≥ 80%。
- 位置与命名：`tests/test_*.py`；同名模块一一对应。
- 摄像头相关测试在有设备环境下执行；后续根据需要再补充自动化方案。

## 提交与 Pull Request 指南

- 建议使用 Conventional Commits：`feat(ui): 添加主界面`、`fix(tracking): 修复瞳孔中心抖动`。
- PR 应包含：问题背景/变更说明、验证步骤、截图或录屏、关联 issue、影响范围与回滚方案。
- 需通过 lint/测试后请求合并；涉及配置/接口变更请同步更新文档与示例配置。

## Git 版本控制（关键流程）

- 分支命名：`feature/<name>`、`fix/<name>`、`chore/<name>`、`docs/<name>`。
- 提交规范：遵循 Conventional Commits，单次提交聚焦一个小改动；中文信息简洁明确。
- 基本命令：
  - 初始化：`git init`，首次提交：`git add -A && git commit -m "feat(app): 初版摄像头预览"`
  - 新功能分支：`git switch -c feature/face-tracking`
  - 同步主线：`git pull --rebase`（避免无意义 merge）
  - 推送：`git push -u origin feature/face-tracking`
- 提交频率：每完成一个可运行的子步骤即提交；跨文件重构先小步提交再整体校验。

## 安全与配置提示（可选）

- 不提交含人脸/个人信息的数据；录屏与样例加入 `.gitignore`。
- 模型与大文件放置 `assets/models/`，优先使用下载脚本获取；避免提交训练数据。
- 首次运行需授予摄像头权限；无法访问时提供模拟输入或关闭相关特性。

## 架构速览（参考）

- 管线：采集 → 检测/特征点 → 眼动映射/校准 → UI 交互。
- 分层：`core` 保持纯算法、可独立单测；`app` 专注展示与事件，`services` 提供设备与配置抽象。

## MCP 与 Agent 使用规范

- 工具选择：优先离线；按意图选择——规划 →Sequential Thinking，官方文档 →Context7（需库 ID），最新网页 →Tavily Search，代码检索/符号级编辑 →Serena（LSP）。
- 单轮单工具：每轮仅使用一种外部服务；确需多种时串行，并在 PR 描述中说明理由与产出。
- 最小必要：限制 tokens/结果数/时间窗与代码范围，避免过度抓取与噪声。
- 可靠与合规：遵守 robots/ToS；不上传敏感数据；遇 429 退避 20s 并降低范围，必要时降级替代。
- 结果追溯：在提交描述/评论附“工具调用简报”，示例：
  ```
  工具: Context7 (/vercel/next.js v14)
  触发: 路由配置查证
  输入: topic=routing, tokens=2000
  时间: 2025-10-01T12:00Z
  来源: 官方文档
  重试: 无
  ```
- Serena 常用操作：`find_symbol`、`find_referencing_symbols`、`insert_before_symbol`、`replace_symbol_body`；变更需标注文件与符号路径，最小化改动并附理由。
- Context7 专用流程：先 `resolve-library-id` 获取精确库 ID，再 `get-library-docs`；提供 `topic`（如 hooks、routing）并控制 `tokens≤5000`；多库匹配优先高信任与覆盖；歧义时请求澄清或在备注说明选择理由。
- Tavily Search 约束：建议 ≤12 关键词；优先使用 `include_domains`/`exclude_domains` 限定权威域，必要时在查询中使用 `site:`/`filetype:`。参数建议：`max_results≤20`、`search_depth=basic`（必要时 `advanced`）、`timeout≤5s`。优先官方/权威来源，域名去重并剔除内容农场；遇 429 退避 20s 并缩小范围或时间窗。
- 失败与降级：首选服务失败 → 尝试替代；网络受限 → 提供离线保守答案并标注不确定性；如需外呼权限，先征得授权（含用途与最小范围）。

Hard Requirement: call binaries directly in functions.shell, always set workdir, and avoid shell wrappers such as `bash -lc`, `sh -lc`, `zsh -lc`, `cmd /c`, `pwsh.exe -NoLogo -NoProfile -Command`, and `powershell.exe -NoLogo -NoProfile -Command`.

- Text Editing Priority: Use the `apply_patch` tool for all routine text edits; fall back to `sed` for single-line substitutions only if `apply_patch` is unavailable, and avoid `python` editing scripts unless both options fail.
- `apply_patch` Usage: Invoke `apply_patch` with the patch payload as the second element in the command array (no shell-style flags). Provide `workdir` and, when helpful, a short `justification` alongside the command.

- Example invocation:

```bash
{"command":["apply_patch","*** Begin Patch\n*** Update File: path/to/file\n@@\n- old\n+ new\n*** End Patch\n"],"workdir":"<workdir>","justification":"Brief reason for the change"}
```
