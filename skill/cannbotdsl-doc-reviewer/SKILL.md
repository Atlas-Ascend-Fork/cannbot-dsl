---
name: cannbotdsl-doc-reviewer
description: CANNBot-DSL 文档（docs/目录）质量检查，支持两种模式：开发者提交 PR 前自检、Committer 检视文档 PR。覆盖开源写作规范（复用 CANN 社区规范）、API 文档写作规范两个维度。触发词：自检文档、文档自检、检查文档、检视文档、文档检视、review文档、检视PR、文档PR、docs检视。
---

# CANNBot-DSL Doc Reviewer

对 `docs/` 目录下的 Markdown 文档变更进行规范化检查。一份规则、两个入口：

| 模式 | 使用者 | 怎么触发 | 得到什么 |
|------|--------|---------|---------|
| 自检 | 开发者（提交 PR 前） | 给文件/目录路径（如 `自检文档 docs/zh/api/kernel/resource-management/vec-sync-block-arrive.md`），或不给参数（检查未提交变更）；粒度可指定"变更点/全量"，不指定时默认：给路径=全量，不给路径=变更点 | 修复建议报告 |
| 检视 | Committer | 给 PR 编号（如 `检视PR 123`） | 行级检视意见报告 |

**模式判定**：输入含 PR 编号 → 工作流 B；否则 → 工作流 A。

**执行方式**：两工作流的全部阶段（git 操作、规则加载、逐维度检查、报告生成）均由 agent 自动执行，无需人工逐步操作；用户仅在三处介入：① 触发时提供输入（路径 / PR 编号 / 检查粒度）；② 工作流 B 阶段 5 提交评论前确认；③ 首次提交评论前完成 Token 配置。

## 检查维度

共 2 个维度。维度 1 复用 CANN 社区外部规范（含产品型号与芯片标签、术语及表达约定要求），其余规则在 `references/` 下；每个规则文件均分「规则」与「检查方法」（bash 命令与核对要点）两部分：

| # | 维度 | 何时必查 | 规则来源 |
|---|------|---------|----------|
| 1 | 开源写作规范 | 所有 `.md` | 外部（webfetch 读取，配套使用）：[开源文档写作规范](https://gitcode.com/cann/docs/blob/master/specification/document_writing_specs.md)（事实来源）及配套检视规则 [md-review-skill](https://gitcode.com/cann/docs/blob/master/skills/md-review-skill/SKILL.md) |
| 2 | API 文档写作规范 | `docs/zh/api/` 下文件，或该目录 `.md` 文件新增/删除/移动/重命名时 | [rules-api.md](references/rules-api.md) |

## 检查执行步骤（两模式通用）

对检查范围内每个 `.md` 文件：

1. **选维度**：按上表「何时必查」确定该文件命中的维度
2. **读规则**：读取命中维度的规则文件（维度 1 用 webfetch 读取开源文档写作规范及配套的 md-review-skill 检视规则）
3. **读文件**：新文件读全文；修改文件按检查粒度——变更点检查读 diff 及所在章节上下文，全量检查读全文
4. **逐条核对**：按「规则」逐项检查，并执行文件末尾「检查方法」中的排查命令。通用排版排查（维度 1，对所有 `.md` 执行）：`grep -nP '[\x{4e00}-\x{9fa5}] [A-Za-z0-9\x{4e00}-\x{9fa5}]|[A-Za-z0-9] [\x{4e00}-\x{9fa5}]' <目标文件>`——中文与英文之间、中文与中文之间均不加空格
5. **记录**：命中问题按「问题输出格式」记为 findings，定级 CRITICAL/HIGH/MEDIUM/LOW。排查命令为文件级扫描，报告按检查粒度过滤——变更点检查（检视模式同）仅报变更行及其上下文内的问题，全量检查报全部命中

---

## 工作流 A：自检模式（开发者）

### 阶段 1：确定检查范围与粒度

**检查对象**：

1. 给了文件/目录路径 → 路径下所有 `.md`（目录递归）——默认**全量**（点名文件即整档检查，且该文件不一定有未提交变更）
2. 未给路径 → `git status --porcelain -- docs/` 中的未提交变更（含未跟踪新文件）——默认**变更点**（自查手头改动）；无变更则报告后结束

**检查粒度**（两种均支持，可用"变更点"/"全量"显式指定）：

- **变更点**：仅检查变更行及其所在章节上下文（新文件查全文），适合提交前自查改动
- **全量**：检查文件全部内容（含存量问题），适合整档治理
- 默认（与检查对象一一对应，见阶段 1）：**给路径 = 全量，不给路径 = 变更点**

### 阶段 2：逐维度检查

按「检查执行步骤」执行。

### 阶段 3：输出修复建议报告

问题按严重级别排序（CRITICAL → LOW），每条给出：位置、规则编号、问题描述、修复建议。报告骨架：

```markdown
## 文档自检报告: <文件/目录，或"未提交变更">（粒度：变更点/全量）

### 概览
| 维度 | CRITICAL | HIGH | MEDIUM | LOW |
|------|----------|------|--------|-----|

### 问题列表
**[1]** [CRITICAL] [维度/规则] 文件:行号
- 问题：...
- 建议：...

### 结论
各维度问题数统计；CRITICAL/HIGH 清零后才建议提交 PR
```

## 工作流 B：检视模式（Committer）

### 阶段 1：确认目标分支并切换到 PR 分支

**必须先确认 MR 的目标分支（base），再同步该分支并切换。目标分支不一定是 master，也可能是 9.2.0 等发布分支；基准选错（如用了落后的本地 master）会把已合入其他分支的提交误算进 PR 范围。** 确认方法按优先级：

1. 查 MR 信息：`GET /repos/cann/cannbot-dsl/pulls/<pr_number>` 返回的 base 字段（复用 md-review-skill 的 api_client.py），或直接询问用户
2. 本地启发式（无法查 API 时）：先 fetch MR head，再对候选分支（`git ls-remote --heads upstream` 列出的 master 与各发布分支）逐一执行 `git merge-base pr_<pr_number> remotes/upstream/<分支>`——正确目标分支的 merge-base 紧贴该分支 tip，且 `git log remotes/upstream/<分支>..pr_<pr_number> --oneline` 只包含 PR 自身提交；两个候选都满足时须询问用户，不得默认 master

merge-requests ref 只存在于上游仓 `cann/cannbot-dsl`，本地 origin 若是 fork 则取不到，统一通过 upstream 远端获取：

```bash
git remote add upstream https://gitcode.com/cann/cannbot-dsl.git 2>/dev/null
git fetch upstream <目标分支>
git fetch upstream refs/merge-requests/<pr_number>/head:pr_<pr_number>
git checkout pr_<pr_number>
```

### 阶段 2：获取变更清单

```bash
# 三个点(...) + 基于 upstream/<目标分支>；全量 stat 用于发现 .py 签名变更（阶段 3 核对 API-13）
git diff remotes/upstream/<目标分支>...pr_<pr_number> --stat
git diff remotes/upstream/<目标分支>...pr_<pr_number> -p -- docs/
```

无 `.md` 变更则报告"无文档变更"并结束。

**范围 sanity check（必做）**：变更范围须与 PR 主题相称。若小型修改类 PR 却出现大量提交/文件（如文档澄清却改了几百个文件），通常是目标分支判断错误（本地分支落后、误用本地分支或 fork origin 做基准），须回到阶段 1 复核后再继续，不得直接开检。

### 阶段 3：逐维度检查

按「检查执行步骤」对 PR 变更的 `.md` 文件执行。检视模式额外注意：
- **只检视 PR 变更引入的问题**，目标分支上已存在的问题不重复报告；合规性以规则文件为准，不得以存量文档惯例作为豁免依据
- 阶段 2 的全量变更清单中出现 `.py` 签名变更时，必须核对 API 文档原型是否同步更新（API-13）

### 阶段 4：输出检视报告

按「问题输出格式」输出行级意见，格式：

```markdown
## 文档检视报告: PR #<N> - [PR标题]

### 概览
| 维度 | CRITICAL | HIGH | MEDIUM | LOW |
|------|----------|------|--------|-----|

### 问题列表
**[1]** [CRITICAL] [API文档写作规范] docs/zh/api/kernel/resource-management/vec-sync-block-arrive.md:45
- 问题：...
- 建议：...
```

### 阶段 5：评论提交（可选，需用户确认）

生成报告后询问用户需提交哪些意见。**严禁未经确认自动提交。**

**Token 配置（提交评论的前提）**：提交评论须调用 GitCode API 做鉴权，Token 存于本地配置文件，由 `api_client.py` 读取，全程不回显、不打印：

- 默认路径：`~/.config/opencode/skills/md-review-skill/config/gitcode-config.json`（或经 `--config` 参数指定自定义路径）
- 文件格式：`{"token": "<GitCode 私人令牌>", "baseUrl": "https://api.gitcode.com/api/v5"}`
- **配置缺失时停止提交**，指引用户自行创建该文件（令牌在 GitCode「设置 → 私人令牌」生成，需具备 PR 评论写权限），待用户完成后再继续

**提交方法**：复用 md-review-skill 的 `api_client.py`（实际位置用 `find / -name 'api_client.py' -path '*md-review*' 2>/dev/null` 定位）：

```bash
python3 <api_client.py路径> POST /repos/cann/cannbot-dsl/pulls/<pr_number>/comments --config <配置路径> \
  --data '{"body": "[级别] [维度/规则] 文件:行号\n问题：...\n建议：...", "path": "docs/zh/api/kernel/resource-management/vec-sync-block-arrive.md", "position": 42, "side": "RIGHT"}'
```

- POST 字段为扁平的 `path`/`position`/`side`，勿写嵌套 position 对象（会被 400 拒绝）；`position` 为 PR head 侧**绝对行号**
- **`path` 必须是 PR 变更文件列表中的文件**：问题涉及不在 diff 中的文件时（如漏登记 `docs/zh/api/README.md`），评论挂到相关变更文件上（如新增文档本身）

**流程：先 smoke test 再批量**：

1. 先 POST 1 条，取响应中数字型 `id`（即 note_id；`discussion_id` 为 SHA，不能用于 API 路径）
2. `GET /repos/cann/cannbot-dsl/pulls/comments/<note_id>` 验证 `position.new_line` 与目标行号一致
3. 错位则 `DELETE /repos/cann/cannbot-dsl/pulls/comments/<note_id>` 撤回（返回 204 即成功），修正字段后重发
4. 验证通过后批量提交剩余意见，最后 GET 评论列表复核

---

## 问题输出格式

每条问题统一使用：

```json
{
  "severity": "CRITICAL|HIGH|MEDIUM|LOW",
  "file": "docs/zh/api/kernel/resource-management/vec-sync-block-arrive.md",
  "line": 42,
  "rule_id": "API-13",
  "dimension": "API文档写作规范",
  "title": "函数原型与代码实现不一致",
  "body": "问题描述与修复建议",
  "code_snippet": "该行内容片段"
}
```

严重级别定义：

| 级别 | 含义 | 示例 |
|------|------|------|
| CRITICAL | 导致构建失败或信息错误误导用户 | README 索引漏登记导致文档不可见；原型与代码不一致 |
| HIGH | 违反强制规范 | API 文档章节缺失或顺序颠倒；产品名称与模板不一致 |
| MEDIUM | 违反建议性规范 | 数据结构类文档固定部分顺序改变；示例缺注释 |
| LOW | 体验优化 | 标点、空行等 |
