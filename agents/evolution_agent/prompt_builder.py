"""
prompt_builder.py

负责构造 Hermes Agent 的系统 Prompt，包括：

1. Hermes 身份和行为原则
2. 工具调用规则
3. 可用工具索引
4. Skill 索引与已加载 Skill 规则
5. 项目规则
6. 运行时环境信息

本模块不负责：
- 加载历史消息
- 加载 Memory
- 添加用户消息
- 调用模型
- 执行工具

这些职责分别由 turn_context.py、conversation_loop.py、
tool_executor.py 等模块承担。
"""

from __future__ import annotations

import json
import platform
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

# prompt_builder.py:
# 项目根目录/agents/evolution_agent/prompt_builder.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# 数据结构
# ============================================================


@dataclass(frozen=True)
class SkillInfo:
    """Skill 元信息和正文内容。"""

    name: str
    description: str
    path: Path
    content: str = ""


@dataclass
class PromptBuilderConfig:
    """PromptBuilder 配置。"""

    project_root: Path
    agent_name: str = "Hermes"
    skill_directories: list[Path] = field(default_factory=list)

    project_rule_files: tuple[str, ...] = (
        "AGENTS.md",
        "HERMES.md",
        "PROJECT_RULES.md",
        ".hermes/rules.md",
        ".hermes/project.md",
    )

    max_skill_description_chars: int = 500
    max_skill_content_chars: int = 20_000
    max_rule_file_chars: int = 20_000

    # hybrid-retrieval 的权重、top_k 和降级规则位于 Skill 正文中。
    # 因此默认将 Skill 正文直接注入 system prompt，而不是只展示索引。
    include_skill_content: bool = True
    include_runtime_info: bool = True

    def __post_init__(self) -> None:
        self.project_root = self.project_root.resolve()
        self.skill_directories = [
            directory.resolve()
            for directory in self.skill_directories
        ]


@dataclass
class PromptBundle:
    """Prompt 构造结果。"""

    system_prompt: str
    tools: list[dict[str, Any]]
    skills: list[SkillInfo]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_system_message(self) -> dict[str, str]:
        """转换为模型使用的 system message。"""

        return {
            "role": "system",
            "content": self.system_prompt,
        }


# ============================================================
# PromptBuilder
# ============================================================


class PromptBuilder:
    """Hermes Agent 系统 Prompt 构造器。"""

    # 同时兼容标准 YAML Front Matter：
    # ---
    # name: xxx
    # description: xxx
    # ---
    #
    # 以及用户当前使用的长横线结束符：
    # ---
    # name: hybrid-retrieval
    # description: xxx
    # --------------------------------------------------------
    _FRONT_MATTER_PATTERN = re.compile(
        r"^\ufeff?\s*-{3,}[ \t]*\r?\n"
        r"(?P<front_matter>.*?)"
        r"\r?\n-{3,}[ \t]*(?:\r?\n|$)",
        flags=re.DOTALL,
    )

    def __init__(self, config: PromptBuilderConfig) -> None:
        self.config = config

    def build(
        self,
        tools: Sequence[Mapping[str, Any] | Any] | None = None,
        extra_project_rules: str | None = None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> PromptBundle:
        """构造完整 PromptBundle。"""

        normalized_tools = self.normalize_tools(tools or [])
        skills = self.discover_skills()
        project_rules = self.load_project_rules()

        if extra_project_rules and extra_project_rules.strip():
            project_rules.append(
                (
                    "运行时附加规则",
                    extra_project_rules.strip(),
                )
            )

        sections = [
            self._build_identity_section(),
            self._build_instruction_priority_section(),
            self._build_workflow_section(),
            self._build_tool_rules_section(),
            self._build_tool_index_section(normalized_tools),
            self._build_skill_rules_section(),
            self._build_skill_index_section(skills),
            self._build_loaded_skill_rules_section(skills),
            self._build_project_rules_section(project_rules),
        ]

        if self.config.include_runtime_info:
            sections.append(
                self._build_runtime_section(runtime_context or {})
            )

        sections.append(self._build_final_response_section())

        system_prompt = "\n\n".join(
            section.strip()
            for section in sections
            if section and section.strip()
        )

        return PromptBundle(
            system_prompt=system_prompt,
            tools=normalized_tools,
            skills=skills,
            metadata={
                "agent_name": self.config.agent_name,
                "project_root": str(self.config.project_root),
                "skill_count": len(skills),
                "tool_count": len(normalized_tools),
                "project_rule_count": len(project_rules),
                "loaded_skill_content": self.config.include_skill_content,
                "created_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
            },
        )

    # ========================================================
    # Hermes 身份
    # ========================================================

    def _build_identity_section(self) -> str:
        return f"""
        # Agent Identity

        你是 {self.config.agent_name}，一个能够读取项目、修改代码、
        运行测试并根据测试结果持续修复问题的软件工程 Agent。

        你的核心目标是：

        1. 准确理解用户任务。
        2. 基于项目中的真实代码和规则进行判断。
        3. 使用工具收集证据，而不是猜测项目状态。
        4. 实施范围尽可能小但完整正确的代码修改。
        5. 使用测试验证修改结果。
        6. 向用户清楚说明修改内容、验证结果和剩余风险。

        你必须区分以下三种信息：

        - 已确认事实：来自用户消息、文件内容或工具结果。
        - 合理推断：根据已有证据得到，但尚未完全验证。
        - 未知信息：当前上下文无法确认的内容。

        不得将推断或未知信息描述为已经确认的事实。
        """

    def _build_instruction_priority_section(self) -> str:
        return """
        # Instruction Priority

        发生规则冲突时，按照以下优先级处理：

        1. 系统安全规则和平台限制
        2. 用户当前明确提出的任务
        3. 当前项目的项目规则
        4. 已加载 Skill 中的操作规则
        5. Hermes 默认工作规则

        项目规则只对当前项目生效。

        Skill 是针对特定任务的操作指南。已经加载到 Prompt 的 Skill 正文，
        必须在对应任务中执行；不得只读取名称和描述后忽略正文参数。
        """

    # ========================================================
    # 工作流程
    # ========================================================

    def _build_workflow_section(self) -> str:
        return """
        # Default Engineering Workflow

        处理代码任务时，默认遵循以下流程：

        1. 理解用户目标和验收条件。
        2. 查看与任务直接相关的文件。
        3. 搜索相关函数、类、调用链和测试。
        4. 判断是否存在适用的 Skill。
        5. 读取并遵循已加载 Skill 的完整规则。
        6. 制定最小修改方案。
        7. 修改代码。
        8. 运行针对性测试。
        9. 根据报错分析根因。
        10. 必要时继续读取、修改和测试。
        11. 测试通过后生成最终说明。

        对于 bug 修复：

        - 优先定位根因，不只处理表面异常。
        - 不得通过删除测试、跳过断言或吞掉异常来制造测试通过。
        - 不得无依据扩大修改范围。
        - 不得在未读取相关实现前直接重写整个模块。

        对于已有代码：

        - 保持原有架构和命名风格。
        - 优先复用现有函数和抽象。
        - 避免无关重构。
        - 避免修改与当前任务无关的公共接口。
        """

    # ========================================================
    # 工具规则
    # ========================================================

    def _build_tool_rules_section(self) -> str:
        return """
        # Tool Usage Rules

        工具返回结果是判断项目实际状态的重要证据。

        必须遵循以下规则：

        1. 不得伪造文件内容、测试结果、命令输出或工具执行结果。
        2. 在修改文件前，先读取相关文件和调用上下文。
        3. 不确定符号位置时，先搜索，再读取。
        4. 修改代码后，应运行与修改内容直接相关的测试。
        5. 优先运行最小范围测试，再根据需要扩大测试范围。
        6. 测试失败时，分析新的报错，不要机械重复相同命令。
        7. 一次工具调用应有明确目的。
        8. 避免重复读取已经完整获得且没有发生变化的文件。
        9. 工具报错时，应判断是参数问题、环境问题还是代码问题。
        10. 不得声称测试通过，除非工具结果明确显示测试成功。
        11. 不得声称文件已修改，除非修改工具明确执行成功。
        12. 不得通过修改测试期望来掩盖生产代码问题，除非用户明确要求修改测试。
        13. 遇到破坏性操作时，应缩小影响范围并保留用户已有工作。
        14. 不得擅自删除用户文件、覆盖无关修改或重置整个仓库。

        pytest 使用规则：

        - 优先运行与修改模块对应的测试文件。
        - 可以使用具体测试节点，例如：

        pytest tests/test_example.py::test_specific_case -q

        - 局部测试通过后，再根据改动风险决定是否运行完整测试。
        - 如果完整测试耗时过高，应至少运行相关测试，并在最终结果中说明验证范围。
        """

    def _build_tool_index_section(
        self,
        tools: Sequence[Mapping[str, Any]],
    ) -> str:
        if not tools:
            return """
            # Available Tools

            当前没有向模型提供可调用工具。

            不要假设自己能够读取文件、修改文件或执行命令。
            """

        lines = ["# Available Tools", ""]

        for index, tool in enumerate(tools, start=1):
            function = tool.get("function", {})
            name = function.get("name", "unknown_tool")
            description = function.get(
                "description",
                "没有提供工具说明。",
            )

            lines.append(
                f"{index}. `{name}`\n"
                f"   - {str(description).strip()}"
            )

        lines.extend(
            [
                "",
                "工具参数结构由模型调用接口单独提供。",
                "选择工具时应依据工具名称、说明和当前任务需要。",
            ]
        )

        return "\n".join(lines)

    # ========================================================
    # Skill
    # ========================================================

    def _build_skill_rules_section(self) -> str:
        return """
        # Skill Usage Rules

        Skill 是存储在 SKILL.md 中的专项操作指南。

        使用规则：

        1. 查看 Skill 索引，判断是否存在与当前任务直接相关的 Skill。
        2. 当 Prompt 中已经提供 Skill 正文时，直接按正文执行。
        3. Skill 中声明为“必须”的规则属于强约束。
        4. Skill 中的权重、top_k、排序、归一化和降级参数不得自行改写。
        5. Skill 与项目规则冲突时，以项目规则为准。
        6. Skill 与用户当前明确要求冲突时，以用户当前要求为准，
           但不得违反系统安全限制。
        7. Skill 描述的组件、索引、字段和工具仍需通过项目代码确认是否存在。
        8. 不得在没有失败证据时主动触发 Skill 中的降级策略。
        """

    def _build_skill_index_section(
        self,
        skills: Sequence[SkillInfo],
    ) -> str:
        if not skills:
            return """
            # Skill Index

            当前项目未发现可用 Skill。
            """

        lines = ["# Skill Index", ""]

        for index, skill in enumerate(skills, start=1):
            relative_path = self._display_path(skill.path)
            load_status = (
                "正文已加载"
                if self.config.include_skill_content and skill.content
                else "仅加载索引"
            )

            lines.append(
                f"{index}. `{skill.name}`\n"
                f"   - 描述：{skill.description}\n"
                f"   - 路径：`{relative_path}`\n"
                f"   - 状态：{load_status}"
            )

        return "\n".join(lines)

    def _build_loaded_skill_rules_section(
        self,
        skills: Sequence[SkillInfo],
    ) -> str:
        """将 Skill 正文作为独立 Markdown 章节注入 Prompt。"""

        if not self.config.include_skill_content:
            return ""

        loaded_skills = [skill for skill in skills if skill.content]

        if not loaded_skills:
            return ""

        lines = [
            "# Loaded Skill Rules",
            "",
            "以下内容是已加载 Skill 的完整执行规则。",
            "任务与某个 Skill 相关时，必须遵循对应正文。",
        ]

        for skill in loaded_skills:
            lines.extend(
                [
                    "",
                    f"## Skill: `{skill.name}`",
                    "",
                    f"来源：`{self._display_path(skill.path)}`",
                    "",
                    skill.content,
                ]
            )

        return "\n".join(lines)

    def discover_skills(self) -> list[SkillInfo]:
        """扫描配置目录中的 SKILL.md。"""

        skill_files: set[Path] = set()

        for skill_directory in self.config.skill_directories:
            if not skill_directory.exists():
                continue

            if skill_directory.is_file():
                if skill_directory.name.lower() == "skill.md":
                    skill_files.add(skill_directory.resolve())
                continue

            direct_skill_file = skill_directory / "SKILL.md"

            if direct_skill_file.is_file():
                skill_files.add(direct_skill_file.resolve())

            for candidate in skill_directory.rglob("*"):
                if (
                    candidate.is_file()
                    and candidate.name.lower() == "skill.md"
                ):
                    skill_files.add(candidate.resolve())

        skills: list[SkillInfo] = []

        for skill_file in sorted(
            skill_files,
            key=lambda item: str(item).lower(),
        ):
            try:
                raw_content = skill_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            front_matter, body = self._split_skill_document(raw_content)
            name, description = self._parse_skill_metadata(
                skill_file=skill_file,
                front_matter=front_matter,
                body=body,
            )

            skill_content = ""

            if self.config.include_skill_content:
                skill_content = self._truncate_text(
                    body.strip(),
                    self.config.max_skill_content_chars,
                )

            skills.append(
                SkillInfo(
                    name=name,
                    description=description,
                    path=skill_file,
                    content=skill_content,
                )
            )

        return skills

    @classmethod
    def _split_skill_document(
        cls,
        content: str,
    ) -> tuple[str, str]:
        """
        分离 Skill Front Matter 与正文。

        结束分隔符允许为任意不少于 3 个连续横线，因此可匹配：

        ---
        name: hybrid-retrieval
        description: ...
        ----------------------------------------------------------------
        """

        match = cls._FRONT_MATTER_PATTERN.match(content)

        if not match:
            return "", content.lstrip("\ufeff")

        front_matter = match.group("front_matter").strip()
        body = content[match.end():].lstrip("\r\n")
        return front_matter, body

    def _parse_skill_metadata(
        self,
        skill_file: Path,
        front_matter: str,
        body: str,
    ) -> tuple[str, str]:
        """从 Front Matter 或正文中提取 name 和 description。"""

        default_name = skill_file.parent.name
        parsed_name = self._extract_front_matter_value(
            front_matter,
            "name",
        )
        name = parsed_name or default_name

        description = self._extract_front_matter_value(
            front_matter,
            "description",
        )

        if not parsed_name:
            heading_match = re.search(
                r"^\s*#\s+(.+?)\s*$",
                body,
                flags=re.MULTILINE,
            )

            if heading_match:
                name = heading_match.group(1).strip()

        if not description:
            description = self._extract_first_paragraph(body)

        if not description:
            description = "未提供 Skill 描述。"

        description = re.sub(r"\s+", " ", description).strip()
        description = self._truncate_text(
            description,
            self.config.max_skill_description_chars,
        )

        return name, description

    @staticmethod
    def _extract_front_matter_value(
        front_matter: str,
        key: str,
    ) -> str:
        if not front_matter:
            return ""

        pattern = rf"^\s*{re.escape(key)}\s*:\s*(.*?)\s*$"
        match = re.search(
            pattern,
            front_matter,
            flags=re.MULTILINE | re.IGNORECASE,
        )

        if not match:
            return ""

        value = match.group(1).strip()

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        return value.strip()

    @staticmethod
    def _extract_first_paragraph(content: str) -> str:
        """提取第一个普通正文段落。"""

        paragraphs = re.split(r"\n\s*\n", content.strip())

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if not paragraph:
                continue
            if paragraph.startswith("#"):
                continue
            if paragraph.startswith("```"):
                continue
            if paragraph.startswith(("-", "*", ">", "|")):
                continue

            return paragraph

        return ""

    # ========================================================
    # 项目规则
    # ========================================================

    def load_project_rules(self) -> list[tuple[str, str]]:
        """加载项目规则文件。"""

        rules: list[tuple[str, str]] = []

        for relative_path in self.config.project_rule_files:
            rule_path = (
                self.config.project_root / relative_path
            ).resolve()

            if not self._is_path_inside_project(rule_path):
                continue
            if not rule_path.is_file():
                continue

            try:
                content = rule_path.read_text(
                    encoding="utf-8"
                ).strip()
            except (OSError, UnicodeDecodeError):
                continue

            if not content:
                continue

            content = self._truncate_text(
                content,
                self.config.max_rule_file_chars,
            )

            rules.append(
                (
                    self._display_path(rule_path),
                    content,
                )
            )

        return rules

    def _build_project_rules_section(
        self,
        project_rules: Sequence[tuple[str, str]],
    ) -> str:
        if not project_rules:
            return """
            # Project Rules

            当前项目未发现额外项目规则文件。

            仍需通过读取项目代码确认架构、风格和约束。
            """

        lines = [
            "# Project Rules",
            "",
            "以下规则仅适用于当前项目：",
        ]

        for source, content in project_rules:
            lines.extend(
                [
                    "",
                    f"## Rule Source: `{source}`",
                    "",
                    content,
                ]
            )

        return "\n".join(lines)

    # ========================================================
    # 运行环境
    # ========================================================

    def _build_runtime_section(
        self,
        runtime_context: Mapping[str, Any],
    ) -> str:
        runtime_data: dict[str, Any] = {
            "project_root": str(self.config.project_root),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "current_time": datetime.now().isoformat(
                timespec="seconds"
            ),
        }

        runtime_data.update(dict(runtime_context))

        rendered = json.dumps(
            runtime_data,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        return f"""
        # Runtime Context

        以下信息描述当前 Agent 运行环境：

        ```json
        {rendered}
        ```

        运行时信息仅用于理解当前执行环境，不得覆盖用户要求、项目规则或 Skill。
        """

    # ========================================================
    # 最终响应
    # ========================================================

    def _build_final_response_section(self) -> str:
        return """
        # Final Response Rules

        最终回复应包含：

        1. 实际完成的修改。
        2. 已执行的验证及结果。
        3. 未验证内容或剩余风险。

        不得声称执行了没有真实执行的命令、测试或文件修改。
        """

    # ========================================================
    # 工具标准化
    # ========================================================

    def normalize_tools(
        self,
        tools: Sequence[Mapping[str, Any] | Any],
    ) -> list[dict[str, Any]]:
        """统一转换为 OpenAI Function Calling 工具结构。"""

        normalized: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        for tool in tools:
            tool_mapping = self._tool_to_mapping(tool)

            if tool_mapping.get("type") == "function":
                function = tool_mapping.get("function")
                if not isinstance(function, Mapping):
                    raise ValueError("function 工具缺少 function 定义。")
                function_mapping = dict(function)
            else:
                function_mapping = tool_mapping

            name = str(function_mapping.get("name", "")).strip()
            if not name:
                raise ValueError("工具定义缺少 name。")

            if name in seen_names:
                continue

            description = str(
                function_mapping.get("description", "")
            ).strip()

            parameters = function_mapping.get(
                "parameters",
                {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )

            if not isinstance(parameters, Mapping):
                raise ValueError(
                    f"工具 {name!r} 的 parameters 必须是 Mapping。"
                )

            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": dict(parameters),
                    },
                }
            )
            seen_names.add(name)

        return normalized

    @staticmethod
    def _tool_to_mapping(tool: Mapping[str, Any] | Any) -> dict[str, Any]:
        if isinstance(tool, Mapping):
            return dict(tool)

        mapping: dict[str, Any] = {}

        for attribute in (
            "type",
            "function",
            "name",
            "description",
            "parameters",
        ):
            if hasattr(tool, attribute):
                mapping[attribute] = getattr(tool, attribute)

        if not mapping:
            raise TypeError(
                "工具必须是 Mapping，或包含 name、description、"
                "parameters 属性的对象。"
            )

        return mapping

    # ========================================================
    # 通用辅助方法
    # ========================================================

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.config.project_root))
        except ValueError:
            return str(path.resolve())

    def _is_path_inside_project(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.config.project_root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text

        omitted_chars = len(text) - max_chars
        return (
            text[:max_chars].rstrip()
            + f"\n\n...[内容已截断，省略 {omitted_chars} 个字符]"
        )


if __name__ == "__main__":

    config = PromptBuilderConfig(
        project_root=PROJECT_ROOT,
        skill_directories=[
            PROJECT_ROOT / "skills",
        ],
    )
    
    builder = PromptBuilder(config)
    bundle = builder.build()

    print(bundle.system_prompt)