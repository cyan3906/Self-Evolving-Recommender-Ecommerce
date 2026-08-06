import re


def _extract_first_paragraph(content: str) -> str:
    """
    提取第一个普通正文段落。

    跳过：
    - 标题
    - 代码块
    - 列表
    - 引用
    """

    paragraphs = re.split(
        r"\n\s*\n",
        content.strip(),
    )

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


content = """
# Pytest Debug

- 读取测试报错
- 定位失败代码

> 修改代码前先确认根因。

这个 Skill 用于分析 pytest 测试失败，并根据报错信息定位业务代码问题。

修复完成后，需要重新执行相关测试。
"""

result = _extract_first_paragraph(content)

print(result)