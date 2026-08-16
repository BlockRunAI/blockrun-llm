"""
Whether the request actually requires an external action/tool, as distinct
from merely being sent by a host that exposes tools on every turn.

Python port of ``@blockrun/router-core`` ``tool-intent.ts``.

The detector intentionally looks for action+target pairs. A generic factual or
multiple-choice question must stay false even when the host attaches a large
tool schema; otherwise every tool-enabled host turn is over-routed as an agent
task and models may browse or mutate state unnecessarily.
"""

from __future__ import annotations

from typing import Any

from ._js import js_regex

# System prompts commonly describe every tool a host exposes. They are not
# evidence that the user asked to perform an action on this turn. Explicit host
# requirements should use tool_choice / requires_tools instead.
_EXPLICIT_TOOL = js_regex(
    r"\b(?:use|call|invoke)\s+(?:the\s+)?[\w.-]+\s+(?:tool|function|api)\b|\btool[_ -]?call\b"
    r"|使用.{0,20}(?:工具|函数|接口)|调用.{0,20}(?:工具|函数|接口)",
    ignorecase=True,
)
_CODE_ENVIRONMENT = js_regex(
    r"\b(?:run|execute)\s+(?:the\s+)?(?:tests?|command|script|build|linter)"
    r"|\b(?:edit|modify|patch|create|write|save|delete|rename|move|inspect|read)\b.{0,60}"
    r"\b(?:file|repository|repo|codebase|directory|folder)\b"
    r"|\b(?:terminal|shell|bash|zsh|pytest|npm test|pnpm test|git\s+(?:status|diff|commit)|docker)\b"
    r"|(?:运行|执行).{0,20}(?:测试|命令|脚本|构建)"
    r"|(?:修改|编辑|修复|创建|读取|检查|保存).{0,30}(?:文件|仓库|代码库|目录)",
    ignorecase=True,
)
_WEB_ACTION = js_regex(
    r"\b(?:browse|search|look up|fetch|open)\b.{0,80}"
    r"\b(?:web|website|url|online|documentation|docs|news|weather|price)\b"
    r"|(?:浏览|搜索|查询|打开).{0,30}(?:网页|网站|链接|文档|新闻|天气|价格)",
    ignorecase=True,
)
_STATEFUL_ACTION = js_regex(
    r"\b(?:refund|cancel|book|reserve|purchase|buy|return|exchange|transfer|update|change)\b.{0,80}"
    r"\b(?:order|booking|reservation|account|address|payment|subscription|ticket|flight|item)\b"
    r"|(?:退款|取消|预订|购买|退货|换货|转账|更新|修改).{0,30}"
    r"(?:订单|预订|账户|地址|付款|订阅|票|航班|商品)",
    ignorecase=True,
)


def infer_tool_requirement(
    prompt: str,
    system_prompt: str | None = None,
    tool_choice: Any = None,
) -> bool:
    """Return ``True`` when this turn actually asks for a tool action."""
    # OpenAI-compatible clients can state this requirement directly. Treat that
    # protocol signal as authoritative instead of trying to infer it from prose.
    if tool_choice == "none":
        return False
    if tool_choice == "required":
        return True
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return True

    text = prompt
    return bool(
        _EXPLICIT_TOOL.search(text)
        or _CODE_ENVIRONMENT.search(text)
        or _WEB_ACTION.search(text)
        or _STATEFUL_ACTION.search(text)
    )
