"""
V3 portfolio router.

Python port of ``@blockrun/router-core`` ``portfolio.ts``.

This is deliberately local and deterministic: feature extraction, eligibility
checks and scoring read only request data plus the in-process model registry.
It is therefore safe for the hot path and provides a stable baseline for the
RouterBench evaluation before health telemetry / an optional judge are added.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from ._js import as_utc, js_bool, js_regex, parse_date
from .model_capabilities import DEFAULT_MODEL_CAPABILITIES
from .model_profiles import HISTORICAL_MODEL_PROFILES, LIVE_MODEL_PROFILES
from .selector import get_fallback_chain, select_model
from .strategy import RulesStrategy, sample_prompt, scan_limit_for
from .tool_intent import infer_tool_requirement
from .types import (
    CandidateScore,
    ModelPerformanceProfile,
    PortfolioBandWeights,
    PortfolioConfig,
    RouterOptions,
    RoutingDecision,
    TaskType,
    Tier,
    TierConfig,
)

DEFAULT_PORTFOLIO_WEIGHTS: PortfolioConfig = {
    "auto": {
        "quality": 0.47,
        "capability": 0.2,
        "cost": 0.18,
        "speed": 0.07,
        "reliability": 0.03,
        "legacy": 0.05,
    },
    "eco": {
        "quality": 0.36,
        "capability": 0.2,
        "cost": 0.28,
        "speed": 0.1,
        "reliability": 0.04,
        "legacy": 0.02,
    },
    "premium": {
        "quality": 0.58,
        "capability": 0.2,
        "cost": 0.08,
        "speed": 0.06,
        "reliability": 0.06,
        "legacy": 0.02,
    },
    "high_stakes_boost": {"quality": 0.08, "reliability": 0.05},
    "latency_sensitive_speed_boost": 0.08,
    "affinity_floor_gap": {"auto": 0.1, "eco": 0.22, "premium": 0.05},
}


@dataclass(frozen=True)
class TaskFeatures:
    task_type: TaskType
    estimated_input_tokens: int
    has_code: bool
    needs_tools: bool
    tools_available: bool
    needs_vision: bool
    needs_structured_output: bool
    latency_sensitive: bool
    high_stakes: bool
    language: str  # "zh" | "other"
    likely_parallel_tool_calls: bool
    complex_multi_tool_plan: bool
    agent_domain: str  # "airline" | "retail" | "web_research" | "other"
    deep_web_research: bool
    #: "standard" | "high" | "complex_high" | "policy_exception_simple" | "policy_exception"
    agent_risk: str
    terminal_tool_signal: bool
    terminal_safety_sensitive: bool
    implicit_terminal_code: bool


# ─── Compiled request features (ported 1:1 from the TypeScript regexes) ───

_EXPLICIT_REPEAT = js_regex(
    r"\b(?:in parallel|simultaneously|concurrently|for each|each of|every one|both"
    r"|(?:two|three|multiple|several)\s+(?:cities|locations|items|tasks|orders|users|files))\b"
    r"|并行|同时|分别|每个|各自|(?:两个|三个|多个)(?:城市|地点|项目|任务|订单|用户|文件)"
    r"|cada uno|para cada|simult[aá]neamente",
    ignorecase=True,
)
_SENTENCE_SPLIT = js_regex(r"[.!?。！？]+")
_ADDITIONALLY = js_regex(r"\b(?:also|additionally|furthermore)\b|另外|此外|그리고", ignorecase=True)
_AND_ALSO = js_regex(r"\band\s+(?:also|for the)\b", ignorecase=True)
_PAIRED_QUANTITY = js_regex(
    r"\b\d+(?:\.\d+)?\s+(?:and|or)\s+\d+(?:\.\d+)?\s*(?:gb|mb|tb|kg|g|ml|oz|cups?|cores?|cpus?)\b",
    ignorecase=True,
)
_TOOL_NAME_SPLIT = js_regex(r"[^a-z0-9\u3400-\u9fff]+")
_LINE_SPLIT = js_regex(r"\r?\n")
_QUANTITY_MENTION = js_regex(
    r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
    r"(?:oz|ounce|ounces|g|gram|grams|kg|ml|cups?|pieces?|tablespoons?)\b",
    ignorecase=True,
)
_REPEATED_LOOKUP = js_regex(
    r"\b(?:weather|climate|clima|tiempo|temperature|snow|news|report)\b"
    r"|天气|气象|温度|降雪|新闻|报告",
    ignorecase=True,
)
_MULTI_LOCATION_CONNECTOR = js_regex(r"\b(?:and also|both|y|e)\b|还有|以及|和|、", ignorecase=True)
_COMMA = js_regex(r"[,，]")
_ASCII_COMMA = js_regex(r",")
_DISTINCT_ORDER_PARTS = js_regex(
    r"\b(?:food|meal)\b[\s\S]*\bdrink\b|\bdrink\b[\s\S]*\b(?:food|meal)\b", ignorecase=True
)
_KOREAN_CLAUSES = js_regex(r"하고|그리고")

_OPERATION_TOKENS = frozenset(
    {
        "add",
        "delete",
        "remove",
        "cancel",
        "return",
        "exchange",
        "modify",
        "book",
        "transfer",
        "send",
        "upload",
        "download",
        "create",
        "close",
    }
)

_EXPLICIT_CODE_SIGNAL = js_regex(
    r"```|\b(?:typescript|javascript|python|rust|java|sql|stack trace|traceback|exception)\b"
    r"|\.(?:ts|tsx|js|py|go|rs)\b",
    ignorecase=True,
)
_CODE_CONSTRUCT_SIGNAL = js_regex(
    r"\b(?:implement|refactor|debug|write|edit|modify|create|define|review|fix)\b[\s\S]{0,48}"
    r"\b(?:api|function|class|method)\b"
    r"|\b(?:api|function|class|method)\b[\s\S]{0,48}"
    r"\b(?:code|implementation|typescript|javascript|python|rust|java)\b",
    ignorecase=True,
)
_NATIVE_CODE_SIGNAL = js_regex(
    r"\b(?:programmed|written|implemented?|code)\s+(?:in|using)\s+(?:c\+\+|c|rust|go)\b",
    ignorecase=True,
)
_AIRLINE_TOOL = js_regex(r"(?:flight|reservation|airport|baggage|passenger)")
_RETAIL_TOOL = js_regex(r"(?:order|product|item|return|exchange|address)")
_WEB_RESEARCH_TOOL = js_regex(r"^(?:web_?search|web_?fetch)$")
_CLUE_CONNECTORS = js_regex(
    r"\b(?:after|before|while|where|whose|which|in \d{4}|as of|over \d+|another|also|furthermore)\b"
    r"|(?:之后|之前|其中|截至|超过|另一个|此外)",
    ignorecase=True,
)
_ENTITY_RESOLUTION = js_regex(
    r"\b(?:identify|who (?:is|was)|what (?:is|was) the name"
    r"|which (?:person|player|company|country|city)|find the (?:person|player|name|entity))\b"
    r"|(?:找出|识别|是谁|哪位|名称是什么)",
    ignorecase=True,
)
_EXACT_ANSWER = js_regex(
    r"\b(?:exact answer|single best-supported answer|following clues|multiple public sources)\b"
    r"|(?:精确答案|根据.*线索|多个公开来源)",
    ignorecase=True,
)
_GLOBAL_OPTIMIZATION = js_regex(
    r"\b(?:cheapest|lowest[- ]price|least expensive|most expensive|highest(?:[- ]priced)?"
    r"|largest|smallest|maximum|minimum|best available|closest|not (?:cost|exceed))\b"
    r"|最便宜|最低价|最贵|最高价|最大|最小",
    ignorecase=True,
)
_GLOBAL_SCOPE = js_regex(
    r"\b(?:everything|all (?:(?:my|your|their|the) )?(?:future |upcoming )?"
    r"(?:items|orders|passengers|flights|reservations|bookings)"
    r"|every (?:item|order|passenger|flight|reservation|booking))\b"
    r"|全部|所有|每个",
    ignorecase=True,
)
_CROSS_RECORD = js_regex(
    r"\b(?:another|other|different|previous)\s+(?:order|reservation|booking|account|address)\b"
    r"|另一(?:个)?(?:订单|预订|账户|地址)|其他(?:订单|预订|账户|地址)",
    ignorecase=True,
)
_RESERVATION_ID = js_regex(r"\b[A-Z0-9]{6}\b")
_CROSS_RESERVATION_BATCH = js_regex(
    r"\b(?:two|three|multiple|several)(?:\s+of\s+(?:my|our|the))?\s+(?:upcoming\s+)?"
    r"(?:reservations?|bookings?)\b"
    r"|\b(?:a\s+)?(?:second|third)\s+(?:reservation|booking)\b",
    ignorecase=True,
)
_CONDITIONAL_GLOBAL_TERMS = js_regex(
    r"\b(?:if|that (?:contain|have)|longer than|shorter than|under|over|at (?:most|least)"
    r"|wherever possible)\b"
    r"|如果|超过|少于|不超过|尽可能",
    ignorecase=True,
)
_CONDITIONAL_GLOBAL_ACTIONS = js_regex(
    r"\b(?:cancel|change|upgrade|move|book)\b[\s\S]*\b(?:cancel|change|upgrade|move|book)\b"
    r"|取消[\s\S]*(?:升级|更改)|升级[\s\S]*(?:取消|更改)",
    ignorecase=True,
)
_RETURN_INTENT = js_regex(
    r"\b(?:return|refund|send back|get (?:my |the )?money back)\b|退货|退款|退回", ignorecase=True
)
_CARD_INTENT = js_regex(
    r"\b(?:amex|american express|visa|mastercard|credit card|debit card|different card"
    r"|another card|other card)\b"
    r"|信用卡|借记卡|其他卡|另一张卡",
    ignorecase=True,
)
_SINGLE_SELECTED_RETURN = js_regex(
    r"\b(?:return|refund|send back)\b[^.!?。！？]{0,96}"
    r"\b(?:the )?(?:pricier|cheaper|more expensive|less expensive|costlier|one)\b",
    ignorecase=True,
)
_NEGOTIATED_WORKFLOW = js_regex(r"\b(?:return|exchange)\b|退货|退回|换货|交换", ignorecase=True)
_NUMBERED_STEP = js_regex(r"(?:^|\s)\d+(?:\.\d+)*[.)]\s+")
_LATENCY_SENSITIVE = js_regex(
    r"\b(?:urgent|asap|fast|quick|low latency|real[- ]time)\b|尽快|马上|快速|低延迟",
    ignorecase=True,
)
_HIGH_STAKES = js_regex(
    r"\b(?:production|security|payment|legal|medical|financial|audit)\b"
    r"|生产|安全|支付|法律|医疗|财务|审计",
    ignorecase=True,
)
_TERMINAL_TOOL = js_regex(r"^(?:terminalexec|terminalinspect|terminalsendkeys)$")
_SIMPLE_TERMINAL_ARTIFACT = js_regex(
    r"\b(?:create|write|convert|generate|build|implement|run|fix|repair|debug|make)\b"
    r"[\s\S]{0,120}\b(?:file|script|csv|parquet|json|txt|server|endpoint)\b",
    ignorecase=True,
)
_TERMINAL_COMPLEX_REPAIR = js_regex(
    r"\b(?:multiple|several)\s+(?:scripts?|files?|components?)\b"
    r"|\b(?:pipeline|dependencies)\b[\s\S]{0,100}\b(?:fail|issue|fix|repair|run|execute)\b"
    r"|\b(?:identify|find|fix|repair)\s+(?:and\s+)?(?:fix\s+)?all\s+(?:the\s+)?issues\b",
    ignorecase=True,
)
_TERMINAL_RUNTIME = js_regex(
    r"\b(?:gcc|clang|rustc|javac|go\s+build|node|python)\b", ignorecase=True
)
_POLYGLOT = js_regex(r"\bpolyglot\b", ignorecase=True)
_BOTH_TOOLCHAINS = js_regex(
    r"\b(?:both|each)\b[\s\S]{0,120}\b(?:compilers?|runtimes?|toolchains?)\b", ignorecase=True
)
_COMPILE_VERB = js_regex(r"\b(?:compile|build|run|execute)\b", ignorecase=True)
_FRAMEWORK_ARTIFACT = js_regex(
    r"\b(?:pytorch|tensorflow|jax|onnx|state[_ -]?dict|checkpoint|safetensors?)\b"
    r"|\.(?:pth|pt|onnx)\b",
    ignorecase=True,
)
_NATIVE_TARGET = js_regex(
    r"\b(?:pure|native|programmed|written|implemented?)\s+(?:in|using)\s+(?:c\+\+|c|rust|go)\b"
    r"|\b(?:c\+\+|c|rust|go)\s+(?:program|binary|executable|cli|tool|implementation)\b",
    ignorecase=True,
)
_INFERENCE_VERB = js_regex(
    r"\b(?:inference|model|weights?|tensor|export|convert|load)\b", ignorecase=True
)
_COMPLEX_TERMINAL_OPERATION = js_regex(
    r"\b(?:git|ssh|nginx|https|certificate|authentication|credential|deploy|production|encrypt"
    r"|gpg|shred|securely delete|decommission|benchmark|evaluate|embedding|chess|image"
    r"|search the web|schema|statistical|statistics|aggregate|join|multiple inputs?)\b",
    ignorecase=True,
)
_TERMINAL_CREDENTIAL = js_regex(
    r"\b(?:ssh|nginx|certificate|authentication|credentials?|passwords?|api keys?|deploy"
    r"|production|encrypt|gpg|shred|securely delete|decommission)\b",
    ignorecase=True,
)
_TERMINAL_TOKEN_CREDENTIAL = js_regex(
    r"\b(?:access|auth|authentication|bearer|secret|api)\s+tokens?\b"
    r"|\btokens?\s+(?:secret|credential|authentication)\b",
    ignorecase=True,
)
_HAN = js_regex(r"[\u3400-\u9fff]")
_MULTIPLE_CHOICE = js_regex(r"(?:^|\n)\s*[A-D][.)]\s+", ignorecase=True, multiline=True)
_NUMERIC = js_regex(r"-?\d+(?:[.,]\d+)?")
_MATH_MARKERS = js_regex(
    r"[+×÷=%$€£¥]|\b(?:total|each|per|times|half|twice|percent|how many|how much|calculate)\b",
    ignorecase=True,
)
_TRAILING_QUESTION = js_regex(r"[?？]\s*\Z")
_DEBUG_TASK = js_regex(
    r"\b(?:bug|debug|error|failure|failing|regression|crash|修复|报错|错误|调试)\b", ignorecase=True
)
_CODE_EDIT_TASK = js_regex(
    r"\b(?:refactor|implement|patch|edit|rewrite|重构|实现|修改)\b", ignorecase=True
)
_EXTRACTION_TASK = js_regex(r"\b(?:extract|json|schema|csv|字段|提取)\b", ignorecase=True)
_REASONING_TASK = js_regex(
    r"\b(?:prove|derive|theorem|formal|mathematical|reasoning|证明|推导|定理|数学)\b",
    ignorecase=True,
)


def _likely_needs_parallel_tool_calls(
    prompt: str,
    needs_tools: bool,
    tool_count: int | None,
    tool_names: list[str] | None,
) -> bool:
    """Detect turns that probably need several tool calls.

    A deliberately conservative request-side feature: it uses only the prompt
    and the visible tool count, never benchmark categories or expected answers.
    """
    if not needs_tools or tool_count is None or tool_count < 1:
        return False
    text = prompt.strip()
    if _EXPLICIT_REPEAT.search(text):
        return True

    sentence_clauses = [
        part.strip() for part in _SENTENCE_SPLIT.split(text) if len(part.strip()) >= 8
    ]
    if (_ADDITIONALLY.search(text) and len(sentence_clauses) >= 2) or _AND_ALSO.search(text):
        return True

    if _PAIRED_QUANTITY.search(text):
        return True

    # Distinctive tokens from two visible tool names are a strong local signal
    # for a multi-operation turn (for example add_task + delete_task).
    lowered = text.lower()
    matched_operation_tokens = {
        token
        for name in (tool_names or [])
        for token in _TOOL_NAME_SPLIT.split(name.lower())
        if token in _OPERATION_TOKENS and token in lowered
    }
    # A single workflow naturally mentions domain nouns like order/item plus one
    # action. Upgrade only when two different visible operation verbs are
    # requested (for example cancel + book or add + delete).
    if len(matched_operation_tokens) >= 2:
        return True

    # Repeated food/logging entries are commonly expressed as several lines,
    # each with its own quantity rather than an explicit "for each" phrase.
    non_empty_lines = [line.strip() for line in _LINE_SPLIT.split(text) if line.strip()]
    quantity_mentions = _QUANTITY_MENTION.findall(text)
    if len(non_empty_lines) >= 2 and len(quantity_mentions) >= 2:
        return True

    # Weather prompts provide a useful language-independent high-confidence
    # pattern: a single lookup tool plus multiple locations joined in one turn.
    repeated_lookup = bool(_REPEATED_LOOKUP.search(text))
    multi_location_connector = bool(_MULTI_LOCATION_CONNECTOR.search(text))
    comma_separated_locations = len(_COMMA.findall(text)) >= 2
    if repeated_lookup and (multi_location_connector or comma_separated_locations):
        return True

    distinct_order_parts = bool(_DISTINCT_ORDER_PARTS.search(text))
    korean_parallel_clauses = len(_ASCII_COMMA.findall(text)) >= 3 and bool(
        _KOREAN_CLAUSES.search(text)
    )
    return distinct_order_parts or korean_parallel_clauses


def classify_task(prompt: str, system_prompt: str | None, options: RouterOptions) -> TaskFeatures:
    """Extract the request-side features the portfolio scorer ranks against."""
    full_text = f"{system_prompt or ''} {prompt}"
    estimated_input_tokens = math.ceil(len(full_text) / 4)
    # Feature regexes need request shape and intent, not the entire document.
    # Sample both ends so a long pasted artifact keeps the task instruction at
    # either boundary, while the full length still drives capacity decisions.
    scan_limit = scan_limit_for(options)
    scanned_prompt = sample_prompt(prompt, scan_limit)
    scanned_system_prompt = sample_prompt(system_prompt or "", scan_limit)
    scanned_full_text = f"{scanned_system_prompt} {scanned_prompt}"
    text = scanned_prompt.lower()

    explicit_code_signal = bool(_EXPLICIT_CODE_SIGNAL.search(scanned_prompt))
    # `class` is common in non-code Agent domains (for example airline cabin
    # class). Treat code constructs as code only when the prompt also contains
    # an implementation/editing cue, instead of letting a single ambiguous noun
    # redirect an entire tool session to the code-agent portfolio.
    code_construct_signal = bool(_CODE_CONSTRUCT_SIGNAL.search(scanned_prompt))
    native_code_signal = bool(_NATIVE_CODE_SIGNAL.search(scanned_prompt))
    has_code = explicit_code_signal or code_construct_signal or native_code_signal

    tools_available = options.get("has_tools", False)
    requires_tools = options.get("requires_tools")
    needs_tools = (
        requires_tools
        if requires_tools is not None
        else bool(tools_available and infer_tool_requirement(scanned_prompt, scanned_system_prompt))
    )
    tool_names = list(options.get("tool_names") or [])
    likely_parallel_tool_calls = _likely_needs_parallel_tool_calls(
        scanned_prompt, needs_tools, options.get("tool_count"), tool_names
    )
    normalized_tool_names = [name.lower() for name in tool_names]
    airline_tool_signal = any(_AIRLINE_TOOL.search(name) for name in normalized_tool_names)
    retail_tool_signal = any(_RETAIL_TOOL.search(name) for name in normalized_tool_names)
    web_research_tool_signal = any(
        _WEB_RESEARCH_TOOL.search(name) for name in normalized_tool_names
    )
    if airline_tool_signal and not retail_tool_signal:
        agent_domain = "airline"
    elif retail_tool_signal and not airline_tool_signal:
        agent_domain = "retail"
    elif web_research_tool_signal:
        agent_domain = "web_research"
    else:
        agent_domain = "other"

    # Distinguish a cheap lookup from a BrowseComp-like investigation. These
    # prompts require joining several clues, resolving an entity, and ending in
    # one exact answer; complete agent trajectories show that treating them as
    # ordinary search causes long, costly loops. This is request/tool-surface
    # evidence only and does not depend on a benchmark id or hidden answer.
    clue_connectors = _CLUE_CONNECTORS.findall(scanned_full_text)
    entity_resolution_signal = bool(_ENTITY_RESOLUTION.search(scanned_full_text))
    exact_answer_signal = bool(_EXACT_ANSWER.search(scanned_full_text))
    deep_web_research = agent_domain == "web_research" and (
        exact_answer_signal
        or (entity_resolution_signal and (len(clue_connectors) >= 3 or len(prompt) >= 320))
    )

    global_optimization_signal = bool(_GLOBAL_OPTIMIZATION.search(scanned_prompt))
    global_scope_signal = bool(_GLOBAL_SCOPE.search(scanned_prompt))
    global_choice_signal = global_optimization_signal or global_scope_signal
    cross_record_signal = bool(_CROSS_RECORD.search(scanned_prompt))
    reservation_ids = _RESERVATION_ID.findall(scanned_prompt)
    cross_reservation_batch_signal = agent_domain == "airline" and (
        bool(_CROSS_RESERVATION_BATCH.search(scanned_prompt)) or len(set(reservation_ids)) >= 2
    )
    conditional_global_workflow_signal = (
        agent_domain == "airline"
        and global_scope_signal
        and bool(_CONDITIONAL_GLOBAL_TERMS.search(scanned_prompt))
        and bool(_CONDITIONAL_GLOBAL_ACTIONS.search(scanned_prompt))
    )
    # A refund explicitly targeted at a named/non-original card can conflict
    # with account state and require escalation rather than a substitute action.
    # This narrow feature is visible on the first turn and avoids sending every
    # ordinary return workflow to the expensive policy specialist.
    policy_exception_signal = (
        agent_domain == "retail"
        and bool(_RETURN_INTENT.search(scanned_prompt))
        and bool(_CARD_INTENT.search(scanned_prompt))
    )
    # A comparative selector can mention two products while requesting only one
    # write (for example "send back the pricier one"). Three-repeat tau2
    # calibration found no quality gain from the policy specialist on these
    # single-write cases, so keep them in a distinct, lower-cost risk band.
    single_selected_policy_exception = policy_exception_signal and bool(
        _SINGLE_SELECTED_RETURN.search(scanned_prompt)
    )
    # Returns and exchanges often pivot after confirmation (return -> rethink ->
    # exchange -> choose a variant). That future state is not visible to a
    # task-start router, so treat the observable workflow verb as the risk cue.
    # Simpler cancellation and one-field order edits stay on the standard path.
    negotiated_workflow_signal = agent_domain == "retail" and bool(
        _NEGOTIATED_WORKFLOW.search(scanned_prompt)
    )
    numbered_steps = len(_NUMBERED_STEP.findall(scanned_prompt))
    complex_multi_tool_plan = likely_parallel_tool_calls and (
        (options.get("tool_count") or 0) >= 6 or numbered_steps >= 3 or len(prompt) > 1_200
    )

    if needs_tools and single_selected_policy_exception:
        agent_risk = "policy_exception_simple"
    elif needs_tools and policy_exception_signal:
        agent_risk = "policy_exception"
    # Airline prompts that require a global optimum (for example the cheapest
    # itinerary across several candidates) are materially harder than applying
    # one change to every passenger in a known reservation. Full-session
    # evidence supports Sonnet for the former, while upgrading the latter merely
    # because it says "all passengers" caused a large cost increase without a
    # quality gain.
    elif (
        needs_tools
        and agent_domain == "airline"
        and (global_optimization_signal or conditional_global_workflow_signal)
    ):
        agent_risk = "complex_high"
    elif needs_tools and (
        likely_parallel_tool_calls
        or global_choice_signal
        or cross_record_signal
        or cross_reservation_batch_signal
        or negotiated_workflow_signal
    ):
        agent_risk = "high"
    else:
        agent_risk = "standard"

    needs_vision = options.get("has_vision", False)
    needs_structured_output = options.get("requires_structured_output", False)
    latency_sensitive = bool(_LATENCY_SENSITIVE.search(scanned_full_text))
    high_stakes = bool(_HIGH_STAKES.search(scanned_full_text))

    # Terminal tasks often describe the desired artifact rather than naming a
    # programming language. Treat only small, deterministic local build/file
    # work as implicit code. Operational deployment, credentials, destructive
    # work, evaluation, vision, and broad search stay on the stronger generic
    # tool-agent path. This is a request-side feature, not a benchmark ID list.
    terminal_tool_signal = any(_TERMINAL_TOOL.search(name) for name in normalized_tool_names)
    simple_terminal_artifact = bool(_SIMPLE_TERMINAL_ARTIFACT.search(scanned_prompt))
    # Multi-file repair is qualitatively different from fixing one known local
    # script. The agent must preserve state across inspections, infer ordering
    # and dependencies, edit several artifacts, and close the loop with tests.
    terminal_complex_repair = terminal_tool_signal and bool(
        _TERMINAL_COMPLEX_REPAIR.search(scanned_prompt)
    )
    # One artifact that must be accepted by multiple compilers/runtimes is not a
    # routine file-writing task. It requires reasoning across incompatible
    # grammars and validating every execution path.
    mentioned_terminal_runtimes = {
        " ".join(name.lower().split()) for name in _TERMINAL_RUNTIME.findall(scanned_prompt)
    }
    terminal_cross_runtime_artifact = terminal_tool_signal and (
        bool(_POLYGLOT.search(scanned_prompt))
        or bool(_BOTH_TOOLCHAINS.search(scanned_prompt))
        or (len(mentioned_terminal_runtimes) >= 2 and bool(_COMPILE_VERB.search(scanned_prompt)))
    )
    # Framework-to-native ports combine binary checkpoint inspection, weight
    # export, tensor-layout reasoning, image/data decoding, and a separately
    # compiled runtime.
    terminal_framework_to_native_artifact = (
        terminal_tool_signal
        and bool(_FRAMEWORK_ARTIFACT.search(scanned_prompt))
        and bool(_NATIVE_TARGET.search(scanned_prompt))
        and bool(_INFERENCE_VERB.search(scanned_prompt))
    )
    if (
        needs_tools
        and (
            terminal_complex_repair
            or terminal_cross_runtime_artifact
            or terminal_framework_to_native_artifact
        )
        and agent_risk in ("standard", "high")
    ):
        agent_risk = "complex_high"

    complex_terminal_operation = bool(_COMPLEX_TERMINAL_OPERATION.search(scanned_prompt))
    # A bare "token" is not a credential signal: blockchain, tokenizer, and LLM
    # tasks use that word routinely (for example "token transfers"). Only treat
    # it as sensitive when the prompt gives it an authentication/secret
    # qualifier. API keys remain an unambiguous high-risk signal on their own.
    terminal_credential_signal = bool(_TERMINAL_CREDENTIAL.search(scanned_prompt)) or bool(
        _TERMINAL_TOKEN_CREDENTIAL.search(scanned_prompt)
    )
    terminal_safety_sensitive = terminal_tool_signal and (high_stakes or terminal_credential_signal)
    implicit_terminal_code = bool(
        needs_tools
        and terminal_tool_signal
        and agent_risk == "standard"
        and not high_stakes
        and not complex_terminal_operation
        and numbered_steps < 3
        and len(prompt) <= 1_000
        and simple_terminal_artifact
    )
    language = "zh" if _HAN.search(scanned_full_text) else "other"
    multiple_choice_signals = len(_MULTIPLE_CHOICE.findall(scanned_prompt))
    numeric_signals = len(_NUMERIC.findall(scanned_prompt))
    compact_math_problem = (
        not has_code
        and len(prompt) < 2_500
        and numeric_signals >= 2
        and (
            bool(_MATH_MARKERS.search(scanned_prompt))
            or bool(_TRAILING_QUESTION.search(scanned_prompt.strip()))
            or numeric_signals >= 3
        )
    )

    task_type: TaskType = "chat"
    if needs_vision:
        task_type = "vision"
    elif estimated_input_tokens > 80_000:
        task_type = "long_context"
    elif needs_tools and (has_code or implicit_terminal_code):
        task_type = "code_agent"
    elif needs_tools and likely_parallel_tool_calls and not complex_multi_tool_plan:
        task_type = "tool_agent_parallel"
    elif needs_tools:
        task_type = "tool_agent"
    elif multiple_choice_signals >= 3:
        task_type = "reasoning_mcq"
    elif compact_math_problem:
        task_type = "reasoning_math"
    elif _DEBUG_TASK.search(text):
        task_type = "debug"
    elif has_code or _CODE_EDIT_TASK.search(text):
        task_type = "code_edit"
    elif needs_structured_output or _EXTRACTION_TASK.search(text):
        task_type = "extraction"
    elif _REASONING_TASK.search(text):
        task_type = "reasoning"

    return TaskFeatures(
        task_type=task_type,
        estimated_input_tokens=estimated_input_tokens,
        has_code=has_code,
        needs_tools=bool(needs_tools),
        tools_available=bool(tools_available),
        needs_vision=bool(needs_vision),
        needs_structured_output=bool(needs_structured_output),
        latency_sensitive=latency_sensitive,
        high_stakes=high_stakes,
        language=language,
        likely_parallel_tool_calls=likely_parallel_tool_calls,
        complex_multi_tool_plan=bool(complex_multi_tool_plan),
        agent_domain=agent_domain,
        deep_web_research=bool(deep_web_research),
        agent_risk=agent_risk,
        terminal_tool_signal=terminal_tool_signal,
        terminal_safety_sensitive=terminal_safety_sensitive,
        implicit_terminal_code=implicit_terminal_code,
    )


_AFFINITY_BASE = 0.68


def affinity(
    model_id: str,
    task: TaskType,
    language: str = "other",
    agent_domain: str = "other",
    deep_web_research: bool = False,
    agent_risk: str = "standard",
    terminal_tool_signal: bool = False,
    terminal_safety_sensitive: bool = False,
) -> float:
    """Task affinity for a model, on the same evidence bands as upstream.

    Model family names are intentionally similar (for example
    ``gemini-2.5-flash`` vs ``gemini-2.5-flash-lite``). A substring match would
    let a smaller sibling inherit a capability claim measured only for the
    flagship, so these assignments are model-exact; a sibling can be added only
    with its own evidence.
    """
    model_id_lower = model_id.lower()
    model_name = model_id_lower[model_id_lower.find("/") + 1 :]

    def match(values: list[str], score: float) -> float:
        return score if model_name in values else 0.0

    base = _AFFINITY_BASE

    if task == "code_agent":
        if terminal_tool_signal and agent_risk == "complex_high":
            # Strong native tool loop until the Responses function-output fix is
            # deployed on both gateways; keep Codex available below the floor.
            return max(
                base,
                match(["claude-sonnet-5"], 1),
                match(["gpt-5.3-codex"], 0.87),
                match(["gpt-5-mini"], 0.78),
                match(["gemini-3.5-flash"], 0.76),
            )
        # Seven valid full agent + official Terminal-Bench trajectories
        # (2026-07-28) gave GPT-5 Mini 4/7 resolved tasks versus 1/7 for the
        # prior dynamic code-agent choice. Its token-normalized total cost was
        # higher in this small calibration, so keep Codex and Sonnet's quality
        # priors above it. DeepSeek V4 Pro is kept below the primary band after
        # two consecutive mid-trajectory provider timeouts.
        return max(
            base,
            match(["gpt-5.3-codex"], 1),
            match(["claude-sonnet-5"], 0.98),
            match(["gpt-5-mini"], 0.96),
            match(["gemini-3.5-flash"], 0.92),
            match(["kimi-k3"], 0.9),
            match(["deepseek-v4-pro", "glm-5.2"], 0.88),
        )

    if task == "tool_agent":
        if terminal_tool_signal and agent_risk == "complex_high":
            # Keep the Responses-API Codex path outside auto's affinity floor
            # until the gateway fix that preserves function_call_output is live
            # on both chains. Sonnet has a verified native multi-turn tool loop.
            return max(
                base,
                match(["claude-sonnet-5"], 1),
                match(["gpt-5.3-codex"], 0.87),
                match(["gpt-5-mini"], 0.78),
                match(["gemini-3.5-flash"], 0.76),
            )
        if terminal_tool_signal and not terminal_safety_sensitive:
            # Seven official Terminal-Bench calibration trajectories favoured
            # GPT-5 Mini over the prior dynamic choice. Admit Codex/Sonnet as
            # close fallbacks, but let actual request cost break the tie.
            return max(
                base,
                match(["gpt-5-mini"], 1),
                match(["gpt-5.3-codex"], 0.98),
                match(["claude-sonnet-5"], 0.9),
                match(["gemini-3.5-flash"], 0.89),
            )
        if terminal_tool_signal and terminal_safety_sensitive:
            # Two complete agent observations on the public Terminal-Bench
            # new-encrypt-command task ended in Codex repeating the same
            # TerminalExec input until the loop guard fired.
            return max(
                base,
                match(["claude-sonnet-5"], 1),
                match(["claude-opus-4.8"], 0.9),
                match(["gpt-5.3-codex"], 0.84),
            )
        if agent_domain == "web_research":
            # Complete-session BrowseComp calibration supersedes the earlier
            # single-case Opus promotion: strict deduplicated evidence has
            # Sonnet 5 at 2/9 versus Opus 5 at 0/3, while Opus also costs more
            # and has a much longer tail.
            if deep_web_research:
                return max(
                    base,
                    match(["claude-sonnet-5"], 1),
                    match(["gpt-5-mini"], 0.88),
                    match(["gemini-3.5-flash"], 0.84),
                    match(["claude-opus-5"], 0.8),
                    match(["claude-opus-4.8"], 0.78),
                )
            return max(
                base,
                match(["claude-sonnet-5"], 1),
                match(["gpt-5-mini"], 0.88),
                match(["gemini-3.5-flash"], 0.86),
                match(["claude-opus-5"], 0.84),
                match(["claude-opus-4.8"], 0.82),
            )
        # Full-trajectory tau2 calibration (2026-07-28, official gpt-4.1
        # simulator): Sonnet 5 completed both an airline policy task and a
        # retail multi-write task with reward 1.0. Gemini 3.5 Flash emitted
        # function calls as plain text after the first structured calls.
        if agent_domain == "retail":
            # Full-session calibration: GPT-5 Mini completed two local/single
            # retail workflows at a fraction of Sonnet's token cost. It remains
            # ineligible for promotion when the prompt asks for multiple
            # actions, cross-record discovery, or a global optimum. DeepSeek V4
            # Pro completed all three high-risk retail calibration trajectories.
            if agent_risk == "standard":
                return max(
                    base,
                    match(["gpt-5-mini"], 1),
                    match(["claude-sonnet-5"], 0.88),
                    match(["gemini-3.5-flash"], 0.82),
                    match(["gpt-5.3-codex"], 0.81),
                    match(["kimi-k3"], 0.78),
                    match(["deepseek-v4-pro"], 0.76),
                )
            if agent_risk == "policy_exception":
                return max(
                    base,
                    match(["gpt-4.1"], 1),
                    match(["claude-sonnet-5"], 0.9),
                    match(["deepseek-v4-pro"], 0.82),
                    match(["gpt-5-mini"], 0.8),
                    match(["gpt-4o-mini"], 0.76),
                )
            if agent_risk == "policy_exception_simple":
                return max(
                    base,
                    match(["gpt-5-mini"], 1),
                    match(["gpt-4.1"], 0.86),
                    match(["deepseek-v4-pro"], 0.82),
                    match(["gpt-4o-mini"], 0.8),
                )
            return max(
                base,
                match(["deepseek-v4-pro"], 1),
                match(["claude-sonnet-5"], 0.88),
                match(["gemini-3.5-flash"], 0.82),
                match(["gpt-5.3-codex"], 0.81),
                match(["kimi-k3"], 0.78),
                match(["gpt-5-mini"], 0.76),
            )
        # Standard airline workflows stay on GPT-5 Mini: six full-session
        # development trajectories gave it the same 5/6 success as Sonnet at
        # roughly one order of magnitude lower normalized token cost. Promote
        # only global optimization / conditional-global work.
        if agent_domain == "airline":
            if agent_risk == "complex_high":
                return max(
                    base,
                    match(["claude-sonnet-5"], 1),
                    match(["gpt-5-mini"], 0.78),
                    match(["gemini-3.5-flash"], 0.76),
                    match(["deepseek-v4-pro"], 0.74),
                )
            return max(
                base,
                match(["gpt-5-mini"], 1),
                match(["claude-sonnet-5"], 0.9),
                match(["gemini-3.5-flash"], 0.8),
                match(["deepseek-v4-pro"], 0.76),
            )
        return max(
            base,
            match(["claude-sonnet-5"], 1),
            match(["gemini-3.5-flash"], 0.88),
            match(["gpt-5.3-codex"], 0.87),
            match(["gpt-5-mini"], 0.84),
            match(["kimi-k3"], 0.85),
            match(["deepseek-v4-pro"], 0.82),
        )

    if task == "tool_agent_parallel":
        if terminal_tool_signal:
            # Multi-file Terminal work is not equivalent to a one-turn parallel
            # function-call benchmark. Sonnet is the strongest trajectory-tested
            # cost-controlled default; Opus remains a close safety fallback.
            if terminal_safety_sensitive:
                return max(
                    base,
                    match(["claude-sonnet-5"], 1),
                    match(["claude-opus-4.8"], 0.9),
                    match(["gpt-5.3-codex"], 0.86),
                )
            return max(
                base,
                match(["gpt-5-mini"], 1),
                match(["gpt-5.3-codex"], 0.98),
                match(["claude-sonnet-5"], 0.92),
                match(["gemini-3.5-flash"], 0.88),
            )
        if agent_domain == "web_research":
            if deep_web_research:
                return max(
                    base,
                    match(["claude-sonnet-5"], 1),
                    match(["gpt-5-mini"], 0.88),
                    match(["gemini-3.5-flash"], 0.84),
                    match(["claude-opus-5"], 0.8),
                    match(["claude-opus-4.8"], 0.78),
                )
            return max(
                base,
                match(["claude-sonnet-5"], 1),
                match(["gpt-5-mini"], 0.88),
                match(["gemini-3.5-flash"], 0.86),
                match(["claude-opus-5"], 0.84),
                match(["claude-opus-4.8"], 0.82),
            )
        if agent_domain == "retail":
            if agent_risk == "policy_exception":
                return max(
                    base,
                    match(["gpt-4.1"], 1),
                    match(["claude-sonnet-5"], 0.9),
                    match(["deepseek-v4-pro"], 0.82),
                    match(["gpt-5-mini"], 0.8),
                    match(["gpt-4o-mini"], 0.76),
                )
            if agent_risk == "policy_exception_simple":
                return max(
                    base,
                    match(["gpt-5-mini"], 1),
                    match(["gpt-4.1"], 0.86),
                    match(["deepseek-v4-pro"], 0.82),
                    match(["gpt-4o-mini"], 0.8),
                )
            return max(
                base,
                match(["deepseek-v4-pro"], 1),
                match(["claude-sonnet-5"], 0.88),
                match(["claude-opus-4.8"], 0.84),
                match(["gpt-5-mini"], 0.78),
                match(["gemini-3.5-flash"], 0.76),
            )
        if agent_domain == "airline":
            if agent_risk == "complex_high":
                return max(
                    base,
                    match(["claude-sonnet-5"], 1),
                    match(["gpt-5-mini"], 0.78),
                    match(["claude-opus-4.8"], 0.76),
                    match(["gemini-3.5-flash"], 0.74),
                )
            return max(
                base,
                match(["gpt-5-mini"], 1),
                match(["claude-sonnet-5"], 0.9),
                match(["gemini-3.5-flash"], 0.8),
            )
        # RouterBench calibration, 2026-07-26: Opus 4.8 produced complete
        # multi-call payloads on 2/3 multilingual BFCL parallel cases. Gemini
        # 3.5 Flash, Sonnet 5, DeepSeek V4 Pro, and Grok 4.5 were 0/3. This
        # narrow prior only applies after the conservative prompt feature above.
        return max(
            base,
            match(["claude-opus-4.8"], 1),
            match(["claude-sonnet-5"], 0.84),
            match(["grok-4.5"], 0.82),
            match(["gemini-3.5-flash"], 0.8),
            match(["deepseek-v4-pro"], 0.78),
        )

    if task in ("code_edit", "debug"):
        return max(
            base,
            match(["gpt-5.3-codex"], 1),
            match(["claude-sonnet-4.6"], 0.94),
            match(["glm-5.2"], 0.9),
            match(["kimi-k2.7", "deepseek-v4-pro"], 0.86),
        )

    if task == "reasoning":
        return max(
            base,
            match(["claude-sonnet-5", "claude-sonnet-4.6"], 0.98),
            match(["deepseek-v4-pro"], 0.95),
            match(["grok-4.5"], 0.94),
            match(["gemini-3.1-pro", "gemini-3.5-flash"], 0.92),
        )

    if task == "reasoning_mcq":
        # RouterBench calibration (2026-07-28, six stratified GPQA Diamond
        # tasks, identical agent adapter and 512-token budget): Gemini 3 Flash
        # Preview scored 5/6, Gemini 3.5 Flash 4/6, and Gemini 3.1 Pro 3/6 while
        # costing ~170x more than Flash. Version recency alone is not a quality
        # signal, and unused host tools must not change this model choice.
        return max(
            base,
            match(["gemini-3-flash-preview"], 1),
            match(["gemini-3.5-flash"], 0.91),
            match(["grok-4.5"], 0.9),
            match(["claude-sonnet-5"], 0.88),
            match(["deepseek-v4-pro"], 0.84),
        )

    if task == "reasoning_math":
        # Same calibration, five multilingual MGSM tasks: Gemini 3.5 Flash was
        # 5/5 with the lowest cost and latency; four current flagships were 4/5
        # and Kimi K2.7 was 3/5.
        return max(
            base,
            match(["gemini-3.5-flash"], 1),
            match(["grok-4.5"], 0.93),
            match(["claude-sonnet-5", "deepseek-v4-pro", "kimi-k3"], 0.9),
            match(["kimi-k2.7"], 0.84),
        )

    if task == "vision":
        return max(
            base,
            match(["gemini-3.1-pro"], 0.96),
            match(["qwen3.7-max", "claude-sonnet-4.6", "kimi-k2.7", "grok-4.3"], 0.9),
        )

    if task == "long_context":
        # Long-context eligibility is necessary but not sufficient: a provider
        # can advertise a 1M window yet return an empty completion near that
        # boundary. Keep the proven long-context flagship in the lead and put
        # less-established alternatives in a separate affinity band so price
        # alone cannot displace it.
        return max(
            base,
            match(["gemini-3.1-pro"], 1),
            match(["qwen3.7-max", "glm-5.2"], 0.89),
            match(["gemini-3.5-flash"], 0.88),
            match(["deepseek-v4-pro"], 0.85),
        )

    if task == "extraction":
        # A structured extraction must preserve both the output contract and the
        # source-language fields. For Mandarin input, keep the language-native
        # Kimi candidate in a distinct affinity band. This is deliberately a
        # candidate-pool decision (rather than a brittle post-hoc override): it
        # still falls back normally if that model is unavailable or ineligible.
        kimi_extraction_affinity = 1.0 if language == "zh" else 0.9
        return max(
            base,
            match(["gemini-3.5-flash", "gemini-2.5-flash", "gpt-4o-mini"], 0.9),
            match(["claude-sonnet-5", "claude-sonnet-4.6"], 0.9),
            match(["kimi-k3", "kimi-k2.7"], kimi_extraction_affinity),
        )

    return max(base, match(["gemini-3.5-flash", "gemini-2.5-flash", "kimi-k3", "kimi-k2.7"], 0.86))


def evidence_candidates(task: TaskType) -> list[str]:
    """Models with task-level calibration evidence, added to the tier chain."""
    if task == "code_agent":
        return [
            "openai/gpt-5.3-codex",
            "anthropic/claude-sonnet-5",
            "openai/gpt-5-mini",
            "google/gemini-3.5-flash",
            "moonshot/kimi-k3",
            "deepseek/deepseek-v4-pro",
        ]
    if task == "tool_agent":
        return [
            "anthropic/claude-sonnet-5",
            "anthropic/claude-opus-5",
            "openai/gpt-5-mini",
            "openai/gpt-4.1",
            "openai/gpt-4o-mini",
            "google/gemini-3.5-flash",
            "openai/gpt-5.3-codex",
            "moonshot/kimi-k3",
            "deepseek/deepseek-v4-pro",
        ]
    if task == "tool_agent_parallel":
        return [
            "anthropic/claude-opus-5",
            "anthropic/claude-opus-4.8",
            "anthropic/claude-sonnet-5",
            "openai/gpt-5-mini",
            "openai/gpt-4.1",
            "openai/gpt-4o-mini",
            "xai/grok-4.5",
            "google/gemini-3.5-flash",
            "deepseek/deepseek-v4-pro",
        ]
    if task == "long_context":
        return [
            "google/gemini-3.1-pro",
            "deepseek/deepseek-v4-pro",
            "qwen/qwen3.7-max",
            "zai/glm-5.2",
            "google/gemini-3.5-flash",
        ]
    if task == "reasoning_mcq":
        return [
            "google/gemini-3-flash-preview",
            "google/gemini-3.5-flash",
            "xai/grok-4.5",
            "anthropic/claude-sonnet-5",
            "deepseek/deepseek-v4-pro",
        ]
    if task == "reasoning_math":
        return [
            "google/gemini-3.5-flash",
            "xai/grok-4.5",
            "anthropic/claude-sonnet-5",
            "deepseek/deepseek-v4-pro",
            "moonshot/kimi-k3",
        ]
    return []


def is_eligible(
    model_id: str,
    features: TaskFeatures,
    max_output_tokens: int,
    options: RouterOptions,
) -> bool:
    """Hard capability filter: capacity, tools, vision, structured output."""
    host_capabilities = options.get("model_capabilities") or {}
    model = host_capabilities.get(model_id) or DEFAULT_MODEL_CAPABILITIES.get(model_id)
    # Preserve compatibility for temporarily catalog-less fallback IDs. They are
    # kept behind known-model candidates but are not silently dropped.
    if not model:
        return True
    if features.needs_tools and not model["supports_tools"]:
        return False
    if features.needs_vision and not model["supports_vision"]:
        return False
    if features.needs_structured_output and not model["supports_tools"]:
        return False
    if model["max_output_tokens"] < max_output_tokens:
        return False
    return model["context_window"] >= (features.estimated_input_tokens + max_output_tokens) * 1.1


def estimated_cost(
    model_id: str, options: RouterOptions, input_tokens: int, output_tokens: int
) -> float:
    price = options["model_pricing"].get(model_id)
    if not price:
        return math.inf
    flat = price.get("flat_price")
    if flat:
        return float(flat)
    return (
        input_tokens * price.get("input_price", 0) + output_tokens * price.get("output_price", 0)
    ) / 1_000_000


@dataclass(frozen=True)
class _ProfileScore:
    quality: float | None
    speed: float
    tail_speed: float
    reliability: float
    freshness: float


def profile_score(model_id: str, options: RouterOptions, now: datetime) -> _ProfileScore | None:
    """Weak speed/reliability priors, decayed by age and sample count."""
    host_performance = options.get("model_performance") or {}
    profile: ModelPerformanceProfile | None = (
        host_performance.get(model_id)
        or LIVE_MODEL_PROFILES.get(model_id)
        or HISTORICAL_MODEL_PROFILES.get(model_id)
    )
    if not profile:
        return None
    measured_at = parse_date(profile.get("measured_at", ""))
    if measured_at is None:
        return None
    age_days = max(0.0, (now - measured_at).total_seconds() / 86_400)
    # A 30-day half-life makes old data a tie-breaker only. Small probe runs are
    # also weak evidence: three quick samples should not overturn a curated tier
    # ordering merely because of a transient provider tail. Callers that inject
    # an observation without a sample count retain the legacy full-confidence
    # behaviour for compatibility.
    samples = profile.get("samples")
    sample_confidence = 1.0 if samples is None else min(1.0, max(0.0, samples) / 10)
    freshness = math.pow(0.5, age_days / 30) * sample_confidence
    intelligence_index = profile.get("intelligence_index")
    quality = None if intelligence_index is None else min(1.0, intelligence_index / 50)
    latency_ms = profile.get("latency_ms", 0)
    speed = min(
        1.0,
        (2_000 / max(500, latency_ms) + profile.get("output_tokens_per_second", 0) / 250) / 2,
    )
    tail_speed = min(1.0, 3_000 / max(750, profile.get("p95_latency_ms", latency_ms)))
    reliability = max(0.0, 1 - profile.get("error_rate", 0))
    return _ProfileScore(
        quality=quality,
        speed=speed,
        tail_speed=tail_speed,
        reliability=reliability,
        freshness=freshness,
    )


_WEB_RESEARCH_FALLBACK_ORDER = [
    "anthropic/claude-sonnet-5",
    "openai/gpt-5-mini",
    "google/gemini-3.5-flash",
    "anthropic/claude-opus-5",
    "anthropic/claude-opus-4.8",
    "openai/gpt-5.3-codex",
]


class PortfolioStrategy:
    """Candidate router used for Auto.

    Rules still set the capability tier; V3 ranks within it.
    """

    name = "portfolio"

    def route(
        self,
        prompt: str,
        system_prompt: str | None,
        max_output_tokens: int,
        options: RouterOptions,
    ) -> RoutingDecision:
        features = classify_task(prompt, system_prompt, options)
        rules_options: RouterOptions = dict(options)  # type: ignore[assignment]
        rules_options["requires_tools"] = features.needs_tools
        base = RulesStrategy().route(prompt, system_prompt, max_output_tokens, rules_options)
        tier_configs = base.get("tier_configs")
        if not tier_configs:
            return base

        target_tier: Tier = (
            "REASONING"
            if features.task_type in ("reasoning_mcq", "reasoning_math")
            and base["tier"] in ("SIMPLE", "MEDIUM")
            else base["tier"]
        )
        tier_config = tier_configs.get(target_tier)
        configured_candidates = get_fallback_chain(target_tier, tier_configs) if tier_config else []
        # Evidence candidates join the configured chain, but the host's
        # unavailable set applies to both: the configured side arrives filtered
        # through RulesStrategy, and a dead evidence model must not re-enter here.
        unavailable = set(options.get("unavailable_models") or ())
        chain = [
            model
            for model in dict.fromkeys(
                [*configured_candidates, *evidence_candidates(features.task_type)]
            )
            if isinstance(model, str) and model and model not in unavailable
        ]
        eligible = [
            model for model in chain if is_eligible(model, features, max_output_tokens, options)
        ]
        eligible_candidates = eligible if eligible else chain
        if not eligible_candidates:
            return base

        routing_profile = options.get("routing_profile")
        portfolio = options["config"].get("portfolio") or DEFAULT_PORTFOLIO_WEIGHTS
        profile_weights: PortfolioBandWeights
        if routing_profile == "eco":
            profile_weights = portfolio["eco"]
            base_floor_gap = portfolio["affinity_floor_gap"]["eco"]
        elif routing_profile == "premium":
            profile_weights = portfolio["premium"]
            base_floor_gap = portfolio["affinity_floor_gap"]["premium"]
        else:
            profile_weights = portfolio["auto"]
            base_floor_gap = portfolio["affinity_floor_gap"]["auto"]

        affinities = {
            model: affinity(
                model,
                features.task_type,
                features.language,
                features.agent_domain,
                features.deep_web_research,
                features.agent_risk,
                features.terminal_tool_signal,
                features.terminal_safety_sensitive,
            )
            for model in eligible_candidates
        }
        best_affinity = max(affinities.values())
        specific_affinity = [
            model for model in eligible_candidates if affinities[model] > _AFFINITY_BASE
        ]
        # A tier's fallback list is primarily an availability/recovery chain, not
        # a set of equally validated substitutes. Re-ranking every fallback lets
        # a cheap generic model displace the curated primary merely because it
        # has a favourable short performance probe. Only promote models with
        # explicit task affinity; otherwise retain the first eligible tier model.
        affinity_pool = specific_affinity if specific_affinity else [eligible_candidates[0]]
        # Generic Terminal work has much wider trajectory variance than a
        # BFCL-like one-turn parallel call. Keep the strong-model safety band,
        # but admit the next capable tier so Auto's cost/reliability score can
        # reject an Opus primary that is materially more expensive without
        # measured benefit.
        affinity_floor_gap = (
            max(base_floor_gap, 0.15 if features.terminal_safety_sensitive else 0.12)
            if features.terminal_tool_signal
            else base_floor_gap
        )
        candidates = [
            model
            for model in affinity_pool
            if affinities[model] >= best_affinity - affinity_floor_gap
        ]
        costs = [
            estimated_cost(model, options, features.estimated_input_tokens, max_output_tokens)
            for model in candidates
        ]
        finite_costs = [cost for cost in costs if math.isfinite(cost)]
        min_cost = min(finite_costs) if finite_costs else 0.0
        max_cost = max(finite_costs) if finite_costs else 1.0

        now = as_utc(options.get("now"))
        ranked_entries: list[CandidateScore] = []
        for index, model in enumerate(candidates):
            cost = estimated_cost(
                model, options, features.estimated_input_tokens, max_output_tokens
            )
            cost_score = (
                1 - (cost - min_cost) / (max_cost - min_cost)
                if math.isfinite(cost) and max_cost > min_cost
                else 0.5
            )
            capability_score = (
                1.0 if is_eligible(model, features, max_output_tokens, options) else 0.0
            )
            profile = profile_score(model, options, now)
            # Fresh observations can refine affinity. Historical observations
            # fade quickly and never replace task-level RouterBench evidence.
            model_affinity = affinities[model]
            if profile is None or profile.quality is None:
                observed_quality = model_affinity
            else:
                observed_quality = (
                    model_affinity * (1 - profile.freshness) + profile.quality * profile.freshness
                )
            observed_speed = profile.speed * profile.freshness if profile else 0.5
            observed_tail_speed = profile.tail_speed * profile.freshness if profile else 0.5
            observed_reliability = (
                profile.reliability * profile.freshness + (1 - profile.freshness)
                if profile
                else 1.0
            )
            # Preserve a small amount of the hand-curated fallback order while
            # V3's task affinity and real request constraints do the main work.
            legacy_score = 1 - index / max(1, len(candidates) - 1)
            quality_weight = profile_weights["quality"] + (
                portfolio["high_stakes_boost"]["quality"] if features.high_stakes else 0
            )
            speed_score = observed_tail_speed if features.latency_sensitive else observed_speed
            speed_weight = profile_weights["speed"] + (
                portfolio["latency_sensitive_speed_boost"] if features.latency_sensitive else 0
            )
            reliability_weight = profile_weights["reliability"] + (
                portfolio["high_stakes_boost"]["reliability"] if features.high_stakes else 0
            )
            score = (
                observed_quality * quality_weight
                + capability_score * profile_weights["capability"]
                + cost_score * profile_weights["cost"]
                + speed_score * speed_weight
                + observed_reliability * reliability_weight
                + legacy_score * profile_weights["legacy"]
            )
            ranked_entries.append(
                {
                    "model": model,
                    "score": score,
                    "quality": observed_quality,
                    "cost": cost_score,
                    "speed": speed_score,
                    "reliability": observed_reliability,
                }
            )
        ranked_entries.sort(key=lambda entry: entry["score"], reverse=True)
        scored_models = [entry["model"] for entry in ranked_entries]
        # The affinity floor controls which models may compete for the primary;
        # it must not erase availability fallbacks. Append all remaining eligible
        # models in their curated chain order after the scored primary pool.
        if features.agent_domain == "web_research":
            ranked = [
                *scored_models,
                *[
                    model
                    for model in _WEB_RESEARCH_FALLBACK_ORDER
                    if model in eligible_candidates and model not in scored_models
                ],
                *[
                    model
                    for model in eligible_candidates
                    if model not in scored_models and model not in _WEB_RESEARCH_FALLBACK_ORDER
                ],
            ]
        else:
            ranked = [
                *scored_models,
                *[model for model in eligible_candidates if model not in scored_models],
            ]

        model = ranked[0] if ranked else base["model"]
        selected_tier_configs: dict[str, TierConfig] = {
            **tier_configs,
            target_tier: {"primary": model, "fallback": ranked[1:]},
        }
        # select_model only reads the selected tier; retain the complete tier map
        # for host fallback.
        decision = select_model(
            target_tier,
            base["confidence"],
            "portfolio",
            f"{base['reasoning']} | v3 task={features.task_type}"
            f" agentRisk={features.agent_risk}"
            f" deepWebResearch={js_bool(features.deep_web_research)}"
            f" terminalCode={js_bool(features.implicit_terminal_code)}"
            f" terminalSafety={js_bool(features.terminal_safety_sensitive)}"
            f" candidates={len(ranked)}",
            selected_tier_configs,
            options["model_pricing"],
            features.estimated_input_tokens,
            max_output_tokens,
            routing_profile,
            base.get("agentic_score"),
        )
        decision["tier_configs"] = selected_tier_configs
        profile_value = base.get("profile")
        if profile_value is not None:
            decision["profile"] = profile_value
        decision["candidates"] = ranked
        decision["candidate_scores"] = ranked_entries
        decision["task_type"] = features.task_type
        decision["router_version"] = "v3-portfolio"
        return decision
