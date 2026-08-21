"""
Parity tests for the Router Core port.

Every case here is a 1:1 port of an upstream ``@blockrun/router-core`` vitest
case (``portfolio.test.ts``, ``selector.test.ts``, ``strategy.test.ts``,
``tool-intent.test.ts``, ``unavailable-models.test.ts`` at commit
``d7bc10c``). They are the regression guard
that the Python port keeps choosing the same models as the TypeScript SDK —
when upstream is re-synced, re-port these alongside the source.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from blockrun_llm.router_core import (
    DEFAULT_ROUTING_CONFIG,
    RulesStrategy,
    apply_unavailable_models,
    calculate_model_cost,
    filter_by_exclude_list,
    filter_by_tool_calling,
    filter_candidates_by_capacity,
    get_strategy,
    infer_tool_requirement,
    register_strategy,
    route,
)
from blockrun_llm.router_core.selector import select_model


def _price(input_price: float, output_price: float) -> dict[str, float]:
    return {"input_price": input_price, "output_price": output_price}


PORTFOLIO_PRICING = {
    "anthropic/claude-sonnet-4.6": _price(3, 15),
    "anthropic/claude-sonnet-5": _price(3, 15),
    "anthropic/claude-opus-5": _price(5, 25),
    "anthropic/claude-opus-4.8": _price(5, 25),
    "openai/gpt-5.3-codex": _price(1.75, 14),
    "openai/gpt-5-mini": _price(0.25, 2),
    "openai/gpt-4.1": _price(2, 8),
    "google/gemini-3.5-flash": _price(0.5, 3),
    "google/gemini-3-flash-preview": _price(0.5, 3),
    "google/gemini-3.1-pro": _price(2, 12),
    "moonshot/kimi-k3": _price(3, 15),
    "deepseek/deepseek-v4-pro": _price(0.435, 0.87),
    "xai/grok-4.5": _price(2, 10),
    "qwen/qwen3.7-max": _price(1.475, 4.425),
    "zai/glm-5.2": _price(1.4, 4.4),
    "moonshot/kimi-k2.7": _price(0.95, 4),
    "moonshot/kimi-k2.6": _price(0.95, 4),
    "moonshot/kimi-k2.5": _price(0.6, 3),
    "xai/grok-4-1-fast-non-reasoning": _price(0.2, 0.5),
    "openai/gpt-4o-mini": _price(0.15, 0.6),
    "deepseek/deepseek-chat": _price(0.2, 0.4),
    "free/seed-oss-36b": _price(0, 0),
}

STRATEGY_PRICING = {
    "moonshot/kimi-k2.5": _price(0.5, 2.4),
    "moonshot/kimi-k2.6": _price(0.95, 4.0),
    "anthropic/claude-opus-4.6": _price(5, 25),
    "anthropic/claude-opus-4.7": _price(5, 25),
    "anthropic/claude-opus-4.8": _price(5, 25),
    "google/gemini-2.5-flash": _price(0.15, 0.6),
    "google/gemini-2.5-flash-lite": _price(0.1, 0.4),
    "deepseek/deepseek-chat": _price(0.14, 0.28),
    "anthropic/claude-sonnet-4.6": _price(3, 15),
    "google/gemini-3.1-pro": _price(1.25, 10),
    "google/gemini-3.5-flash": _price(0.5, 3),
    "xai/grok-4.5": _price(2.5, 9),
    "anthropic/claude-sonnet-5": _price(3, 15),
    "deepseek/deepseek-v4-pro": _price(0.435, 0.87),
    "moonshot/kimi-k3": _price(3, 15),
    "xai/grok-4-1-fast-reasoning": _price(0.2, 0.5),
    "nvidia/gpt-oss-120b": _price(0, 0),
    "nvidia/gpt-oss-20b": _price(0, 0),
    "nvidia/deepseek-v3.2": _price(0, 0),
    "nvidia/deepseek-v4-pro": _price(0, 0),
    "nvidia/deepseek-v4-flash": _price(0, 0),
    "nvidia/qwen3-coder-480b": _price(0, 0),
    "nvidia/glm-4.7": _price(0, 0),
    "nvidia/llama-4-maverick": _price(0, 0),
    "nvidia/qwen3-next-80b-a3b-thinking": _price(0, 0),
    "nvidia/mistral-small-4-119b": _price(0, 0),
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": _price(0, 0),
    "nvidia/qwen3-next-80b-a3b-instruct": _price(0, 0),
    "nvidia/seed-oss-36b": _price(0, 0),
    "nvidia/mistral-nemotron": _price(0, 0),
    "nvidia/step-3.7-flash": _price(0, 0),
    "nvidia/nemotron-nano-9b-v2": _price(0, 0),
    "nvidia/nemotron-nano-12b-v2-vl": _price(0, 0),
}

BASE_OPTIONS = {"config": DEFAULT_ROUTING_CONFIG, "model_pricing": STRATEGY_PRICING}

TERMINAL_TOOLS = ["TerminalExec", "TerminalInspect", "TerminalSendKeys"]

AIRLINE_TOOLS = [
    "get_user_details",
    "get_reservation_details",
    "search_direct_flight",
    "update_reservation_flights",
    "cancel_reservation",
    "book_reservation",
    "update_reservation_baggages",
]

RETURN_TOOLS = [
    "get_order_details",
    "return_delivered_order_items",
    "transfer_to_human_agents",
]

KIMI_MODELS = ("moonshot/kimi-k2.7", "moonshot/kimi-k2.6", "moonshot/kimi-k2.5")


def _portfolio(prompt: str, max_output_tokens: int, **options):
    return route(
        prompt,
        None,
        max_output_tokens,
        {"config": DEFAULT_ROUTING_CONFIG, "model_pricing": PORTFOLIO_PRICING, **options},
    )


def _terminal(prompt: str, max_output_tokens: int = 4096):
    return _portfolio(
        prompt,
        max_output_tokens,
        routing_profile="auto",
        has_tools=True,
        requires_tools=True,
        tool_count=3,
        tool_names=TERMINAL_TOOLS,
    )


def _scored_models(decision) -> list[str]:
    return [row["model"] for row in decision.get("candidate_scores", [])]


# ─── portfolio.test.ts ───


class TestPortfolioStrategy:
    def test_keeps_only_tool_capable_models_for_a_coding_agent_request(self):
        decision = _portfolio(
            "Fix the TypeScript payment retry bug, run tests, and update the patch.",
            4096,
            has_tools=True,
        )

        assert decision["method"] == "portfolio"
        assert decision["task_type"] == "code_agent"
        assert decision["model"] == "openai/gpt-5-mini"
        assert decision["model"] in decision["candidates"]
        assert "openai/gpt-5.3-codex" in decision["candidates"]
        assert decision["model"] not in KIMI_MODELS
        assert "google/gemini-3.1-pro" not in decision["candidates"]

    def test_classifies_a_non_code_function_call_as_a_tool_agent(self):
        decision = _portfolio("Use the lookup_order tool for order B-42.", 256, has_tools=True)

        assert decision["task_type"] in ("tool_agent", "tool_agent_parallel")
        assert decision["model"] == "anthropic/claude-sonnet-5"
        assert decision["model"] in decision["candidates"]
        assert "google/gemini-3.5-flash" in decision["candidates"]
        assert decision["model"] not in KIMI_MODELS

    @pytest.mark.parametrize(
        "prompt",
        [
            "请问北京的当前天气状况如何？还有，上海的天气情况是怎样的？",
            (
                "For breakfast I had a 12 ounce iced coffee and a banana.\n\n"
                "For lunch I had a quesadilla.\n\n"
                "Breakfast four ounces of asparagus and two eggs."
            ),
            "¿Cuáles son las condiciones del clima en Cancún, Playa del Carmen y Tulum?",
            "Could you tell me the current temperature in Boston, MA and San Francisco, please?",
            "What's the snow like in the two cities of Paris and Bordeaux?",
            "What's cost of 2 and 4 gb ram machine on aws ec2 with one CPU?",
            "能帮我查一下中国广州市和北京市现在的天气状况吗？请使用公制单位。",
            (
                "Could you provide the latest news for Paris, France, and also for "
                "Letterkenny, Ireland?"
            ),
            "I'd like to change my food order to a salad, and for the drink, update it to coffee.",
        ],
    )
    def test_routes_repeated_single_tool_requests_to_the_parallel_specialist(self, prompt):
        decision = _portfolio(
            prompt,
            600,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=1,
        )

        assert decision["task_type"] == "tool_agent_parallel"
        assert decision["model"] == "anthropic/claude-opus-4.8"

    def test_keeps_an_ordinary_single_lookup_on_the_standard_tool_agent_path(self):
        decision = _portfolio(
            "Use lookup_order for order B-42.",
            256,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=1,
        )

        assert decision["task_type"] == "tool_agent"
        assert decision["model"] == "anthropic/claude-sonnet-5"
        assert "google/gemini-3.5-flash" in decision["candidates"]

    def test_keeps_deep_multi_clue_web_research_on_sonnet_5(self):
        decision = _portfolio(
            "Research the following clues across multiple public sources and identify the country.",
            2048,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=2,
            tool_names=["web_search", "web_fetch"],
        )

        assert decision["task_type"] in ("tool_agent", "tool_agent_parallel")
        assert decision["model"] == "anthropic/claude-sonnet-5"
        assert "deepWebResearch=true" in decision["reasoning"]
        assert decision["candidates"][:3] == [
            "anthropic/claude-sonnet-5",
            "openai/gpt-5-mini",
            "google/gemini-3.5-flash",
        ]
        assert "candidates=" in decision["reasoning"]

    def test_keeps_a_routine_web_lookup_on_sonnet_5(self):
        decision = _portfolio(
            "Search the official documentation for the current API timeout setting.",
            1024,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=2,
            tool_names=["web_search", "web_fetch"],
        )

        assert decision["model"] == "anthropic/claude-sonnet-5"
        assert "deepWebResearch=false" in decision["reasoning"]

    def test_keeps_a_known_cross_reservation_batch_on_the_cost_controlled_model(self):
        decision = _portfolio(
            "Hi! I’d like to make some changes to my bookings. I need to cancel two of my "
            "upcoming reservations and upgrade another one to business class. "
            "Can you help me with that?",
            4096,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=7,
            tool_names=AIRLINE_TOOLS,
        )

        assert decision["task_type"] in ("tool_agent", "tool_agent_parallel")
        assert "agentRisk=high" in decision["reasoning"]
        assert decision["model"] == "openai/gpt-5-mini"

    def test_promotes_conditional_global_airline_work_to_the_complex_band(self):
        decision = _portfolio(
            "Cancel all your future reservations that contain flights longer than 4 hours. "
            "For flights under 3 hours, upgrade to business wherever possible.",
            4096,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=7,
            tool_names=AIRLINE_TOOLS,
        )

        assert "agentRisk=complex_high" in decision["reasoning"]
        assert decision["model"] == "anthropic/claude-sonnet-5"

    @pytest.mark.parametrize(
        "prompt",
        [
            (
                "Create a file called hello.txt in the current directory. "
                "Write Hello, world! to it and end with a newline."
            ),
            "Convert the file /app/data.csv into a Parquet file named /app/data.parquet.",
            (
                "Create and run a server on port 3000 with a single GET endpoint /fib "
                "that returns JSON."
            ),
            (
                "A script called 'process_data.sh' in the current directory won't run. "
                "Figure out what's wrong and fix it so the script can run successfully."
            ),
        ],
    )
    def test_uses_the_low_cost_code_agent_for_deterministic_local_terminal_work(self, prompt):
        decision = _terminal(prompt)

        assert decision["task_type"] == "code_agent"
        assert decision["model"] == "openai/gpt-5-mini"
        assert "terminalCode=true" in decision["reasoning"]

    def test_promotes_a_multi_script_dependency_repair_to_the_strong_band(self):
        decision = _terminal(
            "There's a data processing pipeline in the current directory consisting of "
            "multiple scripts that need to run in sequence. The main script 'run_pipeline.sh' "
            "is failing to execute properly. Identify and fix all issues with the script files "
            "and dependencies to make the pipeline run successfully."
        )

        assert decision["task_type"] == "tool_agent"
        assert "agentRisk=complex_high" in decision["reasoning"]
        assert decision["model"] == "anthropic/claude-sonnet-5"

    def test_promotes_a_cross_runtime_polyglot_artifact_to_the_strong_band(self):
        decision = _terminal(
            "Write one /app/main.c.rs polyglot file that must compile and run with both "
            "rustc main.c.rs and gcc main.c.rs -o cmain."
        )

        assert decision["task_type"] == "code_agent"
        assert "agentRisk=complex_high" in decision["reasoning"]
        assert decision["model"] == "anthropic/claude-sonnet-5"

    def test_promotes_a_framework_checkpoint_port_to_the_strong_band(self):
        decision = _terminal(
            "Implement a command line tool programmed in C that runs inference using a "
            "pre-trained PyTorch state_dict called simple_mnist.pth. The final output must be "
            "a native cli_tool binary plus weights.json."
        )

        assert decision["task_type"] == "code_agent"
        assert "agentRisk=complex_high" in decision["reasoning"]
        assert decision["model"] == "anthropic/claude-sonnet-5"

    @pytest.mark.parametrize(
        "prompt",
        [
            (
                "Configure a git server over SSH and deploy two branches through Nginx HTTPS "
                "with password authentication."
            ),
            (
                "Securely decommission the service: encrypt the archive with GPG, shred the "
                "sensitive files, then delete them."
            ),
            (
                "Evaluate an embedding model with the MTEB benchmark and write the official "
                "result file."
            ),
            "Inspect the chess board image and write the best move to a file.",
            (
                "Create a JSON processor from three CSV inputs. Requirements: 1. Follow "
                "schema.json. 2. Join departments and employees. 3. Calculate statistics."
            ),
        ],
    )
    def test_keeps_complex_or_risky_terminal_operations_on_the_generic_agent_path(self, prompt):
        decision = _terminal(prompt)

        assert decision["task_type"] != "code_agent"
        assert "terminalCode=false" in decision["reasoning"]

    def test_keeps_codex_below_the_primary_band_for_security_sensitive_file_ops(self):
        decision = _terminal(
            "Please help me encrypt all the files I have in the data/ folder using rencrypt. "
            "Use the most secure encryption and write the outputs to encrypted_data/ with the "
            "same basenames."
        )

        assert decision["task_type"] == "tool_agent"
        assert "terminalSafety=true" in decision["reasoning"]
        assert decision["model"] == "anthropic/claude-sonnet-5"
        assert "openai/gpt-5.3-codex" in decision["candidates"]
        assert "openai/gpt-5.3-codex" not in _scored_models(decision)

    def test_admits_a_cost_controlled_strong_model_for_sensitive_multi_file_work(self):
        decision = _terminal(
            "Sanitize this git repository by replacing all AWS, GitHub, and Hugging Face API "
            "keys with consistent placeholders across every affected file. Also, do not make "
            "any other unnecessary changes to files without sensitive information."
        )

        assert decision["task_type"] == "tool_agent_parallel"
        assert decision["model"] == "anthropic/claude-sonnet-5"
        assert "anthropic/claude-sonnet-5" in decision["candidates"]

    @pytest.mark.parametrize(
        "prompt",
        [
            (
                "Reverse engineer the mystery binary, then write and compile image.c so it "
                "produces the requested path-traced image."
            ),
            (
                "Create a local JSON server for Solana devnet with status, block, account, "
                "transaction, and paginated program-account endpoints."
            ),
            (
                "Create a Solana devnet API whose transaction endpoint returns token transfers "
                "with account, mint, and amount fields."
            ),
        ],
    )
    def test_cost_controls_complex_terminal_work_that_is_not_safety_sensitive(self, prompt):
        decision = _terminal(prompt)

        assert decision["task_type"] in ("tool_agent", "tool_agent_parallel", "code_agent")
        assert decision["model"] == "openai/gpt-5-mini"
        assert "terminalSafety=false" in decision["reasoning"]

    @pytest.mark.parametrize(
        "prompt",
        [
            (
                "Rotate the expired authentication token and update the bearer token used by "
                "the production service."
            ),
            (
                "Replace every leaked API key and password in this repository without changing "
                "unrelated files."
            ),
        ],
    )
    def test_keeps_credential_bearing_terminal_work_safety_sensitive(self, prompt):
        decision = _terminal(prompt)

        assert "terminalSafety=true" in decision["reasoning"]
        assert decision["model"] != "openai/gpt-5-mini"

    def test_uses_the_high_risk_model_for_retail_order_tools(self):
        decision = _portfolio(
            "Exchange both items after I confirm the price difference.",
            512,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=4,
            tool_names=[
                "get_order_details",
                "get_product_details",
                "exchange_delivered_order_items",
                "modify_pending_order_address",
            ],
        )

        assert decision["task_type"] in ("tool_agent", "tool_agent_parallel")
        assert decision["model"] == "deepseek/deepseek-v4-pro"
        assert "openai/gpt-5-mini" in decision["candidates"]

    def test_uses_the_low_cost_model_for_one_local_retail_operation(self):
        decision = _portfolio(
            "Change the blue earbuds in order W5061109 to red after I confirm.",
            4096,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=6,
            tool_names=[
                "find_user_id_by_name_zip",
                "get_order_details",
                "get_product_details",
                "modify_pending_order_items",
            ],
        )

        assert decision["task_type"] == "tool_agent"
        assert decision["model"] == "openai/gpt-5-mini"

    def test_keeps_global_retail_choices_on_the_high_risk_model(self):
        decision = _portfolio(
            "Exchange my tablet for the cheapest available variant in another order.",
            4096,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=6,
            tool_names=[
                "get_order_details",
                "get_product_details",
                "exchange_delivered_order_items",
            ],
        )

        assert decision["model"] == "deepseek/deepseek-v4-pro"

    def test_uses_the_policy_specialist_for_a_refund_to_another_card(self):
        decision = _portfolio(
            "Return everything except the pet bed and refund it to my Amex card.",
            4096,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=6,
            tool_names=RETURN_TOOLS,
        )

        assert decision["task_type"] in ("tool_agent", "tool_agent_parallel")
        assert decision["model"] == "openai/gpt-4.1"
        assert "agentRisk=policy_exception" in decision["reasoning"]

    def test_keeps_a_single_comparative_send_back_on_the_low_cost_model(self):
        decision = _portfolio(
            "Send back the pricier one and get my money back on my credit card.",
            4096,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=6,
            tool_names=RETURN_TOOLS,
        )

        assert decision["model"] == "openai/gpt-5-mini"
        assert "agentRisk=policy_exception_simple" in decision["reasoning"]

    def test_uses_the_policy_specialist_when_a_named_card_refund_covers_two_objects(self):
        decision = _portfolio(
            "Return these two skateboards and refund them to my credit card.",
            4096,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=6,
            tool_names=RETURN_TOOLS,
        )

        assert decision["model"] == "openai/gpt-4.1"
        assert "agentRisk=policy_exception" in decision["reasoning"]

    def test_treats_a_simple_looking_retail_return_as_a_negotiated_high_risk_workflow(self):
        decision = _portfolio(
            "I want to return an office chair that arrived broken.",
            4096,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=6,
            tool_names=[
                "get_order_details",
                "get_product_details",
                "return_delivered_order_items",
                "exchange_delivered_order_items",
            ],
        )

        assert decision["task_type"] == "tool_agent"
        assert decision["model"] == "deepseek/deepseek-v4-pro"
        assert "agentRisk=high" in decision["reasoning"]

    def test_uses_the_cost_efficient_model_for_airline_tools(self):
        decision = _portfolio(
            "Change my flight after checking the reservation.",
            512,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=3,
            tool_names=[
                "get_reservation_details",
                "search_direct_flight",
                "update_reservation_flights",
            ],
        )

        assert decision["task_type"] in ("tool_agent", "tool_agent_parallel")
        assert decision["model"] == "openai/gpt-5-mini"
        assert "anthropic/claude-sonnet-5" in decision["candidates"]

    def test_does_not_mistake_airline_cabin_class_for_a_code_agent_task(self):
        decision = _portfolio(
            "Move my flight to May 24 and upgrade all passengers to business class.",
            4096,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=8,
            tool_names=[
                "get_reservation_details",
                "search_direct_flight",
                "update_reservation_flights",
            ],
        )

        assert decision["task_type"] != "code_agent"
        assert decision["model"] == "openai/gpt-5-mini"
        assert "agentRisk=high" in decision["reasoning"]

    def test_reserves_the_airline_specialist_for_global_itinerary_optimization(self):
        decision = _portfolio(
            "Show my gift card and certificate balances, then change my reservation to the "
            "cheapest business round trip without changing the dates.",
            4096,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=8,
            tool_names=[
                "get_user_details",
                "get_reservation_details",
                "search_onestop_flight",
                "cancel_reservation",
                "book_reservation",
            ],
        )

        assert decision["task_type"] != "code_agent"
        assert decision["model"] == "anthropic/claude-sonnet-5"
        assert "agentRisk=complex_high" in decision["reasoning"]

    def test_does_not_mistake_a_lookup_plus_explanation_for_parallel_tool_use(self):
        decision = _portfolio(
            "Get the weather for London and explain whether I need an umbrella.",
            256,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=1,
            tool_names=["get_current_weather"],
        )

        assert decision["task_type"] == "tool_agent"

    def test_uses_two_distinctive_visible_tool_names_as_a_multi_operation_signal(self):
        decision = _portfolio(
            "Add task draft release notes, then delete task obsolete draft.",
            256,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=2,
            tool_names=["add_task", "delete_task"],
        )

        assert decision["task_type"] == "tool_agent_parallel"

    def test_does_not_spend_upgrade_a_large_numbered_multi_tool_plan(self):
        decision = _portfolio(
            "Do all the following:\n1. Clone the repository.\n2. Analyze it.\n"
            "3. Create Docker and Kubernetes files.\n4. Commit and push.",
            600,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=7,
            tool_names=[
                "clone_repo",
                "analyze_repo",
                "create_docker_file",
                "create_kubernetes_yaml",
                "commit_changes",
                "push_changes",
                "read_file",
            ],
        )

        assert decision["task_type"] != "tool_agent_parallel"
        assert decision["model"] != "anthropic/claude-opus-4.8"

    def test_detects_an_explicit_multi_object_request_with_a_distractor_tool(self):
        decision = _portfolio(
            "What's the weather like in the two cities of Boston and San Francisco?",
            600,
            routing_profile="auto",
            has_tools=True,
            requires_tools=True,
            tool_count=2,
        )

        assert decision["task_type"] == "tool_agent_parallel"
        assert decision["model"] == "anthropic/claude-opus-4.8"

    def test_does_not_classify_ordinary_qa_as_a_tool_task(self):
        decision = _portfolio(
            "Which answer is correct?\nA. One\nB. Two\nC. Three\nD. Four",
            256,
            has_tools=True,
            requires_tools=False,
        )

        assert decision["task_type"] == "reasoning_mcq"
        assert decision["profile"] == "auto"
        assert decision["model"] == "google/gemini-3-flash-preview"

    def test_adds_current_long_context_models_instead_of_a_legacy_tier_chain(self):
        decision = _portfolio("A" * 340_000, 1_024)

        assert decision["task_type"] == "long_context"
        assert "deepseek/deepseek-v4-pro" not in _scored_models(decision)
        assert "deepseek/deepseek-v4-pro" in decision["candidates"]
        assert decision["model"] == "google/gemini-3.1-pro"
        assert decision["model"] in decision["candidates"]

    def test_keeps_mandarin_extraction_in_the_source_language_affinity_band(self):
        decision = _portfolio(
            "只输出 JSON：从订单 A-17，数量 3，状态已发货中提取 orderId、quantity、status 三个字段。",
            256,
        )

        assert decision["task_type"] == "extraction"
        assert decision["model"] == "moonshot/kimi-k2.7"
        assert decision["candidates"][0] == "moonshot/kimi-k2.7"

    def test_does_not_promote_a_generic_recovery_fallback_without_task_affinity(self):
        decision = _portfolio("Patch this API secret validation error.", 256)

        # DeepSeek Chat is a valid availability fallback in the SIMPLE tier, but
        # is not an explicitly profiled code-edit specialist. It must not win the
        # Auto ranking simply because it is inexpensive.
        assert "deepseek/deepseek-chat" not in _scored_models(decision)
        assert "deepseek/deepseek-chat" in decision["candidates"]

    def test_does_not_let_a_flash_lite_sibling_inherit_flash_task_affinity(self):
        exact_name_config = {
            **DEFAULT_ROUTING_CONFIG,
            "tiers": {
                tier: {
                    "primary": "google/gemini-2.5-flash",
                    "fallback": ["google/gemini-2.5-flash-lite"],
                }
                for tier in DEFAULT_ROUTING_CONFIG["tiers"]
            },
        }
        decision = route(
            "Explain the deployment status.",
            None,
            256,
            {
                "config": exact_name_config,
                "model_pricing": {
                    "google/gemini-2.5-flash": _price(1, 1),
                    "google/gemini-2.5-flash-lite": _price(0.1, 0.1),
                },
            },
        )

        assert decision["candidates"][0] == "google/gemini-2.5-flash"
        assert "google/gemini-2.5-flash-lite" in decision["candidates"]
        assert "google/gemini-2.5-flash-lite" not in _scored_models(decision)

    def test_filters_models_that_cannot_satisfy_the_requested_output_length(self):
        decision = _portfolio("Explain this architecture", 20_000)

        assert "xai/grok-4-fast-non-reasoning" not in decision["candidates"]

    def test_only_lets_fresh_performance_observations_influence_candidate_order(self):
        two_candidate_config = {
            **DEFAULT_ROUTING_CONFIG,
            "tiers": {
                tier: {
                    "primary": "xai/grok-4-1-fast-non-reasoning",
                    "fallback": ["openai/gpt-4o-mini"],
                }
                for tier in DEFAULT_ROUTING_CONFIG["tiers"]
            },
        }
        decision = route(
            "Extract the fields as JSON",
            None,
            512,
            {
                "config": two_candidate_config,
                "model_pricing": {
                    "xai/grok-4-1-fast-non-reasoning": _price(1, 1),
                    "openai/gpt-4o-mini": _price(1, 1),
                },
                "now": datetime(2026, 7, 21, tzinfo=timezone.utc),
                "model_performance": {
                    "openai/gpt-4o-mini": {
                        "measured_at": "2026-07-21T00:00:00Z",
                        "latency_ms": 600,
                        "output_tokens_per_second": 250,
                        "intelligence_index": 50,
                    }
                },
            },
        )

        assert decision["task_type"] == "extraction"
        assert decision["candidates"][0] == "openai/gpt-4o-mini"

    def test_treats_a_small_performance_probe_as_a_tie_breaker(self):
        two_candidate_config = {
            **DEFAULT_ROUTING_CONFIG,
            "tiers": {
                tier: {
                    "primary": "xai/grok-4-1-fast-non-reasoning",
                    "fallback": ["openai/gpt-4o-mini"],
                }
                for tier in DEFAULT_ROUTING_CONFIG["tiers"]
            },
        }
        decision = route(
            "Explain the deployment status.",
            None,
            512,
            {
                "config": two_candidate_config,
                "model_pricing": {
                    "xai/grok-4-1-fast-non-reasoning": _price(1, 1),
                    "openai/gpt-4o-mini": _price(1, 1),
                },
                "now": datetime(2026, 7, 21, tzinfo=timezone.utc),
                "model_performance": {
                    "openai/gpt-4o-mini": {
                        "measured_at": "2026-07-21T00:00:00Z",
                        "latency_ms": 600,
                        "output_tokens_per_second": 250,
                        "intelligence_index": 50,
                        "samples": 1,
                    }
                },
            },
        )

        assert decision["candidates"][0] == "xai/grok-4-1-fast-non-reasoning"

    def test_ignores_a_malformed_performance_timestamp(self):
        decision = _portfolio(
            "Extract the fields as JSON",
            512,
            now=datetime(2026, 7, 21, tzinfo=timezone.utc),
            model_performance={
                "openai/gpt-4o-mini": {
                    "measured_at": "not-a-timestamp",
                    "latency_ms": 1,
                    "output_tokens_per_second": 10_000,
                    "intelligence_index": 50,
                }
            },
        )

        assert all(math.isfinite(row["score"]) for row in decision.get("candidate_scores", []))

    def test_falls_back_to_the_rules_decision_when_a_tier_has_no_usable_candidate(self):
        empty_tiers = {
            tier: {"primary": "", "fallback": []} for tier in DEFAULT_ROUTING_CONFIG["tiers"]
        }
        decision = route(
            "hello",
            None,
            128,
            {
                "config": {**DEFAULT_ROUTING_CONFIG, "tiers": empty_tiers},
                "model_pricing": PORTFOLIO_PRICING,
            },
        )

        assert decision["method"] == "rules"
        assert decision["model"] == ""

    def test_lets_a_host_capability_snapshot_override_the_built_in_catalog(self):
        decision = _portfolio(
            "Use the lookup_order tool for order B-42.",
            256,
            has_tools=True,
            requires_tools=True,
            model_capabilities={
                "anthropic/claude-sonnet-5": {
                    "context_window": 1_000_000,
                    "max_output_tokens": 128_000,
                    "supports_tools": False,
                    "supports_vision": True,
                }
            },
        )

        assert "anthropic/claude-sonnet-5" not in decision["candidates"]


# ─── selector.test.ts ───

SELECTOR_TIER_CONFIGS = {
    tier: {"primary": "moonshot/kimi-k2.5", "fallback": []}
    for tier in ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")
}
SELECTOR_PRICING = {
    "moonshot/kimi-k2.5": _price(0.5, 2.4),
    "anthropic/claude-opus-4.7": _price(5, 25),
    "anthropic/claude-opus-4.8": _price(5, 25),
}


def _supports_tool_calling(model: str) -> bool:
    return model not in ("minimax/minimax-m2.5", "nvidia/gpt-oss-120b")


class TestSelector:
    def test_select_model_uses_opus_4_7_as_the_savings_baseline(self):
        decision = select_model(
            "SIMPLE",
            0.95,
            "rules",
            "test",
            SELECTOR_TIER_CONFIGS,
            SELECTOR_PRICING,
            1000,
            1000,
        )

        assert decision["baseline_cost"] > 0
        assert decision["savings"] > 0

    def test_calculate_model_cost_uses_opus_4_7_as_the_baseline(self):
        costs = calculate_model_cost("moonshot/kimi-k2.5", SELECTOR_PRICING, 1000, 1000)

        assert costs["baseline_cost"] > 0
        assert costs["savings"] > 0

    def test_filter_by_tool_calling_removes_models_without_tool_support(self):
        models = ["moonshot/kimi-k2.5", "minimax/minimax-m2.5", "deepseek/deepseek-chat"]

        assert filter_by_tool_calling(models, True, _supports_tool_calling) == [
            "moonshot/kimi-k2.5",
            "deepseek/deepseek-chat",
        ]

    def test_filter_by_tool_calling_keeps_every_model_when_the_request_has_no_tools(self):
        models = ["moonshot/kimi-k2.5", "minimax/minimax-m2.5", "nvidia/gpt-oss-120b"]

        assert filter_by_tool_calling(models, False, _supports_tool_calling) == models

    def test_filter_by_tool_calling_never_returns_an_empty_chain(self):
        unsupported = ["minimax/minimax-m2.5", "nvidia/gpt-oss-120b"]

        assert filter_by_tool_calling(unsupported, True, _supports_tool_calling) == unsupported

    def test_filter_by_exclude_list(self):
        chain = ["moonshot/kimi-k2.5", "deepseek/deepseek-chat", "anthropic/claude-sonnet-4.6"]

        assert filter_by_exclude_list(chain, {"deepseek/deepseek-chat"}) == [
            "moonshot/kimi-k2.5",
            "anthropic/claude-sonnet-4.6",
        ]
        assert filter_by_exclude_list(chain, set(chain)) == chain
        assert filter_by_exclude_list(chain, set()) == chain

    def test_filter_candidates_by_capacity(self):
        capabilities = {
            "small": {"context_window": 8_000, "max_output": 2_000},
            "large": {"context_window": 128_000, "max_output": 32_000},
        }

        assert filter_candidates_by_capacity(
            ["small", "large"], 10_000, 4_000, capabilities.get
        ) == ["large"]
        assert filter_candidates_by_capacity(["small"], 100_000, 40_000, capabilities.get) == []


# ─── strategy.test.ts ───


class TestRulesStrategy:
    def test_returns_tier_configs_in_the_decision(self):
        decision = RulesStrategy().route("hello", None, 100, BASE_OPTIONS)

        assert decision["tier_configs"] is not None
        for tier in ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING"):
            assert tier in decision["tier_configs"]

    def test_returns_profile_in_the_decision(self):
        decision = RulesStrategy().route("hello", None, 100, BASE_OPTIONS)

        assert decision["profile"] in ("auto", "eco", "premium", "agentic")

    def test_honors_the_protocol_structured_output_requirement(self):
        decision = RulesStrategy().route(
            "hello", None, 100, {**BASE_OPTIONS, "requires_structured_output": True}
        )

        assert decision["tier"] == "MEDIUM"
        assert "structured output" in decision["reasoning"]

    def test_sets_eco_profile_when_routing_profile_is_eco(self):
        decision = RulesStrategy().route(
            "hello", None, 100, {**BASE_OPTIONS, "routing_profile": "eco"}
        )

        assert decision["profile"] == "eco"
        assert decision["tier_configs"] == DEFAULT_ROUTING_CONFIG["eco_tiers"]

    def test_sets_premium_profile_when_routing_profile_is_premium(self):
        decision = RulesStrategy().route(
            "hello", None, 100, {**BASE_OPTIONS, "routing_profile": "premium"}
        )

        assert decision["profile"] == "premium"
        assert decision["tier_configs"] == DEFAULT_ROUTING_CONFIG["premium_tiers"]

    def test_eco_tiers_none_falls_back_to_regular_tiers_without_dropping_into_auto(self):
        decision = RulesStrategy().route(
            "hello",
            None,
            100,
            {
                **BASE_OPTIONS,
                "config": {**DEFAULT_ROUTING_CONFIG, "eco_tiers": None},
                "routing_profile": "eco",
                "has_tools": True,
                "now": datetime(2025, 1, 1, tzinfo=timezone.utc),
            },
        )

        assert decision["profile"] == "eco"
        assert decision["tier_configs"] == DEFAULT_ROUTING_CONFIG["tiers"]

    def test_premium_tiers_none_falls_back_to_regular_tiers(self):
        decision = RulesStrategy().route(
            "hello",
            None,
            100,
            {
                **BASE_OPTIONS,
                "config": {**DEFAULT_ROUTING_CONFIG, "premium_tiers": None},
                "routing_profile": "premium",
                "has_tools": True,
                "now": datetime(2025, 1, 1, tzinfo=timezone.utc),
            },
        )

        assert decision["profile"] == "premium"
        assert decision["tier_configs"] == DEFAULT_ROUTING_CONFIG["tiers"]

    def test_sets_agentic_profile_when_tools_are_present(self):
        decision = RulesStrategy().route("hello", None, 100, {**BASE_OPTIONS, "has_tools": True})

        assert decision["profile"] == "agentic"
        assert decision["tier_configs"] == DEFAULT_ROUTING_CONFIG["agentic_tiers"]

    def test_sets_auto_profile_for_default_requests(self):
        decision = RulesStrategy().route(
            "what is the capital of France",
            None,
            100,
            {**BASE_OPTIONS, "now": datetime(2025, 1, 1, tzinfo=timezone.utc)},
        )

        assert decision["profile"] == "auto"
        assert decision["tier_configs"] == DEFAULT_ROUTING_CONFIG["tiers"]

    def test_agentic_mode_false_disables_agentic_tiers_even_with_tools(self):
        config = {
            **DEFAULT_ROUTING_CONFIG,
            "overrides": {**DEFAULT_ROUTING_CONFIG["overrides"], "agentic_mode": False},
        }
        decision = RulesStrategy().route(
            "hello",
            None,
            100,
            {
                **BASE_OPTIONS,
                "config": config,
                "has_tools": True,
                "now": datetime(2025, 1, 1, tzinfo=timezone.utc),
            },
        )

        assert decision["profile"] == "auto"
        assert decision["tier_configs"] == DEFAULT_ROUTING_CONFIG["tiers"]

    def test_agentic_mode_true_forces_agentic_tiers_even_without_tools(self):
        config = {
            **DEFAULT_ROUTING_CONFIG,
            "overrides": {**DEFAULT_ROUTING_CONFIG["overrides"], "agentic_mode": True},
        }
        decision = RulesStrategy().route(
            "hello",
            None,
            100,
            {
                **BASE_OPTIONS,
                "config": config,
                "has_tools": False,
                "now": datetime(2025, 1, 1, tzinfo=timezone.utc),
            },
        )

        assert decision["profile"] == "agentic"
        assert decision["tier_configs"] == DEFAULT_ROUTING_CONFIG["agentic_tiers"]


class TestStrategyRegistry:
    def test_retrieves_the_default_rules_strategy(self):
        strategy = get_strategy("rules")

        assert isinstance(strategy, RulesStrategy)
        assert strategy.name == "rules"

    def test_raises_for_an_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown routing strategy: nonexistent"):
            get_strategy("nonexistent")

    def test_registers_and_retrieves_a_custom_strategy(self):
        class CustomStrategy:
            name = "custom-test"

            def route(self, prompt, system_prompt, max_output_tokens, options):
                return {
                    "model": "test/model",
                    "tier": "SIMPLE",
                    "confidence": 1,
                    "method": "rules",
                    "reasoning": "custom strategy",
                    "cost_estimate": 0,
                    "baseline_cost": 0,
                    "savings": 0,
                    "tier_configs": options["config"]["tiers"],
                    "profile": "auto",
                }

        register_strategy(CustomStrategy())
        retrieved = get_strategy("custom-test")

        assert retrieved.name == "custom-test"
        decision = retrieved.route("test", None, 100, BASE_OPTIONS)
        assert decision["model"] == "test/model"
        assert decision["reasoning"] == "custom strategy"


class TestPortfolioDefault:
    def test_route_uses_the_v3_portfolio_while_retaining_rule_tiers(self):
        simple = route("hello", None, 100, BASE_OPTIONS)

        assert simple["tier"] == "SIMPLE"
        assert simple["method"] == "portfolio"
        assert simple["model"]
        assert simple["candidates"][0] == simple["model"]
        assert simple["router_version"] == "v3-portfolio"

        reasoning = route(
            "prove the theorem step by step using mathematical induction", None, 4096, BASE_OPTIONS
        )

        assert reasoning["tier"] == "REASONING"
        assert reasoning["method"] == "portfolio"
        assert simple["tier_configs"] is not None
        assert simple["profile"] is not None
        assert reasoning["tier_configs"] is not None
        assert reasoning["profile"] is not None

    def test_supports_a_config_only_rollback_to_the_v2_rules_strategy(self):
        decision = route(
            "hello",
            None,
            100,
            {**BASE_OPTIONS, "config": {**DEFAULT_ROUTING_CONFIG, "strategy": "rules"}},
        )

        assert decision["method"] == "rules"

    def test_recognizes_multiple_choice_reasoning(self):
        decision = route(
            "Which statement is correct?\n\nA. First\nB. Second\nC. Third\nD. Fourth\n\n"
            "Return the final answer choice.",
            None,
            512,
            BASE_OPTIONS,
        )

        assert decision["task_type"] == "reasoning_mcq"
        assert decision["tier"] == "REASONING"
        assert decision["model"] == "google/gemini-3-flash-preview"
        assert "xai/grok-4.5" in decision["candidates"]
        assert decision["tier_configs"]["REASONING"]["primary"] == decision["model"]

    def test_recognizes_compact_multilingual_arithmetic(self):
        decision = route(
            "Una caja tiene 12 libros. Hay 4 cajas. ¿Cuántos libros hay en total?",
            None,
            512,
            BASE_OPTIONS,
        )

        assert decision["task_type"] == "reasoning_math"
        assert decision["tier"] == "REASONING"
        assert decision["model"] == "google/gemini-3.5-flash"

    def test_recognizes_math_word_problems_without_question_marks(self):
        decision = route(
            "เรือแล่นได้เร็ว 10 ไมล์ต่อชั่วโมง ตั้งแต่ 13.00 น. ถึง 16.00 น. " "และกลับด้วยความเร็ว 6 ไมล์ต่อชั่วโมง",
            None,
            512,
            BASE_OPTIONS,
        )

        assert decision["task_type"] == "reasoning_math"


# ─── tool-intent.test.ts ───


class TestInferToolRequirement:
    def test_does_not_confuse_available_tools_with_a_tool_requirement(self):
        assert not infer_tool_requirement(
            "Which option best explains the observation?\nA. One\nB. Two\nC. Three\nD. Four"
        )
        assert not infer_tool_requirement("What is 17 times 9?")

    def test_recognizes_explicit_tool_repository_web_and_stateful_actions(self):
        assert infer_tool_requirement("Use the lookup_order tool for order B-42.")
        assert infer_tool_requirement("Patch the repository and run the tests.")
        assert infer_tool_requirement(
            "Calculate the average and save it in a file called result.txt."
        )
        assert infer_tool_requirement("Search the web for today's weather in Shanghai.")
        assert infer_tool_requirement("Cancel my flight booking and refund the ticket.")
        assert infer_tool_requirement("修改仓库里的文件，然后运行测试。")

    def test_honors_the_openai_tool_choice_contract(self):
        assert infer_tool_requirement("Retrieve the account details.", None, "required")
        assert infer_tool_requirement(
            "Retrieve the account details.",
            None,
            {"type": "function", "function": {"name": "get_account"}},
        )
        assert not infer_tool_requirement("What is 17 times 9?", None, "auto")
        assert not infer_tool_requirement(
            "Cancel my flight booking and refund the ticket.", None, "none"
        )

    def test_does_not_treat_host_tool_descriptions_as_a_per_turn_requirement(self):
        system_prompt = (
            "You can use web_search to look up documentation, run tests, "
            "and update account records."
        )

        assert not infer_tool_requirement("What is 17 times 9?", system_prompt)


class TestDimensionWeightKeys:
    """Every scored dimension must find its weight.

    The config transpile that produced ``config.py`` snake_cased key names, and
    ``imperativeVerbs`` is both a keyword-list field *and* a dimension name — so
    the weight landed under ``imperative_verbs`` while the classifier emitted
    ``imperativeVerbs``. ``weights.get(name, 0)`` then silently scored that
    dimension at zero, diverging from the TypeScript SDK on any prompt whose
    imperative verbs would have crossed a tier boundary.
    """

    def test_every_emitted_dimension_has_a_weight(self):
        from blockrun_llm.router_core.rules import classify_by_rules

        scoring = DEFAULT_ROUTING_CONFIG["scoring"]
        result = classify_by_rules("Build and deploy the service", None, 10, scoring)
        emitted = {dimension["name"] for dimension in result["dimensions"]}
        weighted = set(scoring["dimension_weights"])

        assert emitted - weighted == set(), "scored dimensions with no weight"
        assert weighted - emitted == set(), "weights that match no scored dimension"

    def test_the_weights_match_the_upstream_values(self):
        # Ported verbatim from router-core config.ts at d7bc10c.
        assert DEFAULT_ROUTING_CONFIG["scoring"]["dimension_weights"] == {
            "tokenCount": 0.08,
            "codePresence": 0.15,
            "reasoningMarkers": 0.18,
            "technicalTerms": 0.1,
            "creativeMarkers": 0.05,
            "simpleIndicators": 0.02,
            "multiStepPatterns": 0.12,
            "questionComplexity": 0.05,
            "imperativeVerbs": 0.03,
            "constraintCount": 0.04,
            "outputFormat": 0.03,
            "referenceComplexity": 0.02,
            "negationComplexity": 0.01,
            "domainSpecificity": 0.02,
            "agenticTask": 0.04,
        }


class TestUnavailableModels:
    """1:1 port of ``unavailable-models.test.ts`` (d7bc10c)."""

    TIERS = {
        "SIMPLE": {"primary": "a/one", "fallback": ["a/two", "a/three"]},
        "MEDIUM": {"primary": "b/one", "fallback": ["b/two"]},
        "COMPLEX": {"primary": "c/one", "fallback": []},
        "REASONING": {"primary": "d/one", "fallback": ["d/two"]},
    }

    @staticmethod
    def _options(**overrides):
        from blockrun_llm.router_core import DEFAULT_MODEL_CAPABILITIES

        pricing = {
            model: {"input_price": 1.0, "output_price": 3.0} for model in DEFAULT_MODEL_CAPABILITIES
        }
        base = {
            "config": DEFAULT_ROUTING_CONFIG,
            "model_pricing": pricing,
            "now": datetime(2026, 8, 20, tzinfo=timezone.utc),
        }
        base.update(overrides)
        return base

    def test_is_the_identity_for_an_absent_or_empty_list(self):
        assert apply_unavailable_models(self.TIERS, None) is self.TIERS
        assert apply_unavailable_models(self.TIERS, []) is self.TIERS

    def test_promotes_the_first_surviving_fallback_when_the_primary_is_dead(self):
        result = apply_unavailable_models(self.TIERS, ["a/one"])
        assert result["SIMPLE"] == {"primary": "a/two", "fallback": ["a/three"]}
        # Untouched tiers keep their original config objects.
        assert result["MEDIUM"] is self.TIERS["MEDIUM"]

    def test_removes_dead_rungs_from_the_middle_of_a_chain(self):
        result = apply_unavailable_models(self.TIERS, ["a/two"])
        assert result["SIMPLE"] == {"primary": "a/one", "fallback": ["a/three"]}

    def test_keeps_the_original_config_when_a_tiers_whole_chain_is_dead(self):
        result = apply_unavailable_models(self.TIERS, ["c/one"])
        assert result["COMPLEX"] is self.TIERS["COMPLEX"]

    def test_does_not_mutate_its_input(self):
        apply_unavailable_models(self.TIERS, ["a/one", "b/one"])
        assert self.TIERS["SIMPLE"]["primary"] == "a/one"
        assert self.TIERS["MEDIUM"]["primary"] == "b/one"

    def test_never_selects_or_lists_a_model_the_host_declared_dead(self):
        baseline = route("What is the capital of France?", None, 256, self._options())
        dead = baseline["model"]
        decision = route(
            "What is the capital of France?",
            None,
            256,
            self._options(unavailable_models=[dead]),
        )
        assert decision["model"] != dead
        assert dead not in (decision.get("candidates") or [])

    def test_keeps_dead_evidence_candidates_out_of_the_portfolio_chain(self):
        math_prompt = "Solve for x: 3x^2 - 12x + 9 = 0. Show your work."
        baseline = route(math_prompt, None, 1024, self._options())
        evidence = baseline.get("candidates") or []
        assert len(evidence) > 1
        dead = evidence[0]
        decision = route(math_prompt, None, 1024, self._options(unavailable_models=[dead]))
        assert decision["model"] != dead
        assert dead not in (decision.get("candidates") or [])

    def test_applies_to_the_rules_strategy_as_well(self):
        config = {**DEFAULT_ROUTING_CONFIG, "strategy": "rules"}
        baseline = route("What is the capital of France?", None, 256, self._options(config=config))
        dead = baseline["model"]
        decision = route(
            "What is the capital of France?",
            None,
            256,
            self._options(config=config, unavailable_models=[dead]),
        )
        assert decision["model"] != dead

    def test_survives_killing_an_entire_tier_chain(self):
        chain = [
            DEFAULT_ROUTING_CONFIG["tiers"]["SIMPLE"]["primary"],
            *DEFAULT_ROUTING_CONFIG["tiers"]["SIMPLE"]["fallback"],
        ]
        decision = route(
            "What is the capital of France?",
            None,
            256,
            self._options(unavailable_models=chain),
        )
        assert len(decision["model"]) > 0
