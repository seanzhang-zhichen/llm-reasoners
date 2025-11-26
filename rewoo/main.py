"""ReWOO planning graph entrypoint."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

import dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_fireworks import ChatFireworks
from typing_extensions import TypedDict

dotenv.load_dotenv(override=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

PlanStep = Tuple[str, str, str, str]
ResultMap = Dict[str, str]


class ReWOO(TypedDict):
    """State shared across planner, tool executor, and solver."""

    task: str
    plan_string: str
    steps: List[PlanStep]
    results: ResultMap
    result: str


search = TavilySearchResults()
model = ChatFireworks(
    model="accounts/fireworks/models/deepseek-v3p1-terminus",
    max_tokens=25_344,
    streaming=True,
)


prompt = """针对以下任务，请制定可逐步解决问题的计划。每个计划需要指明将使用的外部工具以及对应的输入，以便检索证据。你可以将证据存储在 \
变量 #E 中，供后续工具调用。（Plan, #E1, Plan, #E2, Plan, ...）

可使用的工具如下：
(1) Google[input]：在 Google 上搜索结果的工具。当你需要就特定主题找到简洁答案时非常有用。输入应为搜索查询。
(2) LLM[input]：与你一样的预训练大语言模型。当你可以依靠通用知识与常识解决问题时优先使用。输入可以是任意指令。

示例：
Task: Thomas、Toby 和 Rebecca 在一周内总共工作了 157 小时。Thomas 工作了 x 小时。Toby 的工作时长比 Thomas 的两倍少 10 小时，Rebecca 的工作时长比 Toby 少 8 小时。Rebecca 工作了多少小时？
Plan: 已知 Thomas 工作了 x 小时，将题目转写为代数表达式并使用 Wolfram Alpha 求解。#E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]
Plan: 求出 Thomas 工作的小时数。#E2 = LLM[What is x, given #E1]
Plan: 计算 Rebecca 工作的小时数。#E3 = Calculator[(2 ∗ #E2 − 10) − 8]

开始！
请详细描述每个计划。每个 Plan 后只能跟一个 #E。

Task: {task}"""
regex_pattern = re.compile(r"Plan:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]")
prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
planner = prompt_template | model


def _parse_plan(content: str) -> List[PlanStep]:
    """Parse planner output into structured steps."""
    matches: List[PlanStep] = regex_pattern.findall(content)
    if not matches:
        raise ValueError("Planner response missing structured steps.")
    return matches


def _substitute_results(value: str, results: ResultMap) -> str:
    """Replace step placeholders (#E...) with actual tool outputs."""
    for placeholder, tool_output in results.items():
        value = value.replace(placeholder, tool_output)
    return value


def get_plan(state: ReWOO):
    """Planner node: produce outline and extracted steps."""
    logger.info("Planning for task: %s", state["task"])
    result = planner.invoke({"task": state["task"]})
    steps = _parse_plan(result.content)
    logger.debug("Planner raw response: %s", result.content)
    logger.info("Planner extracted %d steps.", len(steps))
    return {"steps": steps, "plan_string": result.content}


def _get_current_task(state: ReWOO):
    if "results" not in state or state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1


def _execute_tool(tool: str, tool_input: str):
    if tool == "Google":
        return search.invoke(tool_input)
    if tool == "LLM":
        return model.invoke(tool_input)
    raise ValueError(f"Unsupported tool: {tool}")


def tool_execution(state: ReWOO):
    """Worker node that executes the tools of a given plan."""
    _step = _get_current_task(state)
    logger.info("Executing step %s/%s", _step, len(state["steps"]))
    _, step_name, tool, tool_input = state["steps"][_step - 1]
    _results = (state.get("results") or {}).copy()
    resolved_input = _substitute_results(tool_input, _results)
    logger.info("Tool %s called with input: %s", tool, resolved_input)
    result = _execute_tool(tool, resolved_input)
    logger.info("Tool %s finished. Result type: %s", tool, type(result))
    _results[step_name] = str(result)
    return {"results": _results}



solve_prompt = """请解决以下任务或问题。为了解决该问题，我们已经制定了逐步的 Plan，并为每个 Plan 检索了对应的证据。使用这些证据时请谨慎，较长的证据可能包含无关信息。

{plan}

现在请根据上述证据回答问题或完成任务。回答时只输出最终结果，不要添加额外文字。

Task: {task}
Response:"""


def _format_plan_for_solver(state: ReWOO) -> str:
    """Builds the evidence string expected by the solver prompt."""
    formatted_parts: List[str] = []
    results = state.get("results") or {}
    for plan_text, step_name, tool, tool_input in state["steps"]:
        formatted_step = (
            f"Plan: {plan_text}\n"
            f"{_substitute_results(step_name, results)} = "
            f"{tool}[{_substitute_results(tool_input, results)}]"
        )
        formatted_parts.append(formatted_step)
    return "\n".join(formatted_parts)


def solve(state: ReWOO):
    logger.info("Solving final response for task: %s", state["task"])
    plan = _format_plan_for_solver(state)
    prompt = solve_prompt.format(plan=plan, task=state["task"])
    result = model.invoke(prompt)
    logger.info("Solver produced response.")
    return {"result": result.content}


def _route(state):
    _step = _get_current_task(state)
    if _step is None:
        # We have executed all tasks
        logger.info("All steps complete. Routing to solve node.")
        return "solve"
    else:
        # We are still executing tasks, loop back to the "tool" node
        logger.info("Routing back to tool node for step %s.", _step)
        return "tool"

from langgraph.graph import END, StateGraph, START

graph = StateGraph(ReWOO)
graph.add_node("plan", get_plan)
graph.add_node("tool", tool_execution)
graph.add_node("solve", solve)
graph.add_edge("plan", "tool")
graph.add_edge("solve", END)
graph.add_conditional_edges("tool", _route)
graph.add_edge(START, "plan")

app = graph.compile()


def run(task: str):
    """Utility runner for manual debugging."""
    logger.info("Starting run for task: %s", task)
    for chunk in app.stream({"task": task}):
        print(chunk)
        print("===" * 10)
    logger.info("Run complete for task: %s", task)


if __name__ == "__main__":
    run("2024中国首富的家乡是哪里？")