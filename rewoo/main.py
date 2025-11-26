import dotenv
from typing import List
import re

from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults

dotenv.load_dotenv(override=True)

search = TavilySearchResults()


class ReWOO(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str



from langchain_fireworks import ChatFireworks

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
task = "2024中国首富的家乡是哪里？"
# result = model.invoke(prompt.format(task=task))

# print(result.content)


# Regex to match expressions of the form E#... = ...[...]
regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
planner = prompt_template | model


def get_plan(state: ReWOO):
    task = state["task"]
    result = planner.invoke({"task": task})
    # Find all matches in the sample text
    matches = re.findall(regex_pattern, result.content)
    print("planner.content:")
    print(result.content)
    print("matches:")
    print(matches)
    return {"steps": matches, "plan_string": result.content}


def _get_current_task(state: ReWOO):
    if "results" not in state or state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1


def tool_execution(state: ReWOO):
    """Worker node that executes the tools of a given plan."""
    _step = _get_current_task(state)
    _, step_name, tool, tool_input = state["steps"][_step - 1]
    _results = (state["results"] or {}) if "results" in state else {}
    for k, v in _results.items():
        tool_input = tool_input.replace(k, v)
    if tool == "Google":
        result = search.invoke(tool_input)
    elif tool == "LLM":
        result = model.invoke(tool_input)
    else:
        raise ValueError
    _results[step_name] = str(result)
    return {"results": _results}



solve_prompt = """请解决以下任务或问题。为了解决该问题，我们已经制定了逐步的 Plan，并为每个 Plan 检索了对应的证据。使用这些证据时请谨慎，较长的证据可能包含无关信息。

{plan}

现在请根据上述证据回答问题或完成任务。回答时只输出最终结果，不要添加额外文字。

Task: {task}
Response:"""


def solve(state: ReWOO):
    plan = ""
    for _plan, step_name, tool, tool_input in state["steps"]:
        _results = (state["results"] or {}) if "results" in state else {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
            step_name = step_name.replace(k, v)
        plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
    prompt = solve_prompt.format(plan=plan, task=state["task"])
    result = model.invoke(prompt)
    return {"result": result.content}


def _route(state):
    _step = _get_current_task(state)
    if _step is None:
        # We have executed all tasks
        return "solve"
    else:
        # We are still executing tasks, loop back to the "tool" node
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


for s in app.stream({"task": task}):
    print(s)
    print("==="*10)