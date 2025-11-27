"""ReWOO 规划图执行入口。"""

from __future__ import annotations

import asyncio
import logging
import operator
from pathlib import Path
from typing import Annotated, List, Tuple, Union

import dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_fireworks import ChatFireworks
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

dotenv.load_dotenv(override=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_PROMPT = "你是一名乐于助人的助手。"
DEFAULT_CONFIG = {"recursion_limit": 50}
LLM_MODEL = "accounts/fireworks/models/deepseek-v3p1-terminus"
LLM_MAX_TOKENS = 25_344


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """后续要执行的计划。"""

    steps: List[str] = Field(
        description="需要依次执行的步骤，保持按顺序排列。"
    )


class Response(BaseModel):
    """对用户的回复。"""

    response: str


class Act(BaseModel):
    """需要执行的动作。"""

    action: Union[Response, Plan] = Field(
        description="可执行的动作。如需直接回复用户，使用 Response；若需继续借助工具求解，使用 Plan。"
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_llm() -> ChatFireworks:
    return ChatFireworks(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        streaming=True,
    )


def build_tools():
    return [TavilySearchResults(max_results=3)]


def build_agent(llm: ChatFireworks):
    return create_react_agent(llm, build_tools(), prompt=DEFAULT_PROMPT)


def build_planner(llm: ChatFireworks):
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """针对给定目标，制定一个简单的分步计划。\
该计划应包含独立任务，只要正确执行就能得到正确答案。不要添加任何多余步骤。\
最后一步的结果应是最终答案。确保每一步都包含所需信息——不要跳过步骤。""",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    return planner_prompt | llm.with_structured_output(Plan)


def build_replanner(llm: ChatFireworks):
    replanner_prompt = ChatPromptTemplate.from_template(
        """针对给定目标，制定一个简单的分步计划。\
该计划应包含独立任务，只要正确执行就能得到正确答案。不要添加任何多余步骤。\
最后一步的结果应是最终答案。确保每一步都包含所需信息——不要跳过步骤。

你的目标是：
{input}

你最初的计划是：
{plan}

你目前已完成以下步骤：
{past_steps}

请据此更新计划。如果不再需要更多步骤即可向用户回复，就直接给出答复。否则，仅填写仍需执行的步骤。不要把已经完成的步骤重新写入计划。"""
    )
    return replanner_prompt | llm.with_structured_output(Act)


llm = build_llm()
agent_executor = build_agent(llm)
planner = build_planner(llm)
replanner = build_replanner(llm)


async def execute_step(state: PlanExecute) -> dict:
    """执行当前计划的首个步骤。"""
    plan = state.get("plan", [])
    if not plan:
        logger.warning("计划为空，无法执行。")
        return {"response": "目前没有可执行的计划。"}

    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    step_number = len(state.get("past_steps", [])) + 1
    task_formatted = (
        f"针对以下计划：\n{plan_str}\n\n你的任务是执行第{step_number}步，即：{task}。"
    )
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


async def plan_step(state: PlanExecute) -> dict:
    """根据用户输入生成初始计划。"""
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute) -> dict:
    """根据当前进度决定后续行动。"""
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    """若已有答案则结束，否则继续执行。"""
    return END if state.get("response") else "agent"


def build_workflow() -> StateGraph:
    workflow = StateGraph(PlanExecute)
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")
    workflow.add_conditional_edges(
        "replan",
        should_end,
        ["agent", END],
    )
    return workflow


# 最后编译为 LangChain Runnable，以常规 Runnable 方式调用。
app = build_workflow().compile()


def display_workflow_graph():
    """尝试生成并展示工作流图。"""
    ipy_display = None
    try:
        from IPython.display import Image as IPyImage, display as ipy_display_func

        ipy_display = (IPyImage, ipy_display_func)
    except ImportError:
        logger.debug("未检测到 IPython，跳过内联图展示。")

    try:
        png_bytes = app.get_graph(xray=True).draw_mermaid_png()
        output_path = Path(__file__).with_name("workflow_graph.png")
        output_path.write_bytes(png_bytes)
        logger.info("已将图像保存到 %s", output_path)
        if ipy_display:
            image_cls, display_func = ipy_display
            display_func(image_cls(png_bytes))
    except Exception as exc:  # pragma: no cover - 仅用于可视化
        logger.warning("无法生成图可视化：%s", exc)


async def stream_app(
    question: str,
    recursion_limit: int = DEFAULT_CONFIG["recursion_limit"],
):
    """以流式方式运行编排应用。"""
    config = {"recursion_limit": recursion_limit}
    inputs = {"input": question}
    async for event in app.astream(inputs, config=config):
        for key, value in event.items():
            if key != "__end__":
                print(value)


def main(question: str | None = None, recursion_limit: int | None = None):
    """入口函数，可指定问题与递归深度。"""
    display_workflow_graph()
    asyncio.run(
        stream_app(
            question=question,
            recursion_limit=recursion_limit or DEFAULT_CONFIG["recursion_limit"],
        )
    )


if __name__ == "__main__":
    question = "2024年的中国首富的家乡是哪里？"
    print("开始执行")
    print("问题：", question)
    main(question)