from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_fireworks import ChatFireworks
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

import dotenv
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dotenv.load_dotenv(override=True)
logger.info("环境变量加载完成")

llm = ChatFireworks(
    model="accounts/fireworks/models/deepseek-v3p1-terminus", max_tokens=25344, streaming=True
)
logger.info("LLM 客户端初始化完成")

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
logger.info("Tavily 搜索工具初始化完成")

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import ValidationError

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="对答案中“缺失内容”的批评与指出")
    superfluous: str = Field(description="对答案中“多余或冗余内容”的批评与指出。")


class AnswerQuestion(BaseModel):
    """回答问题。提供答案、反思，并随后提出搜索查询以改进答案。"""

    answer: str = Field(description="约250字的详细答案。")
    reflection: Reflection = Field(description="对初始答案的反思。")
    search_queries: list[str] = Field(
        description="1-3个搜索查询，用于研究如何改进当前答案以回应批评。"
    )


class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: dict):
        response = []
        for attempt in range(3):
            logger.info(f"尝试第 {attempt + 1} 次调用 runnable")
            response = self.runnable.invoke(
                {"messages": state["messages"]}, {"tags": [f"attempt:{attempt}"]}
            )
            logger.info(f"runnable 响应: {response}")
            try:
                self.validator.invoke(response)
                logger.info("验证通过，返回响应")
                return {"messages": response}
            except ValidationError as e:
                logger.warning(f"验证失败: {e}")
                state = state + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\n"
                        + "请仔细关注函数模式。\n\n"
                        + self.validator.schema_json()
                        + " 请修复所有验证错误。",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
        logger.error("达到最大重试次数，返回最后一次响应")
        return {"messages": response}


import datetime

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一名专业的研究人员。
当前时间：{time}

1. {first_instruction}
2. 反思并批评你的答案。要严格，以最大化改进。
3. 推荐搜索查询以研究信息并改进你的答案。""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "\n\n<system>请反思用户的原始问题和至今已采取的操作。"
            " 使用{function_name}函数回复。</reminder>",
        ),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)
initial_answer_chain = actor_prompt_template.partial(
    first_instruction="请提供一个约250字的详细答案。",
    function_name=AnswerQuestion.__name__,
) | llm.bind_tools(tools=[AnswerQuestion])
validator = PydanticToolsParser(tools=[AnswerQuestion])

first_responder = ResponderWithRetries(
    runnable=initial_answer_chain, validator=validator
)


example_question = "为什么反思在AI中有用？"
logger.info(f"开始处理示例问题: {example_question}")
initial = first_responder.respond(
    {"messages": [HumanMessage(content=example_question)]}
)
logger.info("初始答案生成完成")
logger.info(f"初始答案: {initial['messages']}")


revise_instructions = """请使用新信息完善之前的答案。
    - 你应该使用之前的反思来添加重要信息到你的答案中。
        - 你必须在你的答案中包含数值引用，以确保它可以被验证。
        - 在答案的底部添加一个“References”部分（这不会计入单词限制）。格式为：
            - [1] https://example.com
            - [2] https://example.com
    - 你应该使用之前的反思来移除多余或冗余的信息，确保答案不超过250字。
"""


# Extend the initial answer schema to include references.
# Forcing citation in the model encourages grounded responses
class ReviseAnswer(AnswerQuestion):
    """根据新信息修订原始答案。提供答案、反思，

    用参考文献支持你的反思，并最终
    添加搜索查询以改进答案。"""

    references: list[str] = Field(
        description="支持你更新答案的引用。"
    )


revision_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions,
    function_name=ReviseAnswer.__name__,
) | llm.bind_tools(tools=[ReviseAnswer])
revision_validator = PydanticToolsParser(tools=[ReviseAnswer])

revisor = ResponderWithRetries(runnable=revision_chain, validator=revision_validator)


import json

logger.info("开始修订答案")
revised = revisor.respond(
    {
        "messages": [
            HumanMessage(content=example_question),
            initial["messages"],
            ToolMessage(
                tool_call_id=initial["messages"].tool_calls[0]["id"],
                content=json.dumps(
                    tavily_tool.invoke(
                        {
                            "query": initial["messages"].tool_calls[0]["args"][
                                "search_queries"
                            ][0]
                        }
                    )
                ),
            ),
        ]
    }
)
logger.info("修订答案生成完成")
logger.info(f"修订答案: {revised['messages']}")


from langchain_core.tools import StructuredTool

from langgraph.prebuilt import ToolNode


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


tool_node = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
logger.info("工具节点初始化完成")


from typing import Literal

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]


MAX_ITERATIONS = 5
builder = StateGraph(State)
builder.add_node("draft", first_responder.respond)


builder.add_node("execute_tools", tool_node)
builder.add_node("revise", revisor.respond)
# draft -> execute_tools
builder.add_edge("draft", "execute_tools")
# execute_tools -> revise
builder.add_edge("execute_tools", "revise")

# Define looping logic:


def _get_num_iterations(state: list):
    i = 0
    for m in state[::-1]:
        if m.type not in {"tool", "ai"}:
            break
        i += 1
    return i


def event_loop(state: list):
    # in our case, we'll just stop after N plans
    num_iterations = _get_num_iterations(state["messages"])
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


# revise -> execute_tools OR end
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
builder.add_edge(START, "draft")
graph = builder.compile()


from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
logger.info("图编译完成")


events = graph.stream(
    {"messages": [("user", "我们该如何应对气候危机？")]},
    stream_mode="values",
)
for i, step in enumerate(events):
    logger.info(f"Step {i}")
    step["messages"][-1].pretty_print()