import datetime
import json
import logging
from typing import Annotated

import dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_fireworks import ChatFireworks
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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


class ReviseAnswer(AnswerQuestion):
    """根据新信息修订原始答案，并添加参考资料。"""

    references: list[str] = Field(description="支持更新答案的引用。")


class ResponderWithRetries:
    """对 LangChain runnable 进行封装，失败时自动重试。"""

    def __init__(self, runnable, validator, max_attempts: int = 3):
        self.runnable = runnable
        self.validator = validator
        self.max_attempts = max_attempts

    def respond(self, state: dict):
        messages: list[BaseMessage] = state["messages"]
        response: AIMessage | None = None

        for attempt in range(self.max_attempts):
            logger.info("尝试第 %s 次调用 runnable", attempt + 1)
            response = self.runnable.invoke(
                {"messages": messages},
                {"tags": [f"attempt:{attempt}"]},
            )

            try:
                self.validator.invoke(response)
                logger.info("验证通过，返回响应")
                return {"messages": response}
            except ValidationError as err:
                logger.warning("验证失败: %s", err)
                messages = messages + [
                    response,
                    ToolMessage(
                        content=_format_validation_error(err, self.validator.schema_json()),
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]

        logger.error("达到最大重试次数，返回最后一次响应")
        return {"messages": response}


def _format_validation_error(err: ValidationError, schema: str) -> str:
    return (
        f"{repr(err)}\n\n请仔细关注函数模式。\n\n{schema} 请修复所有验证错误。"
    )


def configure_environment() -> None:
    dotenv.load_dotenv(override=True)
    logger.info("环境变量加载完成")


def build_llm() -> ChatFireworks:
    llm = ChatFireworks(
        model="accounts/fireworks/models/deepseek-v3p1-terminus",
        max_tokens=25_344,
        streaming=True,
    )
    logger.info("LLM 客户端初始化完成")
    return llm


def build_tavily_tool() -> TavilySearchResults:
    search = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
    logger.info("Tavily 搜索工具初始化完成")
    return tavily_tool


def build_actor_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
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
    ).partial(time=lambda: datetime.datetime.now().isoformat())


def build_initial_responder(llm: ChatFireworks, actor_prompt: ChatPromptTemplate) -> ResponderWithRetries:
    chain = actor_prompt.partial(
        first_instruction="请提供一个约250字的详细答案。",
        function_name=AnswerQuestion.__name__,
    ) | llm.bind_tools(tools=[AnswerQuestion])

    validator = PydanticToolsParser(tools=[AnswerQuestion])
    return ResponderWithRetries(chain, validator)


REVISION_INSTRUCTIONS = """请使用新信息完善之前的答案。
- 使用之前的反思来添加重要信息到你的答案中。
    - 你必须在你的答案中包含数值引用，以确保它可以被验证。
    - 在答案底部添加“References”部分（不计入字数），格式：
        - [1] https://example.com
        - [2] https://example.com
- 使用反思移除多余或冗余的信息，确保答案不超过250字。
"""


def build_revision_responder(llm: ChatFireworks, actor_prompt: ChatPromptTemplate) -> ResponderWithRetries:
    chain = actor_prompt.partial(
        first_instruction=REVISION_INSTRUCTIONS,
        function_name=ReviseAnswer.__name__,
    ) | llm.bind_tools(tools=[ReviseAnswer])

    validator = PydanticToolsParser(tools=[ReviseAnswer])
    return ResponderWithRetries(chain, validator)


def build_tool_node(tavily_tool: TavilySearchResults) -> ToolNode:
    def run_queries(search_queries: list[str], **_) -> list[dict]:
        """执行模型生成的查询。"""
        return tavily_tool.batch([{"query": query} for query in search_queries])

    return ToolNode(
        [
            StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
            StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
        ]
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]


MAX_ITERATIONS = 5


def build_graph(
    first_responder: ResponderWithRetries,
    revisor: ResponderWithRetries,
    tool_node: ToolNode,
):
    builder = StateGraph(State)
    builder.add_node("draft", first_responder.respond)
    builder.add_node("execute_tools", tool_node)
    builder.add_node("revise", revisor.respond)
    builder.add_edge("draft", "execute_tools")
    builder.add_edge("execute_tools", "revise")
    builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
    builder.add_edge(START, "draft")
    graph = builder.compile()
    logger.info("图编译完成")
    return graph


def event_loop(state: dict):
    num_iterations = _get_num_iterations(state["messages"])
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


def _get_num_iterations(messages: list[BaseMessage]) -> int:
    count = 0
    for message in reversed(messages):
        if getattr(message, "type", None) not in {"tool", "ai"}:
            break
        count += 1
    return count


def run_initial_pass(responder: ResponderWithRetries, question: str):
    logger.info("开始处理示例问题: %s", question)
    initial = responder.respond({"messages": [HumanMessage(content=question)]})
    logger.info("初始答案生成完成")
    return initial


def run_revision_pass(
    responder: ResponderWithRetries,
    tavily_tool: TavilySearchResults,
    question: str,
    initial_message: AIMessage,
):
    search_query = initial_message.tool_calls[0]["args"]["search_queries"][0]
    logger.info("开始修订答案，搜索查询: %s", search_query)

    tool_message = ToolMessage(
        tool_call_id=initial_message.tool_calls[0]["id"],
        content=json.dumps(tavily_tool.invoke({"query": search_query})),
    )

    revised = responder.respond(
        {
            "messages": [
                HumanMessage(content=question),
                initial_message,
                tool_message,
            ]
        }
    )
    logger.info("修订答案生成完成")
    return revised


def stream_graph(graph, question: str):
    events = graph.stream(
        {"messages": [("user", question)]},
        stream_mode="values",
    )
    for idx, step in enumerate(events):
        logger.info("Graph Step %s", idx)
        step["messages"][-1].pretty_print()


def display_graph_image(graph) -> None:
    try:
        from IPython.display import Image, display

        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        # 可选依赖，不可用时跳过
        pass


def main():
    configure_environment()

    llm = build_llm()
    tavily_tool = build_tavily_tool()
    actor_prompt = build_actor_prompt()

    first_responder = build_initial_responder(llm, actor_prompt)
    revisor = build_revision_responder(llm, actor_prompt)
    tool_node = build_tool_node(tavily_tool)

    initial_question = "为什么反思在AI中有用？"
    initial_response = run_initial_pass(first_responder, initial_question)

    revised_response = run_revision_pass(
        revisor,
        tavily_tool,
        initial_question,
        initial_response["messages"],
    )

    logger.info("初始答案: %s", initial_response["messages"])
    logger.info("修订答案: %s", revised_response["messages"])

    graph = build_graph(first_responder, revisor, tool_node)
    display_graph_image(graph)

    graph_question = "我们该如何应对气候危机？"
    stream_graph(graph, graph_question)


if __name__ == "__main__":
    main()

