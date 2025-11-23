from typing import Annotated, List, Sequence
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_fireworks import ChatFireworks
import dotenv


dotenv.load_dotenv(override=True)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名作文助手，负责撰写优质的五段式作文。"
            " 请根据用户的请求生成尽可能优秀的文章。"
            " 如果用户提供批评意见，请基于你之前的尝试给出修订版本。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
llm = ChatFireworks(
    model="accounts/fireworks/models/glm-4p6", max_tokens=25344, streaming=True
)
generate = prompt | llm


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名教师，负责对学生的作文进行评估。"
            " 请根据学生的作文内容，生成对其的批评和建议。"
            " 建议包括长度、深度、风格等方面的改进。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
reflect = reflection_prompt | llm



class State(TypedDict):
    messages: Annotated[list, add_messages]


async def generation_node(state: State) -> State:
    return {"messages": [await generate.ainvoke(state["messages"])]}


async def reflection_node(state: State) -> State:
    # Other messages we need to adjust
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    res = await reflect.ainvoke(translated)
    # We treat the output of this as human feedback for the generator
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(State)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_edge(START, "generate")


def should_continue(state: State):
    if len(state["messages"]) > 6:
        # End after 3 iterations
        return END
    return "reflect"


builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)


async def run_graph(prompt: str, thread_id: str = "1"):
    config = {"configurable": {"thread_id": thread_id}}
    async for event in graph.astream({"messages": [HumanMessage(content=prompt)]}, config):
        print(event)
        print("---")
        
    state = graph.get_state(config)
    ChatPromptTemplate.from_messages(state.values["messages"]).pretty_print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_graph("写一篇关于《小王子》为何与现代童年相关的文章。"))
