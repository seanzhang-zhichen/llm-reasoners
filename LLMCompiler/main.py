"""
LLM Compiler - A LangGraph-based agent that plans, schedules, and executes tasks.
"""

# ============================================================================
# Imports
# ============================================================================
import itertools
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import (
    Annotated,
    Any,
    Dict,
    Iterable,
    List,
    Sequence,
    Union,
)

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    chain as as_runnable,
)
from langchain_core.tools import BaseTool
from langchain_fireworks import ChatFireworks
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from math_tools import get_math_tool
from output_parser import LLMCompilerPlanParser, Task

load_dotenv()


# ============================================================================
# Configuration and Initialization
# ============================================================================

# Initialize LLM
llm = ChatFireworks(
    model="accounts/fireworks/models/deepseek-v3p1-terminus",
    max_tokens=25_344,
    streaming=True,
)

# Initialize tools
calculate = get_math_tool(llm)
search = TavilySearchResults(
    max_results=1,
    description='tavily_search_results_json(query="the search query") - a search engine.',
)
tools = [search, calculate]

# Load prompts
prompt = hub.pull("wfh/llm-compiler")


# ============================================================================
# Pydantic Models
# ============================================================================

class FinalResponse(BaseModel):
    """The final response/answer."""

    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""

    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]


class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterable[Task]


class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================================
# Helper Functions
# ============================================================================

def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    """Get all previous tool responses."""
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    if results:
        print(f"\n[获取历史观察结果] 找到 {len(results)} 个之前的任务结果: {list(results.keys())}")
    return results


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
    """Resolve task arguments that reference other task outputs."""
    ID_PATTERN = r"\$\{?(\d+)\}?"

    def replace_match(match):
        idx = int(match.group(1))
        resolved_value = observations.get(idx, match.group(0))
        if idx in observations:
            print(f"  [参数解析] ${idx} -> {resolved_value}")
        return str(resolved_value)

    if isinstance(arg, str):
        return re.sub(ID_PATTERN, replace_match, arg)
    elif isinstance(arg, list):
        return [_resolve_arg(a, observations) for a in arg]
    else:
        return str(arg)


def _execute_task(task, observations, config):
    """Execute a single task with resolved arguments."""
    tool_to_use = task["tool"]
    if isinstance(tool_to_use, str):
        return tool_to_use
    args = task["args"]
    try:
        if isinstance(args, str):
            resolved_args = _resolve_arg(args, observations)
        elif isinstance(args, dict):
            resolved_args = {
                key: _resolve_arg(val, observations) for key, val in args.items()
            }
        else:
            resolved_args = args
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}.)"
            f" Args could not be resolved. Error: {repr(e)}"
        )
    try:
        return tool_to_use.invoke(resolved_args, config)
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}."
            + f" Args resolved to {resolved_args}. Error: {repr(e)})"
        )


# ============================================================================
# Planner
# ============================================================================

def create_planner(
    llm: BaseChatModel, tools: Sequence[BaseTool], base_prompt: ChatPromptTemplate
):
    """Create a planner that generates task plans."""
    tool_descriptions = "\n".join(
        f"{i + 1}. {tool.description}\n"
        for i, tool in enumerate(tools)
    )
    planner_prompt = base_prompt.partial(
        replan="",
        num_tools=len(tools) + 1,  # Add one because we're adding the join() tool at the end.
        tool_descriptions=tool_descriptions,
    )
    replanner_prompt = base_prompt.partial(
        replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
        "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
        'You MUST use these information to create the next plan under "Current Plan".\n'
        ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
        " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
        " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )

    def should_replan(state: list):
        """Check if we need to replan based on system message."""
        return isinstance(state[-1], SystemMessage)

    def wrap_messages(state: list):
        return {"messages": state}

    def wrap_and_get_last_index(state: list):
        next_task = 0
        for message in state[::-1]:
            if isinstance(message, FunctionMessage):
                next_task = message.additional_kwargs["idx"] + 1
                break
        state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
        return {"messages": state}

    return (
        RunnableBranch(
            (should_replan, wrap_and_get_last_index | replanner_prompt),
            wrap_messages | planner_prompt,
        )
        | llm
        | LLMCompilerPlanParser(tools=tools)
    )


planner = create_planner(llm, tools, prompt)


# ============================================================================
# Task Scheduling
# ============================================================================

@as_runnable
def schedule_task(task_inputs, config):
    """Execute a single task."""
    task: Task = task_inputs["task"]
    observations: Dict[int, Any] = task_inputs["observations"]
    task_idx = task["idx"]
    tool_name = task["tool"] if isinstance(task["tool"], str) else task["tool"].name
    print(f"\n[执行任务 {task_idx}] 工具: {tool_name}")
    print(f"  原始参数: {task['args']}")
    print(f"  依赖任务: {task.get('dependencies', [])}")
    try:
        observation = _execute_task(task, observations, config)
        print(f"  ✓ 任务 {task_idx} 执行成功")
        print(
            f"  结果: {str(observation)[:200]}..."
            if len(str(observation)) > 200
            else f"  结果: {observation}"
        )
    except Exception:
        import traceback

        observation = traceback.format_exception()
        print(f"  ✗ 任务 {task_idx} 执行失败: {observation}")
    observations[task["idx"]] = observation


def schedule_pending_task(
    task: Task, observations: Dict[int, Any], retry_after: float = 0.2
):
    """Schedule a task that has dependencies, waiting for them to complete."""
    task_idx = task["idx"]
    tool_name = task["tool"] if isinstance(task["tool"], str) else task["tool"].name
    wait_count = 0
    while True:
        deps = task["dependencies"]
        if deps and (any([dep not in observations for dep in deps])):
            missing_deps = [dep for dep in deps if dep not in observations]
            if wait_count == 0:
                print(
                    f"\n[等待任务 {task_idx}] 工具: {tool_name} 等待依赖任务完成: {missing_deps}"
                )
            wait_count += 1
            time.sleep(retry_after)
            continue
        if wait_count > 0:
            print(f"  ✓ 依赖任务已完成，开始执行任务 {task_idx}")
        schedule_task.invoke({"task": task, "observations": observations})
        break


@as_runnable
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    """Group the tasks into a DAG schedule and execute them."""
    print("\n" + "=" * 80)
    print("步骤2: 任务调度和执行")
    print("=" * 80)
    tasks = scheduler_input["tasks"]
    args_for_tasks = {}
    messages = scheduler_input["messages"]
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)
    futures = []
    retry_after = 0.25
    task_count = 0
    print(f"\n[任务调度] 开始处理任务流...")
    with ThreadPoolExecutor() as executor:
        for task in tasks:
            task_count += 1
            deps = task["dependencies"]
            task_names[task["idx"]] = (
                task["tool"] if isinstance(task["tool"], str) else task["tool"].name
            )
            args_for_tasks[task["idx"]] = task["args"]
            if deps and (any([dep not in observations for dep in deps])):
                print(f"[调度任务 {task['idx']}] 有依赖，加入等待队列")
                futures.append(
                    executor.submit(
                        schedule_pending_task, task, observations, retry_after
                    )
                )
            else:
                print(f"[调度任务 {task['idx']}] 无依赖或依赖已满足，立即执行")
                schedule_task.invoke(dict(task=task, observations=observations))

        print(f"\n[任务调度] 共处理了 {task_count} 个任务")
        if futures:
            print(f"[等待完成] 等待 {len(futures)} 个依赖任务完成...")
        wait(futures)
        print("[任务调度] 所有任务执行完成")
    
    new_observations = {
        k: (task_names[k], args_for_tasks[k], observations[k])
        for k in sorted(observations.keys() - originals)
    }
    print(f"\n[生成消息] 将 {len(new_observations)} 个任务结果转换为FunctionMessage")
    tool_messages = [
        FunctionMessage(
            name=name,
            content=str(obs),
            additional_kwargs={"idx": k, "args": task_args},
            tool_call_id=k,
        )
        for k, (name, task_args, obs) in new_observations.items()
    ]
    return tool_messages


@as_runnable
def plan_and_schedule(state):
    """Plan tasks and schedule their execution."""
    messages = state["messages"]
    print("\n" + "=" * 80)
    print("开始执行: plan_and_schedule")
    print("=" * 80)
    user_message = next(
        (msg.content for msg in messages if isinstance(msg, HumanMessage)), "N/A"
    )
    print(f"用户问题: {user_message}\n")

    print("[生成计划] 调用planner生成任务计划...")
    tasks = planner.stream(messages)
    try:
        first_task = next(tasks)
        print(f"[计划生成] 第一个任务: {first_task['idx']} - {first_task['tool']}")
        tasks = itertools.chain([first_task], tasks)
    except StopIteration:
        print("[计划生成] 警告: 没有生成任何任务")
        tasks = iter([])
    scheduled_tasks = schedule_tasks.invoke(
        {
            "messages": messages,
            "tasks": tasks,
        }
    )
    print("\n" + "=" * 80)
    print("plan_and_schedule 执行完成")
    print("=" * 80)
    return {"messages": scheduled_tasks}


# ============================================================================
# Joiner
# ============================================================================

def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    """Parse the joiner output and return appropriate messages."""
    print("\n[解析Joiner输出] 分析决策结果...")
    print(f"  思考过程: {decision.thought}")

    response = [AIMessage(content=f"Thought: {decision.thought}")]
    if isinstance(decision.action, Replan):
        print(f"  → 决策: 需要重新规划 (Replan)")
        print(f"  反馈信息: {decision.action.feedback}")
        return {
            "messages": response
            + [
                SystemMessage(
                    content=f"Context from last attempt: {decision.action.feedback}"
                )
            ]
        }
    else:
        print(f"  → 决策: 返回最终答案 (FinalResponse)")
        print(f"  最终答案: {decision.action.response}")
        return {"messages": response + [AIMessage(content=decision.action.response)]}


def select_recent_messages(state) -> dict:
    """Select recent messages from the last HumanMessage."""
    print("\n[选择消息] 从状态中选择最近的消息（从最后一个HumanMessage开始）...")
    messages = state["messages"]
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage):
            content_preview = (
                msg.content[:100] + "..."
                if len(msg.content) > 100
                else msg.content
            )
            print(f"  找到用户消息: {content_preview}")
            break
    selected_messages = selected[::-1]
    print(f"  选择了 {len(selected_messages)} 条消息用于joiner决策")
    for i, msg in enumerate(selected_messages, 1):
        msg_type = type(msg).__name__
        content_preview = (
            str(msg.content)[:80] + "..."
            if len(str(msg.content)) > 80
            else str(msg.content)
        )
        print(f"    {i}. {msg_type}: {content_preview}")
    return {"messages": selected_messages}


# Initialize joiner
joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(examples="")
joiner_runnable = joiner_prompt | llm.with_structured_output(
    JoinOutputs, method="function_calling"
)
joiner = select_recent_messages | joiner_runnable | _parse_joiner_output


# ============================================================================
# Graph Construction
# ============================================================================

def should_continue(state):
    """Determine whether to continue execution or end."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    last_msg_type = type(last_message).__name__ if last_message else "None"
    last_msg_content = (
        str(last_message.content)[:100] + "..."
        if last_message and len(str(last_message.content)) > 100
        else (str(last_message.content) if last_message else "None")
    )

    print(f"\n[条件判断] 检查是否需要继续执行...")
    print(f"  当前消息总数: {len(messages)}")
    print(f"  最后一条消息类型: {last_msg_type}")
    if last_message:
        print(f"  最后一条消息内容预览: {last_msg_content}")

    if isinstance(last_message, AIMessage):
        print(f"  → 决策: 结束执行 (END)")
        print(f"    原因: 最后一条消息是AIMessage，表示已返回最终答案")
        return END
    else:
        print(f"  → 决策: 继续执行 (plan_and_schedule)")
        print(f"    原因: 需要重新规划，返回 'plan_and_schedule' 节点")
        return "plan_and_schedule"


# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("plan_and_schedule", plan_and_schedule)
graph_builder.add_node("join", joiner)
graph_builder.add_edge("plan_and_schedule", "join")
graph_builder.add_conditional_edges("join", should_continue)
graph_builder.add_edge(START, "plan_and_schedule")
chain = graph_builder.compile()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("执行示例1: 查询纽约的GDP")
    print("=" * 80)
    # example1_question = "What's the GDP of New York?"
    # print(f"问题: {example1_question}\n")

    # step_count = 0
    # for step in chain.stream(
    #     {"messages": [HumanMessage(content=example1_question)]}
    # ):
    #     step_count += 1
    #     print(f"\n{'='*80}")
    #     print(f"步骤 {step_count}")
    #     print(f"{'='*80}")

    #     for node_name, node_output in step.items():
    #         print(f"\n节点: {node_name}")
    #         if isinstance(node_output, dict) and "messages" in node_output:
    #             messages = node_output["messages"]
    #             print(f"  消息数量: {len(messages)}")
    #             for i, msg in enumerate(messages[-3:], 1):
    #                 msg_type = type(msg).__name__
    #                 content_preview = (
    #                     str(msg.content)[:150] + "..."
    #                     if len(str(msg.content)) > 150
    #                     else str(msg.content)
    #                 )
    #                 print(f"    {i}. [{msg_type}] {content_preview}")
    #         else:
    #             print(f"  输出: {node_output}")
    #     print("\n" + "-" * 80)

    # print(f"\n{'='*80}")
    # print(f"示例1执行完成，共执行了 {step_count} 个步骤")
    # print(f"{'='*80}")

    print("\n\n" + "=" * 80)
    print("执行示例2: 复杂数学计算")
    print("=" * 80)
    example2_question = (
        "What's ((3*(4+5)/0.5)+3245) + 8? "
        "What's 32/4.23? "
        "What's the sum of those two values?"
    )
    print(f"问题: {example2_question}\n")

    step_count = 0
    for step in chain.stream(
        {"messages": [HumanMessage(content=example2_question)]}
    ):
        step_count += 1
        print(f"\n{'='*80}")
        print(f"步骤 {step_count}")
        print(f"{'='*80}")

        for node_name, node_output in step.items():
            print(f"\n节点: {node_name}")
            if isinstance(node_output, dict) and "messages" in node_output:
                messages = node_output["messages"]
                print(f"  消息数量: {len(messages)}")
                for i, msg in enumerate(messages[-3:], 1):
                    msg_type = type(msg).__name__
                    content_preview = (
                        str(msg.content)[:150] + "..."
                        if len(str(msg.content)) > 150
                        else str(msg.content)
                    )
                    print(f"    {i}. [{msg_type}] {content_preview}")
            else:
                print(f"  输出: {node_output}")
        print("\n" + "-" * 80)

    print(f"\n{'='*80}")
    print(f"示例2执行完成，共执行了 {step_count} 个步骤")
    print(f"{'='*80}")
