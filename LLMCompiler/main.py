from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_fireworks import ChatFireworks
from math_tools import get_math_tool
from dotenv import load_dotenv
load_dotenv()


llm = ChatFireworks(model="accounts/fireworks/models/deepseek-v3p1-terminus", max_tokens=25_344, streaming=True)

calculate = get_math_tool(llm)
search = TavilySearchResults(
    max_results=1,
    description='tavily_search_results_json(query="搜索查询") - 搜索引擎。',
)

tools = [search, calculate]

print("tools: ", tools)

print("====="*10)
response = calculate.invoke(
    {
        "problem": "旧金山的温度加5是多少？",
        "context": ["旧金山的温度是32度"],
    }
)

print("response: ", response)



from typing import Any, Sequence

from langchain_classic import hub
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from output_parser import LLMCompilerPlanParser, Task

prompt = hub.pull("wfh/llm-compiler")
print("prompt: ", prompt.pretty_print())



def create_planner(
    llm: BaseChatModel, tools: Sequence[BaseTool], base_prompt: ChatPromptTemplate
):
    """
    创建LLM编译器规划器，用于将用户查询转换为可执行的任务序列。
    
    该函数构建一个规划器，能够：
    1. 根据用户查询生成任务计划
    2. 支持重新规划（当需要修复之前的计划时）
    3. 自动处理任务索引的连续性
    
    参数:
        llm (BaseChatModel): 用于生成计划的语言模型实例
        tools (Sequence[BaseTool]): 可用的工具列表，规划器可以使用这些工具来执行任务
        base_prompt (ChatPromptTemplate): 基础提示模板，包含规划器的系统提示
    
    返回:
        Runnable: 一个可运行的规划器链，接受消息列表作为输入，输出任务流
        
    示例:
        输入:
            llm = ChatFireworks(model="...")
            tools = [search_tool, math_tool]
            prompt = hub.pull("wfh/llm-compiler")
            planner = create_planner(llm, tools, prompt)
            
            # 使用规划器
            tasks = planner.stream([HumanMessage(content="旧金山的温度的三次方是多少？")])
            for task in tasks:
                print(task["tool"], task["args"])
        
        输出:
            任务流，每个任务包含：
            - idx: 任务索引
            - tool: 要使用的工具
            - args: 工具参数
            - dependencies: 依赖的其他任务索引
            - thought: 思考过程（可选）
    """
    tool_descriptions = "\n".join(
        f"{i + 1}. {tool.description}\n"
        for i, tool in enumerate(
            tools
        )  # +1 以抵消从0开始的索引，我们希望从1开始正常计数。
    )
    planner_prompt = base_prompt.partial(
        replan="",
        num_tools=len(tools)
        + 1,  # 加1是因为我们在最后添加了 join() 工具。
        tool_descriptions=tool_descriptions,
    )
    replanner_prompt = base_prompt.partial(
        replan=' - 你被给予"先前计划"，这是先前代理创建的计划，以及每个计划的执行结果 '
        "（以观察的形式给出）和关于执行结果的一般思考（以思考的形式给出）。"
        '你必须使用这些信息在"当前计划"下创建下一个计划。\n'
        ' - 开始当前计划时，你应该以"思考"开始，概述下一个计划的策略。\n'
        " - 在当前计划中，你绝不应该重复先前计划中已经执行的操作。\n"
        " - 你必须从前一个计划的末尾继续任务索引。不要重复任务索引。",
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )

    def should_replan(state: list):
        """
        判断当前状态是否需要重新规划。
        
        当最后一条消息是SystemMessage时，表示需要基于之前的执行结果重新规划。
        
        参数:
            state (list): 消息状态列表，包含BaseMessage对象
        
        返回:
            bool: True表示需要重新规划，False表示首次规划
            
        示例:
            输入:
                state = [HumanMessage(content="问题"), FunctionMessage(...), SystemMessage(content="需要重新规划")]
            输出:
                True
                
            输入:
                state = [HumanMessage(content="问题")]
            输出:
                False
        """
        # 上下文作为系统消息传递
        return isinstance(state[-1], SystemMessage)

    def wrap_messages(state: list):
        """
        将消息列表包装为字典格式，用于传递给提示模板。
        
        参数:
            state (list): 消息状态列表
        
        返回:
            dict: 包含"messages"键的字典
            
        示例:
            输入:
                state = [HumanMessage(content="问题")]
            输出:
                {"messages": [HumanMessage(content="问题")]}
        """
        return {"messages": state}

    def wrap_and_get_last_index(state: list):
        """
        包装消息并获取最后一个任务索引，用于重新规划时保持任务索引的连续性。
        
        在重新规划时，需要从上一个计划的最后一个任务索引继续计数，避免索引重复。
        
        参数:
            state (list): 消息状态列表，可能包含之前的FunctionMessage
        
        返回:
            dict: 包含"messages"键的字典，最后一条SystemMessage的内容会被更新
            
        示例:
            输入:
                state = [
                    HumanMessage(content="问题"),
                    FunctionMessage(additional_kwargs={"idx": 3}, ...),
                    SystemMessage(content="需要重新规划")
                ]
            输出:
                {
                    "messages": [
                        HumanMessage(content="问题"),
                        FunctionMessage(additional_kwargs={"idx": 3}, ...),
                        SystemMessage(content="需要重新规划 - 从以下位置开始计数：4")
                    ]
                }
        """
        next_task = 0
        for message in state[::-1]:
            if isinstance(message, FunctionMessage):
                next_task = message.additional_kwargs["idx"] + 1
                break
        state[-1].content = state[-1].content + f" - 从以下位置开始计数：{next_task}"
        return {"messages": state}

    return (
        RunnableBranch[list, Any](
            (should_replan, wrap_and_get_last_index | replanner_prompt),
            wrap_messages | planner_prompt,
        )
        | llm
        | LLMCompilerPlanParser(tools=tools)
    )

print("====="*10)

print("create_planner")

# 这是我们应用程序中的主要"代理"
planner = create_planner(llm, tools, prompt)


example_question = "旧金山的温度的三次方是多少？"

for task in planner.stream([HumanMessage(content=example_question)]):
    print("tool: ", task["tool"])
    print("args: ", task["args"])
    print("---")

print("====="*10)

# exit()

import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, List, Union

from langchain_core.runnables import (
    chain as as_runnable,
)
from typing_extensions import TypedDict


def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    """
    从消息列表中提取所有工具执行的观察结果（输出）。
    
    该函数从消息列表中反向遍历，找到所有FunctionMessage（工具执行结果），
    并将其按任务索引组织成字典。
    
    参数:
        messages (List[BaseMessage]): 消息列表，可能包含FunctionMessage
        
    返回:
        Dict[int, Any]: 任务索引到观察结果的映射字典
        
    示例:
        输入:
            messages = [
                HumanMessage(content="问题"),
                FunctionMessage(additional_kwargs={"idx": 1}, content="32"),
                FunctionMessage(additional_kwargs={"idx": 2}, content="32768")
            ]
        输出:
            {1: "32", 2: "32768"}
            
        输入:
            messages = [HumanMessage(content="问题")]
        输出:
            {}
    """
    # 获取所有先前的工具响应
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    return results


class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterable[Task]


def _execute_task(task, observations, config):
    """
    执行单个任务，解析参数并调用相应的工具。
    
    该函数负责：
    1. 解析任务参数中的依赖引用（如$1, ${2}）
    2. 调用相应的工具执行任务
    3. 处理执行过程中的异常
    
    参数:
        task (Task): 任务对象，包含：
            - tool: 要使用的工具（BaseTool或字符串"join"）
            - args: 工具参数（可能是字符串或字典）
        observations (Dict[int, Any]): 之前任务的执行结果，用于解析依赖引用
        config (RunnableConfig): 运行配置对象
        
    返回:
        Any: 工具执行的结果，如果执行失败则返回错误信息字符串
        
    示例:
        输入:
            task = {
                "tool": math_tool,
                "args": {"problem": "$1 + 5", "context": None}
            }
            observations = {1: "32"}
            config = None
        输出:
            "37"  # 32 + 5的结果
            
        输入:
            task = {
                "tool": "join",
                "args": {}
            }
            observations = {}
            config = None
        输出:
            "join"  # 特殊工具，直接返回字符串
    """
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
            # 这很可能会失败
            resolved_args = args
    except Exception as e:
        return (
            f"错误（使用参数 {args} 调用 {tool_to_use.name} 失败。）"
            f" 参数无法解析。错误：{repr(e)}"
        )
    try:
        return tool_to_use.invoke(resolved_args, config)
    except Exception as e:
        return (
            f"错误（使用参数 {args} 调用 {tool_to_use.name} 失败。"
            + f" 参数解析为 {resolved_args}。错误：{repr(e)}）"
        )


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
    """
    解析参数中的依赖引用，将$1、${2}等占位符替换为实际观察值。
    
    该函数支持以下格式的依赖引用：
    - $1: 引用任务1的结果
    - ${2}: 引用任务2的结果（带花括号）
    
    参数:
        arg (Union[str, Any]): 需要解析的参数，可以是字符串、列表或其他类型
        observations (Dict[int, Any]): 任务索引到观察结果的映射
        
    返回:
        Union[str, List[str], Any]: 解析后的参数值
        
    示例:
        输入:
            arg = "$1 + 5"
            observations = {1: "32"}
        输出:
            "32 + 5"
            
        输入:
            arg = "计算 ${1} 和 ${2} 的和"
            observations = {1: "10", 2: "20"}
        输出:
            "计算 10 和 20 的和"
            
        输入:
            arg = ["$1", "$2"]
            observations = {1: "10", 2: "20"}
        输出:
            ["10", "20"]
            
        输入:
            arg = "$3"
            observations = {1: "10", 2: "20"}  # 没有任务3的结果
        输出:
            "$3"  # 如果观察值不存在，保持原样
    """
    # $1 or ${1} -> 1
    ID_PATTERN = r"\$\{?(\d+)\}?"

    def replace_match(match):
        """
        替换匹配到的依赖引用。
        
        参数:
            match: 正则表达式匹配对象
            
        返回:
            str: 替换后的字符串
        """
        # 如果字符串是 ${123}，match.group(0) 是 ${123}，match.group(1) 是 123。

        # 从字符串返回匹配组，在这种情况下是索引。这是我们返回的索引号。
        idx = int(match.group(1))
        return str(observations.get(idx, match.group(0)))

    # 用于其他任务的依赖关系
    if isinstance(arg, str):
        return re.sub(ID_PATTERN, replace_match, arg)
    elif isinstance(arg, list):
        return [_resolve_arg(a, observations) for a in arg]
    else:
        return str(arg)


@as_runnable
def schedule_task(task_inputs, config):
    """
    调度并执行单个任务，将执行结果存储到observations字典中。
    
    这是一个可运行的函数，用于在任务调度系统中执行任务。
    执行结果会被添加到observations字典中，供后续任务使用。
    
    参数:
        task_inputs (dict): 包含以下键的字典：
            - task (Task): 要执行的任务对象
            - observations (Dict[int, Any]): 所有任务的观察结果字典（会被修改）
        config (RunnableConfig): 运行配置对象
        
    返回:
        None: 函数不返回值，而是直接修改observations字典
        
    示例:
        输入:
            task_inputs = {
                "task": {
                    "idx": 1,
                    "tool": math_tool,
                    "args": {"problem": "2 + 3"}
                },
                "observations": {}
            }
            config = None
        执行后observations变为:
            {1: "5"}
            
        如果执行失败:
            task_inputs = {
                "task": {
                    "idx": 1,
                    "tool": math_tool,
                    "args": {"problem": "invalid expression"}
                },
                "observations": {}
            }
        执行后observations可能包含错误信息:
            {1: "错误（使用参数 {'problem': 'invalid expression'} 调用 math 失败。...）"}
    """
    task: Task = task_inputs["task"]
    observations: Dict[int, Any] = task_inputs["observations"]
    try:
        observation = _execute_task(task, observations, config)
    except Exception:
        import traceback

        observation = traceback.format_exception()  # repr(e) +
    observations[task["idx"]] = observation


def schedule_pending_task(
    task: Task, observations: Dict[int, Any], retry_after: float = 0.2
):
    """
    调度一个待处理的任务，等待其依赖项完成后再执行。
    
    该函数会循环检查任务的依赖项是否都已满足（即依赖的任务都已执行完成）。
    如果依赖未满足，会等待一段时间后重试，直到所有依赖都满足后再执行任务。
    
    参数:
        task (Task): 待执行的任务对象，包含dependencies字段
        observations (Dict[int, Any]): 所有任务的观察结果字典，用于检查依赖是否满足
        retry_after (float): 重试间隔时间（秒），默认0.2秒
        
    返回:
        None: 函数不返回值，执行结果会写入observations字典
        
    示例:
        输入:
            task = {
                "idx": 3,
                "tool": math_tool,
                "args": {"problem": "$1 + $2"},
                "dependencies": [1, 2]
            }
            observations = {1: "10"}  # 任务2还未完成
            retry_after = 0.2
        行为:
            - 检查依赖：任务1已完成，任务2未完成
            - 等待0.2秒
            - 再次检查（假设此时observations变为{1: "10", 2: "20"}）
            - 所有依赖满足，执行任务
            - observations变为{1: "10", 2: "20", 3: "30"}
            
        输入:
            task = {
                "idx": 1,
                "tool": math_tool,
                "args": {"problem": "2 + 3"},
                "dependencies": []  # 无依赖
            }
            observations = {}
            retry_after = 0.2
        行为:
            - 检查依赖：无依赖，立即执行
            - observations变为{1: "5"}
    """
    while True:
        deps = task["dependencies"]
        if deps and (any([dep not in observations for dep in deps])):
            # 依赖关系尚未满足
            time.sleep(retry_after)
            continue
        schedule_task.invoke({"task": task, "observations": observations})
        break


@as_runnable
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    """
    将任务分组为DAG（有向无环图）调度，并行执行无依赖的任务，串行执行有依赖的任务。
    
    该函数实现了任务调度系统，能够：
    1. 识别任务之间的依赖关系
    2. 立即执行无依赖或依赖已满足的任务
    3. 将依赖未满足的任务提交到线程池等待
    4. 将所有执行结果转换为FunctionMessage列表
    
    注意：对于流式处理，假设LLM不会创建循环依赖，也不会生成具有未来依赖的任务。
    
    参数:
        scheduler_input (SchedulerInput): 调度器输入，包含：
            - messages (List[BaseMessage]): 消息列表，用于提取之前的观察结果
            - tasks (Iterable[Task]): 要调度的任务迭代器
            
    返回:
        List[FunctionMessage]: 执行结果消息列表，每个消息对应一个任务
        
    示例:
        输入:
            scheduler_input = {
                "messages": [HumanMessage(content="问题")],
                "tasks": [
                    {
                        "idx": 1,
                        "tool": search_tool,
                        "args": {"query": "旧金山温度"},
                        "dependencies": []
                    },
                    {
                        "idx": 2,
                        "tool": math_tool,
                        "args": {"problem": "$1 + 5"},
                        "dependencies": [1]
                    }
                ]
            }
        输出:
            [
                FunctionMessage(
                    name="tavily_search_results_json",
                    content="旧金山温度是32度",
                    additional_kwargs={"idx": 1, "args": {...}},
                    tool_call_id=1
                ),
                FunctionMessage(
                    name="math",
                    content="37",
                    additional_kwargs={"idx": 2, "args": {...}},
                    tool_call_id=2
                )
            ]
    """
    # 对于流式处理，我们做了几个简化假设：
    # 1. LLM 不会创建循环依赖
    # 2. LLM 不会生成具有未来依赖的任务
    # 如果这不再是一个好的假设，你可以
    # 调整为进行适当的拓扑排序（非流式）
    # 或使用更复杂的数据结构
    tasks = scheduler_input["tasks"]
    args_for_tasks = {}
    messages = scheduler_input["messages"]
    # 如果我们正在重新规划，我们可能有依赖于先前计划的调用。从这些开始。
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)
    # ^^ 我们假设每个任务在上面插入不同的键以
    # 避免竞争条件...
    futures = []
    retry_after = 0.25  # 每四分之一秒重试一次
    with ThreadPoolExecutor() as executor:
        for task in tasks:
            deps = task["dependencies"]
            task_names[task["idx"]] = (
                task["tool"] if isinstance(task["tool"], str) else task["tool"].name
            )
            args_for_tasks[task["idx"]] = task["args"]
            if (
                # 依赖于其他任务
                deps and (any([dep not in observations for dep in deps]))
            ):
                futures.append(
                    executor.submit(
                        schedule_pending_task, task, observations, retry_after
                    )
                )
            else:
                # 没有依赖或所有依赖已满足
                # 现在可以调度
                schedule_task.invoke(dict(task=task, observations=observations))
                # futures.append(executor.submit(schedule_task.invoke, dict(task=task, observations=observations)))

        # 所有任务已提交或入队
        # 等待它们完成
        wait(futures)
    # 将观察结果转换为新的工具消息以添加到状态
    new_observations = {
        k: (task_names[k], args_for_tasks[k], observations[k])
        for k in sorted(observations.keys() - originals)
    }
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

import itertools


@as_runnable
def plan_and_schedule(state):
    """
    规划并调度任务：首先使用规划器生成任务计划，然后调度执行这些任务。
    
    这是工作流中的核心节点，负责：
    1. 从状态中获取消息
    2. 使用规划器生成任务流
    3. 立即开始执行规划器（通过预取第一个任务）
    4. 将任务提交给调度器执行
    5. 返回执行结果消息
    
    参数:
        state (dict): 状态字典，包含：
            - messages (List[BaseMessage]): 当前的消息列表
            
    返回:
        dict: 包含执行结果消息的字典：
            - messages (List[FunctionMessage]): 任务执行结果消息列表
            
    示例:
        输入:
            state = {
                "messages": [HumanMessage(content="旧金山的温度的三次方是多少？")]
            }
        输出:
            {
                "messages": [
                    FunctionMessage(
                        name="tavily_search_results_json",
                        content="旧金山温度是32度",
                        additional_kwargs={"idx": 1, ...}
                    ),
                    FunctionMessage(
                        name="math",
                        content="32768",
                        additional_kwargs={"idx": 2, ...}
                    )
                ]
            }
            
        如果规划器返回空任务：
            输入:
                state = {"messages": [HumanMessage(content="简单问题")]}
            输出:
                {"messages": []}
    """
    messages = state["messages"]
    tasks = planner.stream(messages)
    # 立即开始执行规划器
    try:
        tasks = itertools.chain([next(tasks)], tasks)
    except StopIteration:
        # 处理任务为空的情况。
        tasks = iter([])
    scheduled_tasks = schedule_tasks.invoke(
        {
            "messages": messages,
            "tasks": tasks,
        }
    )
    return {"messages": scheduled_tasks}


tool_messages = plan_and_schedule.invoke(
    {"messages": [HumanMessage(content=example_question)]}
)["messages"]

print("====="*10)   

print("tool_messages: ", tool_messages)


from langchain_core.messages import AIMessage

from pydantic import BaseModel, Field


class FinalResponse(BaseModel):
    """最终响应/答案。"""

    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="对先前尝试的分析以及关于需要修复什么内容的建议。"
    )


class JoinOutputs(BaseModel):
    """决定是重新规划还是可以返回最终响应。"""

    thought: str = Field(
        description="所选行动的思维链推理"
    )
    action: Union[FinalResponse, Replan]


joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(
    examples=""
)  # 你可以选择性地添加示例

print("====="*10)

print("joiner_prompt: ", joiner_prompt.pretty_print())

runnable = joiner_prompt | llm.with_structured_output(
    JoinOutputs, method="function_calling"
)

def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    """
    解析连接器的输出决策，将其转换为消息列表。
    
    连接器会决定是返回最终答案还是需要重新规划。该函数根据决策类型
    生成相应的消息：
    - 如果需要重新规划：返回思考消息和包含反馈的系统消息
    - 如果返回最终答案：返回思考消息和包含答案的AI消息
    
    参数:
        decision (JoinOutputs): 连接器的输出决策，包含：
            - thought (str): 思考过程
            - action (Union[FinalResponse, Replan]): 行动决策
            
    返回:
        dict: 包含消息列表的字典：
            - messages (List[BaseMessage]): 生成的消息列表
            
    示例:
        输入（需要重新规划）:
            decision = JoinOutputs(
                thought="之前的计算可能有误，需要重新检查",
                action=Replan(feedback="温度值可能不正确，需要重新搜索")
            )
        输出:
            {
                "messages": [
                    AIMessage(content="思考：之前的计算可能有误，需要重新检查"),
                    SystemMessage(content="上次尝试的上下文：温度值可能不正确，需要重新搜索")
                ]
            }
            
        输入（返回最终答案）:
            decision = JoinOutputs(
                thought="所有计算已完成，可以返回答案",
                action=FinalResponse(response="旧金山温度的三次方是32768")
            )
        输出:
            {
                "messages": [
                    AIMessage(content="思考：所有计算已完成，可以返回答案"),
                    AIMessage(content="旧金山温度的三次方是32768")
                ]
            }
    """
    response = [AIMessage(content=f"思考：{decision.thought}")]
    if isinstance(decision.action, Replan):
        return {
            "messages": response
            + [
                SystemMessage(
                    content=f"上次尝试的上下文：{decision.action.feedback}"
                )
            ]
        }
    else:
        return {"messages": response + [AIMessage(content=decision.action.response)]}


def select_recent_messages(state) -> dict:
    """
    从状态中选择最近的消息，从最后一个HumanMessage开始到消息列表末尾。
    
    该函数用于连接器节点，只选择与当前查询相关的最新消息，
    避免将整个对话历史都传递给连接器。
    
    参数:
        state (dict): 状态字典，包含：
            - messages (List[BaseMessage]): 完整的消息列表
            
    返回:
        dict: 包含选中消息的字典：
            - messages (List[BaseMessage]): 从最后一个HumanMessage到末尾的消息列表
            
    示例:
        输入:
            state = {
                "messages": [
                    HumanMessage(content="第一个问题"),
                    FunctionMessage(...),
                    HumanMessage(content="第二个问题"),
                    FunctionMessage(...),
                    FunctionMessage(...)
                ]
            }
        输出:
            {
                "messages": [
                    HumanMessage(content="第二个问题"),
                    FunctionMessage(...),
                    FunctionMessage(...)
                ]
            }
            
        输入:
            state = {
                "messages": [HumanMessage(content="问题")]
            }
        输出:
            {
                "messages": [HumanMessage(content="问题")]
            }
    """
    messages = state["messages"]
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage):
            break
    return {"messages": selected[::-1]}


joiner = select_recent_messages | runnable | _parse_joiner_output

print("====="*10)

print("joiner: ", joiner)

input_messages = [HumanMessage(content=example_question)] + tool_messages

print("====="*10)

print("input_messages: ", input_messages)

print("====="*10)

print("joiner.invoke: ", joiner.invoke({"messages": input_messages}))

print("====="*10)


from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

# 1. 定义顶点
# 我们已经在上面定义了 plan_and_schedule
# 将每个节点分配给要更新的状态变量
graph_builder.add_node("plan_and_schedule", plan_and_schedule)
graph_builder.add_node("join", joiner)


## 定义边
graph_builder.add_edge("plan_and_schedule", "join")

### 此条件决定循环逻辑


def should_continue(state):
    """
    判断工作流是否应该继续执行，还是应该结束。
    
    该函数用于LangGraph的条件边，决定从"join"节点后的下一步：
    - 如果最后一条消息是AIMessage（包含最终答案），则结束工作流
    - 否则，返回"plan_and_schedule"继续规划和执行
    
    参数:
        state (dict): 状态字典，包含：
            - messages (List[BaseMessage]): 当前的消息列表
            
    返回:
        Union[str, END]: 
            - END: 表示工作流应该结束
            - "plan_and_schedule": 表示应该继续规划和执行
            
    示例:
        输入（应该结束）:
            state = {
                "messages": [
                    HumanMessage(content="问题"),
                    FunctionMessage(...),
                    AIMessage(content="最终答案：32768")
                ]
            }
        输出:
            END
            
        输入（应该继续）:
            state = {
                "messages": [
                    HumanMessage(content="问题"),
                    FunctionMessage(...),
                    SystemMessage(content="需要重新规划")
                ]
            }
        输出:
            "plan_and_schedule"
    """
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage):
        return END
    return "plan_and_schedule"

print("====="*10)

print("graph_builder: ", graph_builder)

graph_builder.add_conditional_edges(
    "join",
    # 接下来，我们传入将决定接下来调用哪个节点的函数。
    should_continue,
)
graph_builder.add_edge(START, "plan_and_schedule")
chain = graph_builder.compile()

print("====="*10)

print("chain: ", chain)

print("====="*10)

print("chain.stream: ")

# for step in chain.stream(
#     {"messages": [HumanMessage(content="纽约的GDP是多少？")]}
# ):
#     print("step: ", step)
#     print("---")


# # 最终答案

# # Final answer
# print(step["join"]["messages"][-1].content)


for step in chain.stream(
    {
        "messages": [
            HumanMessage(
                content="((3*(4+5)/0.5)+3245) + 8 是多少？32/4.23 是多少？这两个值的和是多少？"
            )
        ]
    }
):
    print(step)
    print('---')

# Final answer
print(step["join"]["messages"][-1].content)
