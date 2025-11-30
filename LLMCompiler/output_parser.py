import ast
import re
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from typing_extensions import TypedDict

THOUGHT_PATTERN = r"Thought: ([^\n]*)"
ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
# $1 or ${1} -> 1
ID_PATTERN = r"\$\{?(\d+)\}?"
END_OF_PLAN = "<END_OF_PLAN>"


### Helper functions


def _ast_parse(arg: str) -> Any:
    """
    使用AST安全地解析字符串字面量，将其转换为Python对象。
    
    该函数使用ast.literal_eval()来安全地解析字符串，只支持Python字面量：
    - 数字（整数、浮点数、复数）
    - 字符串
    - 列表、元组、字典
    - 布尔值、None
    
    如果解析失败，则返回原始字符串。
    
    参数:
        arg (str): 要解析的字符串字面量
        
    返回:
        Any: 解析后的Python对象，如果解析失败则返回原始字符串
        
    示例:
        输入:
            arg = "123"
        输出:
            123  # int
            
        输入:
            arg = "3.14"
        输出:
            3.14  # float
            
        输入:
            arg = "'hello'"
        输出:
            "hello"  # str
            
        输入:
            arg = "[1, 2, 3]"
        输出:
            [1, 2, 3]  # list
            
        输入:
            arg = "{'key': 'value'}"
        输出:
            {"key": "value"}  # dict
            
        输入（无法解析）:
            arg = "some text"
        输出:
            "some text"  # 返回原字符串
    """
    try:
        return ast.literal_eval(arg)
    except:  # noqa
        return arg


def _parse_llm_compiler_action_args(args: str, tool: Union[str, BaseTool]) -> list[Any]:
    """
    从字符串中解析LLM编译器动作的参数。
    
    该函数解析类似"key1=value1, key2=value2"格式的参数字符串，
    将其转换为字典格式。参数值会使用AST解析以支持复杂类型。
    
    参数:
        args (str): 参数字符串，格式为"key1=value1, key2=value2"或空字符串
        tool (Union[str, BaseTool]): 工具对象或字符串"join"
            - 如果是BaseTool，使用tool.args.keys()获取参数名
            - 如果是"join"，返回空字典
            
    返回:
        dict: 解析后的参数字典，键为参数名，值为解析后的参数值
        
    示例:
        输入:
            args = "problem='2 + 3'"
            tool = math_tool  # 有problem参数
        输出:
            {"problem": "2 + 3"}
            
        输入:
            args = "query='旧金山', max_results=5"
            tool = search_tool  # 有query和max_results参数
        输出:
            {"query": "旧金山", "max_results": 5}
            
        输入:
            args = "problem='$1 + 5', context=['$1']"
            tool = math_tool
        输出:
            {"problem": "$1 + 5", "context": ["$1"]}
            
        输入:
            args = ""
            tool = math_tool
        输出:
            {}
            
        输入:
            args = "anything"
            tool = "join"
        输出:
            {}
    """
    if args == "":
        return ()
    if isinstance(tool, str):
        return ()
    extracted_args = {}
    tool_key = None
    prev_idx = None
    for key in tool.args.keys():
        # Split if present
        if f"{key}=" in args:
            idx = args.index(f"{key}=")
            if prev_idx is not None:
                extracted_args[tool_key] = _ast_parse(
                    args[prev_idx:idx].strip().rstrip(",")
                )
            args = args.split(f"{key}=", 1)[1]
            tool_key = key
            prev_idx = 0
    if prev_idx is not None:
        extracted_args[tool_key] = _ast_parse(
            args[prev_idx:].strip().rstrip(",").rstrip(")")
        )
    return extracted_args


def default_dependency_rule(idx, args: str):
    """
    检查任务索引是否在参数字符串中被引用（作为依赖）。
    
    该函数通过正则表达式查找参数中的$1、${2}等依赖引用，
    判断给定的任务索引是否被引用。
    
    参数:
        idx (int): 要检查的任务索引
        args (str): 参数字符串，可能包含$1、${2}等依赖引用
        
    返回:
        bool: True表示该任务被引用（是依赖），False表示不是依赖
        
    示例:
        输入:
            idx = 1
            args = "problem='$1 + 5'"
        输出:
            True  # 参数中引用了$1
            
        输入:
            idx = 2
            args = "problem='$1 + 5'"
        输出:
            False  # 参数中没有引用$2
            
        输入:
            idx = 3
            args = "context=['${1}', '${3}']"
        输出:
            True  # 参数中引用了${3}
            
        输入:
            idx = 1
            args = "query='旧金山'"
        输出:
            False  # 参数中没有依赖引用
    """
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers


def _get_dependencies_from_graph(
    idx: int, tool_name: str, args: Dict[str, Any]
) -> dict[str, list[str]]:
    """
    从任务图中获取任务的依赖关系。
    
    该函数根据任务索引、工具名称和参数来确定任务的依赖：
    - 对于"join"工具，依赖所有之前的任务（1到idx-1）
    - 对于其他工具，使用default_dependency_rule检查哪些之前的任务被引用
    
    参数:
        idx (int): 当前任务的索引
        tool_name (str): 工具名称，如"math"、"tavily_search_results_json"或"join"
        args (Dict[str, Any]): 任务的参数字典
        
    返回:
        List[int]: 依赖的任务索引列表
        
    示例:
        输入:
            idx = 3
            tool_name = "join"
            args = {}
        输出:
            [1, 2]  # join工具依赖所有之前的任务
            
        输入:
            idx = 2
            tool_name = "math"
            args = {"problem": "$1 + 5"}
        输出:
            [1]  # 任务2依赖任务1（因为参数中引用了$1）
            
        输入:
            idx = 4
            tool_name = "math"
            args = {"problem": "$1 + $2", "context": ["$3"]}
        输出:
            [1, 2, 3]  # 任务4依赖任务1、2、3
            
        输入:
            idx = 2
            tool_name = "tavily_search_results_json"
            args = {"query": "旧金山"}
        输出:
            []  # 没有依赖引用
    """
    if tool_name == "join":
        return list(range(1, idx))
    return [i for i in range(1, idx) if default_dependency_rule(i, str(args))]


class Task(TypedDict):
    idx: int
    tool: BaseTool
    args: list
    dependencies: Dict[str, list]
    thought: Optional[str]


def instantiate_task(
    tools: Sequence[BaseTool],
    idx: int,
    tool_name: str,
    args: Union[str, Any],
    thought: Optional[str] = None,
) -> Task:
    """
    实例化一个任务对象，从解析的LLM输出中创建完整的任务结构。
    
    该函数将LLM输出的任务信息（索引、工具名、参数字符串）转换为
    完整的Task对象，包括：
    - 解析参数字符串为字典
    - 查找对应的工具对象
    - 计算依赖关系
    
    参数:
        tools (Sequence[BaseTool]): 可用工具列表
        idx (int): 任务索引
        tool_name (str): 工具名称，如"math"、"tavily_search_results_json"或"join"
        args (Union[str, Any]): 参数字符串或字典
        thought (Optional[str]): 可选的思考过程
        
    返回:
        Task: 完整的任务对象，包含：
            - idx: 任务索引
            - tool: 工具对象或"join"字符串
            - args: 解析后的参数字典
            - dependencies: 依赖的任务索引列表
            - thought: 思考过程
            
    异常:
        OutputParserException: 当工具名称在tools列表中找不到时抛出
        
    示例:
        输入:
            tools = [search_tool, math_tool]
            idx = 1
            tool_name = "tavily_search_results_json"
            args = "query='旧金山'"
            thought = "需要搜索旧金山的信息"
        输出:
            Task(
                idx=1,
                tool=search_tool,
                args={"query": "旧金山"},
                dependencies=[],
                thought="需要搜索旧金山的信息"
            )
            
        输入:
            tools = [search_tool, math_tool]
            idx = 2
            tool_name = "math"
            args = "problem='$1 + 5'"
            thought = None
        输出:
            Task(
                idx=2,
                tool=math_tool,
                args={"problem": "$1 + 5"},
                dependencies=[1],  # 依赖任务1
                thought=None
            )
            
        输入:
            tools = [search_tool, math_tool]
            idx = 3
            tool_name = "join"
            args = ""
            thought = None
        输出:
            Task(
                idx=3,
                tool="join",
                args={},
                dependencies=[1, 2],  # join依赖所有之前的任务
                thought=None
            )
    """
    if tool_name == "join":
        tool = "join"
    else:
        try:
            tool = tools[[tool.name for tool in tools].index(tool_name)]
        except ValueError as e:
            raise OutputParserException(f"Tool {tool_name} not found.") from e
    tool_args = _parse_llm_compiler_action_args(args, tool)
    dependencies = _get_dependencies_from_graph(idx, tool_name, tool_args)

    return Task(
        idx=idx,
        tool=tool,
        args=tool_args,
        dependencies=dependencies,
        thought=thought,
    )


class LLMCompilerPlanParser(BaseTransformOutputParser[dict], extra="allow"):
    """Planning output parser."""

    tools: List[BaseTool]

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[Task]:
        """
        将输入流转换为任务流，实时解析LLM输出并生成任务对象。
        
        该方法是流式解析的核心，它：
        1. 逐个处理输入块（token或消息）
        2. 使用缓冲区累积文本直到遇到换行符
        3. 解析每行文本，提取任务信息
        4. 生成Task对象
        
        参数:
            input (Iterator[Union[str, BaseMessage]]): 输入迭代器，包含字符串块或消息对象
            
        返回:
            Iterator[Task]: 任务对象迭代器
            
        示例:
            输入:
                input = iter([
                    "Thought: 需要搜索信息\n",
                    "1. tavily_search_results_json(query='旧金山')\n",
                    "2. math(problem='$1 + 5')\n"
                ])
            输出:
                迭代器产生:
                Task(idx=1, tool=search_tool, args={"query": "旧金山"}, ...)
                Task(idx=2, tool=math_tool, args={"problem": "$1 + 5"}, ...)
        """
        texts = []
        # TODO: Cleanup tuple state tracking here.
        thought = None
        for chunk in input:
            # Assume input is str. TODO: support vision/other formats
            text = chunk if isinstance(chunk, str) else str(chunk.content)
            for task, thought in self.ingest_token(text, texts, thought):
                yield task
        # Final possible task
        if texts:
            task, _ = self._parse_task("".join(texts), thought)
            if task:
                yield task

    def parse(self, text: str) -> List[Task]:
        """
        解析完整的文本字符串，返回所有任务列表。
        
        这是非流式解析方法，用于一次性解析完整的LLM输出文本。
        
        参数:
            text (str): 完整的LLM输出文本，可能包含多行任务
            
        返回:
            List[Task]: 解析出的所有任务列表
            
        示例:
            输入:
                text = Thought: 需要计算
                1. math(problem='2 + 3')
                2. math(problem='$1 * 2')
            输出:
                [
                    Task(idx=1, tool=math_tool, args={"problem": "2 + 3"}, ...),
                    Task(idx=2, tool=math_tool, args={"problem": "$1 * 2"}, ...)
                ]
        """
        return list(self._transform([text]))

    def stream(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Task]:
        """
        流式解析输入，返回任务迭代器。
        
        这是流式解析的入口方法，将单个输入转换为任务流。
        
        参数:
            input (str | BaseMessage): 输入字符串或消息对象
            config (RunnableConfig | None): 可选的运行配置
            **kwargs: 其他关键字参数
            
        返回:
            Iterator[Task]: 任务对象迭代器
            
        示例:
            输入:
                input = "1. math(problem='2 + 3')\n2. math(problem='$1 * 2')"
                config = None
            输出:
                迭代器产生:
                Task(idx=1, tool=math_tool, args={"problem": "2 + 3"}, ...)
                Task(idx=2, tool=math_tool, args={"problem": "$1 * 2"}, ...)
        """
        yield from self.transform([input], config, **kwargs)

    def ingest_token(
        self, token: str, buffer: List[str], thought: Optional[str]
    ) -> Iterator[Tuple[Optional[Task], str]]:
        """
        处理输入的token，累积到缓冲区，当遇到换行符时解析完整行。
        
        该方法是增量解析的核心，它：
        1. 将token添加到缓冲区
        2. 当token包含换行符时，分割缓冲区
        3. 解析每行文本，提取任务或思考
        4. 保留最后不完整的行在缓冲区中
        
        参数:
            token (str): 输入的文本块（可能是一个字符或多个字符）
            buffer (List[str]): 文本缓冲区，用于累积不完整的行
            thought (Optional[str]): 当前的思考内容，可能被更新
            
        返回:
            Iterator[Tuple[Optional[Task], str]]: 任务和思考的元组迭代器
                - Task: 如果解析出任务则返回Task对象，否则为None
                - str: 更新后的思考内容
                
        示例:
            输入:
                token = "Thought: 需要计算\n"
                buffer = []
                thought = None
            输出:
                迭代器产生: (None, "需要计算")
                
            输入:
                token = "1. math(problem='2+3')\n"
                buffer = []
                thought = "需要计算"
            输出:
                迭代器产生: (Task(idx=1, ...), None)
                
            输入:
                token = "1. math(pro"
                buffer = []
                thought = None
            输出:
                迭代器产生: (None, None)  # 不完整，不解析
                # buffer变为["1. math(pro"]
        """
        buffer.append(token)
        if "\n" in token:
            buffer_ = "".join(buffer).split("\n")
            suffix = buffer_[-1]
            for line in buffer_[:-1]:
                task, thought = self._parse_task(line, thought)
                if task:
                    yield task, thought
            buffer.clear()
            buffer.append(suffix)

    def _parse_task(self, line: str, thought: Optional[str] = None):
        """
        解析单行文本，提取任务或思考信息。
        
        该函数使用正则表达式匹配两种模式：
        1. "Thought: ..." - 提取思考内容
        2. "数字. 工具名(参数)" - 提取任务信息
        
        参数:
            line (str): 要解析的单行文本
            thought (Optional[str]): 当前的思考内容，可能被更新
            
        返回:
            Tuple[Optional[Task], Optional[str]]: 
                - Task: 如果解析出任务则返回Task对象，否则为None
                - str: 更新后的思考内容，如果没有思考则为None
                
        示例:
            输入:
                line = "Thought: 需要搜索信息"
                thought = None
            输出:
                (None, "需要搜索信息")
                
            输入:
                line = "1. tavily_search_results_json(query='旧金山')"
                thought = "需要搜索信息"
            输出:
                (Task(idx=1, tool=search_tool, args={"query": "旧金山"}, thought="需要搜索信息", ...), None)
                
            输入:
                line = "2. math(problem='$1 + 5')"
                thought = None
            输出:
                (Task(idx=2, tool=math_tool, args={"problem": "$1 + 5"}, thought=None, ...), None)
                
            输入:
                line = "普通文本，不匹配任何模式"
                thought = None
            输出:
                (None, None)  # 不匹配，被丢弃
        """
        task = None
        if match := re.match(THOUGHT_PATTERN, line):
            # Optionally, action can be preceded by a thought
            thought = match.group(1)
        elif match := re.match(ACTION_PATTERN, line):
            # if action is parsed, return the task, and clear the buffer
            idx, tool_name, args, _ = match.groups()
            idx = int(idx)
            task = instantiate_task(
                tools=self.tools,
                idx=idx,
                tool_name=tool_name,
                args=args,
                thought=thought,
            )
            thought = None
        # Else it is just dropped
        return task, thought