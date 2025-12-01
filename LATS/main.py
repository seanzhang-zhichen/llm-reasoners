"""
LATS (Language Agent Tree Search) 实现

该模块实现了基于蒙特卡洛树搜索 (MCTS) 的语言代理，结合 LLM 和工具使用能力，
通过树搜索的方式迭代改进回答质量。

主要组件:
    - Reflection: 反思评估类，用于评价回答质量
    - Node: MCTS 树节点，存储状态和统计信息
    - TreeState: 状态类型定义
    - LATS 核心函数: select, expand, generate_initial_response 等
"""

# =============================================================================
# 1. 导入依赖
# =============================================================================
import os
import math
import dotenv
from collections import defaultdict, deque
from typing import Optional

from typing_extensions import TypedDict

# LangChain 核心组件
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import chain as as_runnable, RunnableConfig

# LangChain 社区工具
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# LangGraph
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

# LLM 提供商
from langchain_fireworks import ChatFireworks
from langchain_openai import ChatOpenAI 

# Pydantic
from pydantic import BaseModel, Field


# =============================================================================
# 2. 类定义
# =============================================================================

class Reflection(BaseModel):
    """
    反思类：用于评估和反思助手回答的质量
    
    该类作为 Pydantic 模型，用于结构化存储对候选答案的评估结果。
    在 LATS 算法中，每次生成候选答案后都会进行反思评估。
    
    属性:
        reflections (str): 对回答的批评和反思内容
        score (int): 回答质量评分，范围 0-10
        found_solution (bool): 是否已完全解决问题
    """
    reflections: str = Field(
        description="The critique and reflections on the sufficiency, superfluency,"
        " and general quality of the response"
    )
    score: int = Field(
        description="Score from 0-10 on the quality of the candidate response.",
        gte=0,
        lte=10,
    )
    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task."
    )

    def as_message(self) -> HumanMessage:
        """
        将反思结果转换为 HumanMessage 格式
        
        输入: 无
        输出: HumanMessage - 包含反思内容和评分的消息对象
        
        用途: 在构建对话轨迹时，将反思结果作为消息加入上下文
        """
        print(f"  [Reflection.as_message] 转换反思为消息: Score={self.score}")
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        """
        获取归一化评分
        
        输入: 无
        输出: float - 归一化后的评分 (0.0 ~ 1.0)
        
        用途: 在反向传播更新节点值时使用归一化评分
        """
        return self.score / 10.0


class Node:
    """
    蒙特卡洛树搜索 (MCTS) 节点类
    
    该类表示搜索树中的一个节点，存储了候选答案、反思结果、
    父子关系以及 MCTS 所需的统计信息（访问次数、累积价值）。
    
    属性:
        messages (list[BaseMessage]): 该节点对应的消息列表（答案内容）
        parent (Optional[Node]): 父节点引用
        children (list[Node]): 子节点列表
        value (float): 节点累积价值（平均奖励）
        visits (int): 节点被访问次数
        reflection (Reflection): 对该节点答案的反思评估
        depth (int): 节点在树中的深度
        _is_solved (bool): 是否找到解决方案
    """
    
    def __init__(
        self,
        messages: list[BaseMessage],
        reflection: Reflection,
        parent: Optional["Node"] = None,
    ):
        """
        初始化节点
        
        输入:
            messages (list[BaseMessage]): 该节点的消息列表
            reflection (Reflection): 对该节点的反思评估结果
            parent (Optional[Node]): 父节点，根节点为 None
        
        输出: 无（构造函数）
        
        流程:
            1. 初始化节点属性
            2. 如果找到解决方案，标记整棵树
            3. 反向传播评分到祖先节点
        """
        print(f"\n=== 创建新节点 ===")
        print(f"  深度: {parent.depth + 1 if parent else 1}")
        print(f"  消息数量: {len(messages)}")
        
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution if reflection else False
        
        print(f"  反思评分: {reflection.score}/10, 归一化: {reflection.normalized_score:.2f}")
        print(f"  是否解决: {self._is_solved}")
        
        if self._is_solved:
            print("  >>> 找到解决方案! 标记整棵树为已解决 <<<")
            self._mark_tree_as_solved()
        
        print(f"  开始反向传播评分: {reflection.normalized_score:.2f}")
        self.backpropagate(reflection.normalized_score)

    def __repr__(self) -> str:
        """节点的字符串表示"""
        return (
            f"<Node depth={self.depth}, value={self.value:.3f}, visits={self.visits},"
            f" solved={self._is_solved}/>"
        )

    # -------------------------------------------------------------------------
    # 属性方法
    # -------------------------------------------------------------------------
    
    @property
    def is_solved(self) -> bool:
        """
        检查是否已找到解决方案
        
        输入: 无
        输出: bool - 如果当前节点或其子树中存在解决方案则返回 True
        
        用途: 用于判断搜索是否可以提前终止
        """
        return self._is_solved

    @property
    def is_terminal(self) -> bool:
        """
        检查是否为叶子节点
        
        输入: 无
        输出: bool - 如果没有子节点则返回 True
        
        用途: 在选择最佳解决方案时，只考虑叶子节点
        """
        return not self.children

    @property
    def best_child_score(self) -> Optional["Node"]:
        """
        获取最佳子节点
        
        输入: 无
        输出: Node 或 None - 价值最高的子节点
        
        选择策略: 优先考虑已解决的节点，然后按价值排序
        """
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """
        计算以当前节点为根的子树高度
        
        输入: 无
        输出: int - 子树的高度（最深叶子节点的深度）
        
        用途: 用于控制搜索深度，防止无限展开
        """
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    # -------------------------------------------------------------------------
    # MCTS 核心方法
    # -------------------------------------------------------------------------
    
    def upper_confidence_bound(self, exploration_weight: float = 1.0) -> float:
        """
        计算 UCB (Upper Confidence Bound) 分数
        
        输入:
            exploration_weight (float): 探索权重，默认 1.0
                - 值越大越倾向探索未访问的节点
                - 值越小越倾向利用已知高价值节点
        
        输出: float - UCB 分数
        
        公式: UCB = 平均奖励 + 探索权重 * sqrt(ln(父访问次数) / 当前访问次数)
        
        用途: 在 select 阶段选择下一个要展开的节点，平衡探索与利用
        """
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        
        # 利用项：鼓励选择高价值轨迹
        average_reward = self.value / self.visits
        # 探索项：鼓励选择访问次数少的轨迹
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float) -> None:
        """
        反向传播奖励值到祖先节点
        
        输入:
            reward (float): 奖励值（归一化评分 0.0 ~ 1.0）
        
        输出: 无
        
        算法:
            从当前节点向上遍历到根节点，更新每个节点的:
            - visits: 访问次数 +1
            - value: 增量平均 = (旧值 * 旧访问次数 + 新奖励) / 新访问次数
        
        用途: MCTS 核心步骤，用于更新节点的价值估计
        """
        print(f"\n  === 反向传播 (Backpropagate) ===")
        print(f"  奖励值: {reward:.3f}")
        
        node = self
        path_info = []
        
        while node:
            old_value = node.value
            old_visits = node.visits
            node.visits += 1
            node.value = (node.value * old_visits + reward) / node.visits
            path_info.append(
                f"深度{node.depth}: 访问{old_visits}->{node.visits}, "
                f"价值{old_value:.3f}->{node.value:.3f}"
            )
            node = node.parent
        
        print(f"  传播路径: {' -> '.join(path_info)}")

    # -------------------------------------------------------------------------
    # 轨迹和消息方法
    # -------------------------------------------------------------------------
    
    def get_messages(self, include_reflections: bool = True) -> list[BaseMessage]:
        """
        获取当前节点的消息列表
        
        输入:
            include_reflections (bool): 是否包含反思消息，默认 True
        
        输出: list[BaseMessage] - 消息列表
        
        用途: 构建对话上下文时使用
        """
        if include_reflections:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        """
        获取从根节点到当前节点的完整对话轨迹
        
        输入:
            include_reflections (bool): 是否包含反思消息，默认 True
        
        输出: list[BaseMessage] - 按时间顺序排列的消息列表
            格式: [根节点答案, 根节点反思, 子节点1答案, 子节点1反思, ...]
        
        算法: 从当前节点回溯到根节点，收集所有消息后反转
        
        用途: 
            - 作为 LLM 的上下文输入
            - 展示最终的解决方案轨迹
        """
        print(f"\n  [get_trajectory] 获取轨迹，当前深度: {self.depth}, 包含反思: {include_reflections}")
        
        messages = []
        node = self
        node_count = 0
        
        while node:
            messages.extend(
                node.get_messages(include_reflections=include_reflections)[::-1]
            )
            node_count += 1
            node = node.parent
        
        print(f"  [get_trajectory] 遍历了 {node_count} 个节点，总消息数: {len(messages)}")
        
        # 反转回溯得到的轨迹，返回正确的时间顺序
        return messages[::-1]

    # -------------------------------------------------------------------------
    # 树操作方法
    # -------------------------------------------------------------------------
    
    def _get_all_children(self) -> list["Node"]:
        """
        获取当前节点的所有后代节点（BFS 遍历）
        
        输入: 无
        输出: list[Node] - 所有后代节点列表（不包含当前节点）
        
        算法: 使用队列进行广度优先遍历
        
        用途: 在 get_best_solution 中遍历整棵子树寻找最佳解
        """
        all_nodes = []
        nodes = deque([self])
        
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            nodes.extend(node.children)
        
        print(f"  [_get_all_children] 找到 {len(all_nodes)} 个后代节点")
        return all_nodes

    def get_best_solution(self) -> "Node":
        """
        在当前子树中找到最佳解决方案
        
        输入: 无
        输出: Node - 最佳解决方案节点
        
        选择标准:
            1. 必须是叶子节点 (is_terminal)
            2. 必须已解决问题 (is_solved)
            3. 在满足上述条件的节点中选择价值最高的
        
        用途: 搜索结束后获取最终答案
        """
        print("\n=== 寻找最佳解决方案 ===")
        
        all_nodes = [self] + self._get_all_children()
        print(f"  总节点数: {len(all_nodes)}")
        
        # 统计各类节点
        terminal_nodes = [n for n in all_nodes if n.is_terminal]
        solved_nodes = [n for n in all_nodes if n.is_solved]
        terminal_solved = [n for n in all_nodes if n.is_terminal and n.is_solved]
        print(f"  叶子节点: {len(terminal_nodes)}, 已解决: {len(solved_nodes)}, "
              f"叶子且已解决: {len(terminal_solved)}")
        
        best_node = max(
            all_nodes,
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        
        print(f"  最佳节点: 深度={best_node.depth}, 价值={best_node.value:.3f}, "
              f"访问={best_node.visits}, 已解决={best_node.is_solved}")
        
        return best_node

    def _mark_tree_as_solved(self) -> None:
        """
        将整棵树标记为已解决
        
        输入: 无
        输出: 无
        
        算法: 从当前节点向上遍历，将所有祖先节点的 _is_solved 设为 True
        
        用途: 当某个节点找到解决方案时，整棵树都应该知道已有解
        """
        print("  [_mark_tree_as_solved] 标记祖先节点为已解决")
        
        parent = self.parent
        marked_count = 0
        
        while parent:
            parent._is_solved = True
            marked_count += 1
            parent = parent.parent
        
        print(f"  [_mark_tree_as_solved] 共标记 {marked_count} 个祖先节点")


class TreeState(TypedDict):
    """
    LATS 状态类型定义
    
    属性:
        root (Node): 搜索树的根节点
        input (str): 原始用户输入
    """
    root: Node
    input: str


# =============================================================================
# 3. 配置和初始化
# =============================================================================

def setup_environment():
    """加载环境变量"""
    dotenv.load_dotenv()
    print("\n=== 环境变量已加载 ===")


def create_llm():
    """
    创建 LLM 实例
    
    输出: ChatFireworks - 配置好的 LLM 实例
    """
    print("\n=== 初始化 LLM ===")

    return ChatOpenAI(
        model="gpt-4o",
        max_tokens=16384,
        streaming=True,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def create_tools():
    """
    创建工具列表
    
    输出: tuple(list, ToolNode) - (工具列表, 工具节点)
    """
    print("\n=== 初始化工具 ===")
    search = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)
    tools = [tavily_tool]
    tool_node = ToolNode(tools=tools)
    return tools, tool_node


def create_chains(llm, tools):
    """
    创建各种处理链
    
    输入:
        llm: LLM 实例
        tools: 工具列表
    
    输出: dict - 包含所有链的字典
    """
    print("\n=== 初始化处理链 ===")
    
    # 反思提示模板
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", "Reflect and grade the assistant response to the user question below."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="candidate"),
    ])
    
    # 反思链
    reflection_llm_chain = (
        reflection_prompt
        | llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(
            run_name="Reflection"
        )
        | PydanticToolsParser(tools=[Reflection])
    )
    
    # 通用提示模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ])
    
    # 初始答案链
    initial_answer_chain = prompt_template | llm.bind_tools(tools=tools).with_config(
        run_name="GenerateInitialCandidate"
    )
    
    # JSON 解析器
    parser = JsonOutputToolsParser(return_id=True)
    
    return {
        "reflection_llm_chain": reflection_llm_chain,
        "prompt_template": prompt_template,
        "initial_answer_chain": initial_answer_chain,
        "parser": parser,
    }


# =============================================================================
# 4. LATS 核心函数
# =============================================================================

def create_reflection_chain(reflection_llm_chain):
    """
    创建反思链函数
    
    输入:
        reflection_llm_chain: 反思 LLM 链
    
    输出: Callable - 反思链函数
    """
    @as_runnable
    def reflection_chain(inputs) -> Reflection:
        """
        反思链：评估候选答案的质量
        
        输入:
            inputs (dict): 包含以下键值:
                - "input" (str): 原始用户问题
                - "candidate" (list[BaseMessage]): 候选答案的消息列表
        
        输出: Reflection - 包含评分、反思内容和是否解决的评估结果
        """
        print("\n=== 运行反思链 (Reflection Chain) ===")
        print(f"  输入问题: {inputs.get('input', '')[:80]}...")
        print(f"  候选答案消息数: {len(inputs.get('candidate', []))}")
        
        tool_choices = reflection_llm_chain.invoke(inputs)
        print(f"\n  [Reflection] LLM 原始响应:")
        print(f"  {tool_choices}")
        
        reflection = tool_choices[0]
        
        print(f"\n  反思结果:")
        print(f"    - 评分: {reflection.score}/10")
        print(f"    - 是否解决: {reflection.found_solution}")
        print(f"    - 反思摘要: {reflection.reflections[:100]}...")
        
        if not isinstance(inputs["candidate"][-1], AIMessage):
            reflection.found_solution = False
            print("  警告: 最后一条消息不是 AIMessage，强制设置 found_solution=False")
        
        return reflection
    
    return reflection_chain


def create_generate_candidates(llm, tools):
    """
    创建候选生成函数
    
    输入:
        llm: LLM 实例
        tools: 工具列表
    
    输出: Callable - 候选生成函数
    """
    def generate_candidates(messages: ChatPromptValue, config: RunnableConfig):
        """
        生成多个候选答案
        
        输入:
            messages (ChatPromptValue): 包含对话历史的提示值
            config (RunnableConfig): 运行配置
        
        输出: list[AIMessage] - N 个候选答案消息
        """
        n = config["configurable"].get("N", 5)
        print(f"\n  [generate_candidates] 生成 {n} 个候选答案...")
        
        bound_kwargs = llm.bind_tools(tools=tools).kwargs
        chat_result = llm.generate(
            [messages.to_messages()],
            n=n,
            callbacks=config["callbacks"],
            run_name="GenerateCandidates",
            **bound_kwargs,
        )
        
        candidates = [gen.message for gen in chat_result.generations[0]]
        print(f"  [generate_candidates] 成功生成 {len(candidates)} 个候选答案")
        
        # 打印每个候选答案的详细内容
        for idx, candidate in enumerate(candidates):
            print(f"\n  --- 候选答案 {idx + 1} ---")
            print(f"  内容: {str(candidate.content)[:200]}..." if len(str(candidate.content)) > 200 else f"  内容: {candidate.content}")
            if hasattr(candidate, 'tool_calls') and candidate.tool_calls:
                print(f"  工具调用数: {len(candidate.tool_calls)}")
                for tc_idx, tc in enumerate(candidate.tool_calls):
                    print(f"    工具 {tc_idx + 1}: {tc.get('name', 'unknown')}")
                    print(f"    参数: {tc.get('args', {})}")
        
        return candidates
    
    return generate_candidates


def select(root: Node) -> Node:
    """
    选择阶段：从根节点选择一个叶子节点进行展开
    
    输入:
        root (Node): 搜索树的根节点
    
    输出: Node - 被选中的叶子节点
    
    算法:
        从根节点开始，在每一层选择 UCB 分数最高的子节点，
        直到到达叶子节点。
    
    用途: MCTS 的 Select 阶段，决定下一步在哪个节点展开
    """
    print("\n  [Select] 选择要展开的节点...")
    
    if not root.children:
        print(f"  [Select] 根节点无子节点，直接返回根节点")
        return root

    node = root
    path = [f"根(深度{node.depth})"]
    
    while node.children:
        max_child = max(node.children, key=lambda child: child.upper_confidence_bound())
        path.append(f"节点(深度{max_child.depth}, UCB={max_child.upper_confidence_bound():.3f})")
        node = max_child

    print(f"  [Select] 选择路径: {' -> '.join(path)}")
    print(f"  [Select] 选中节点: 深度={node.depth}, 价值={node.value:.3f}, 访问={node.visits}")
    return node


def create_generate_initial_response(initial_answer_chain, parser, tool_node, reflection_chain):
    """
    创建初始响应生成函数
    
    输入:
        initial_answer_chain: 初始答案链
        parser: JSON 解析器
        tool_node: 工具节点
        reflection_chain: 反思链
    
    输出: Callable - 初始响应生成函数
    """
    def generate_initial_response(state: TreeState) -> dict:
        """
        生成初始候选响应（LATS 的第一步）
        
        输入:
            state (TreeState): 当前状态
        
        输出: dict - 更新后的状态
        """
        print("\n" + "="*60)
        print("=== 步骤1: 生成初始响应 (generate_initial_response) ===")
        print("="*60)
        print(f"  用户问题: {state['input']}")
        
        print("\n  [1.1] 调用 LLM 生成初始答案...")
        res = initial_answer_chain.invoke({"input": state["input"]})
        print("\n  [1.1] LLM 初始响应:")
        print(f"  ├── 类型: {type(res).__name__}")
        print(f"  ├── 内容: {str(res.content)[:300]}..." if len(str(res.content)) > 300 else f"  ├── 内容: {res.content}")
        if hasattr(res, 'tool_calls') and res.tool_calls:
            print(f"  └── 工具调用: {len(res.tool_calls)} 个")
            for i, tc in enumerate(res.tool_calls):
                print(f"      [{i+1}] 工具名: {tc.get('name', 'unknown')}")
                print(f"          参数: {tc.get('args', {})}")
        else:
            print(f"  └── 工具调用: 无")
        
        print("\n  [1.2] 解析工具调用...")
        parsed = parser.invoke(res)
        print(f"  解析到 {len(parsed)} 个工具调用")
        print("\n  [1.2] 解析结果: \n", parsed)
        for i, p in enumerate(parsed):
            print(f"    工具调用 {i+1}: {p.get('type', 'unknown')}")
        
        print("\n  [1.3] 执行工具调用...")
        tool_responses = []
        for r_idx, r in enumerate(parsed):
            print(f"\n  --- 执行工具调用 {r_idx + 1}/{len(parsed)} ---")
            print(f"  工具名: {r['type']}")
            print(f"  参数: {r['args']}")
            print(f"  调用ID: {r['id']}")
            
            response = tool_node.invoke({
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": r["type"], "args": r["args"], "id": r["id"]}],
                    )
                ]
            })
            tool_responses.append(response)
            
            # 打印工具响应结果
            if response.get("messages"):
                tool_msg = response["messages"][0]
                content = str(tool_msg.content) if hasattr(tool_msg, 'content') else str(tool_msg)
                print(f"  响应内容: {content[:500]}..." if len(content) > 500 else f"  响应内容: {content}")
        
        print(f"\n  工具执行完成，获得 {len(tool_responses)} 个响应")
        
        output_messages = [res] + [tr["messages"][0] for tr in tool_responses]
        print(f"  输出消息总数: {len(output_messages)}")
        print("\n  [1.3] 输出消息: \n", output_messages)
        
        
        print("\n  [1.4] 对初始答案进行反思评估...")
        reflection = reflection_chain.invoke(
            {"input": state["input"], "candidate": output_messages}
        )
        print("\n  [1.4] 反思评估结果: \n", reflection)
        
        print("\n  [1.5] 创建根节点...")
        root = Node(output_messages, reflection=reflection)
        print(f"  根节点创建完成: {root}")
        
        return {**state, "root": root}
    
    return generate_initial_response


def create_expand(expansion_chain, parser, tool_node, reflection_chain):
    """
    创建展开函数
    
    输入:
        expansion_chain: 展开链
        parser: JSON 解析器
        tool_node: 工具节点
        reflection_chain: 反思链
    
    输出: Callable - 展开函数
    """
    def expand(state: TreeState, config: RunnableConfig) -> dict:
        """
        展开阶段：从最佳节点生成 N 个候选下一步
        
        输入:
            state (TreeState): 当前状态
            config (RunnableConfig): 运行配置
        
        输出: TreeState - 更新后的状态
        """
        print("\n" + "="*60)
        print("=== 步骤2: 展开节点 (Expand) ===")
        print("="*60)
        
        root = state["root"]
        print(f"  当前树高度: {root.height}, 是否已解决: {root.is_solved}")
        
        print("\n  [2.1] Select - 选择最佳节点...")
        best_candidate: Node = select(root)
        
        print("\n  [2.2] 获取到选中节点的轨迹...")
        messages = best_candidate.get_trajectory()
        
        print("\n  [2.3] 生成候选答案...")
        new_candidates = expansion_chain.invoke(
            {"input": state["input"], "messages": messages}, config
        )
        print(f"  生成了 {len(new_candidates)} 个候选答案")

        print("候选答案:\n {}".format(new_candidates))

        
        # 打印每个候选答案的详细内容
        for idx, candidate in enumerate(new_candidates):
            print(f"\n  --- 展开候选 {idx + 1} ---")
            content_str = str(candidate.content) if hasattr(candidate, 'content') else str(candidate)
            print(f"  内容: {content_str[:200]}..." if len(content_str) > 200 else f"  内容: {content_str}")
            if hasattr(candidate, 'tool_calls') and candidate.tool_calls:
                print(f"  工具调用数: {len(candidate.tool_calls)}")
                for tc_idx, tc in enumerate(candidate.tool_calls):
                    print(f"    工具 {tc_idx + 1}: {tc.get('name', 'unknown')}")
                    print(f"    参数: {tc.get('args', {})}")
        
        print("\n  [2.4] 解析并执行工具调用...")
        parsed = parser.batch(new_candidates)
        flattened = [
            (i, tool_call)
            for i, tool_calls in enumerate(parsed)
            for tool_call in tool_calls
        ]
        print(f"  共有 {len(flattened)} 个工具调用需要执行")
        
        tool_responses = []
        for tc_idx, (i, tool_call) in enumerate(flattened):
            print(f"\n  --- 展开工具调用 {tc_idx + 1}/{len(flattened)} (候选{i + 1}) ---")
            print(f"  工具名: {tool_call['type']}")
            print(f"  参数: {tool_call['args']}")
            
            response = tool_node.invoke({
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{
                            "name": tool_call["type"],
                            "args": tool_call["args"],
                            "id": tool_call["id"],
                        }],
                    )
                ]
            })
            tool_responses.append((i, response))
            
            # 打印工具响应
            if response.get("messages"):
                tool_msg = response["messages"][0]
                content = str(tool_msg.content) if hasattr(tool_msg, 'content') else str(tool_msg)
                print(f"  响应: {content[:300]}..." if len(content) > 300 else f"  响应: {content}")
        
        collected_responses = defaultdict(list)
        for i, resp in tool_responses:
            collected_responses[i].append(resp["messages"][0])
        
        output_messages = [
            [candidate] + collected_responses[i]
            for i, candidate in enumerate(new_candidates)
        ]
        print(f"\n  工具调用执行完成，共处理 {len(tool_responses)} 个")

        print("\n  [2.5] Evaluate - 对每个候选进行反思评估...")
        reflections = reflection_chain.batch(
            [{"input": state["input"], "candidate": msges} for msges in output_messages],
            config,
        )
        print(f"\n  反思评估完成，评分分布: {[r.score for r in reflections]}")
        for ref_idx, ref in enumerate(reflections):
            print(f"\n  --- 候选 {ref_idx + 1} 反思结果 ---")
            print(f"  评分: {ref.score}/10")
            print(f"  是否解决: {ref.found_solution}")
            print(f"  反思: {ref.reflections[:150]}..." if len(ref.reflections) > 150 else f"  反思: {ref.reflections}")
        
        print("\n  [2.6] 创建子节点并扩展树...")
        child_nodes = [
            Node(cand, parent=best_candidate, reflection=reflection)
            for cand, reflection in zip(output_messages, reflections)
        ]
        best_candidate.children.extend(child_nodes)
        print(f"  添加了 {len(child_nodes)} 个子节点")
        print(f"  更新后树高度: {root.height}")
        
        return state
    
    return expand


def should_loop(state: TreeState) -> str:
    """
    决定是否继续搜索
    
    输入:
        state (TreeState): 当前状态
    
    输出: str - 下一个节点名称
        - END: 结束搜索
        - "expand": 继续展开
    
    终止条件:
        1. 已找到解决方案 (is_solved)
        2. 树高度超过 5（防止无限搜索）
    """
    root = state["root"]
    print(f"\n  [should_loop] 检查是否继续搜索...")
    print(f"    - 是否已解决: {root.is_solved}")
    print(f"    - 当前树高度: {root.height}")
    
    if root.is_solved:
        print(f"  >>> 决定: 已找到解决方案，结束搜索 <<<")
        return END
    if root.height > 5:
        print(f"  >>> 决定: 达到最大深度，结束搜索 <<<")
        return END
    
    print(f"  >>> 决定: 继续展开 <<<")
    return "expand"


# =============================================================================
# 5. 图构建
# =============================================================================

def build_graph(generate_initial_response_fn, expand_fn):
    """
    构建 LATS 状态图
    
    输入:
        generate_initial_response_fn: 初始响应生成函数
        expand_fn: 展开函数
    
    输出: CompiledGraph - 编译后的状态图
    """
    print("\n=== 构建状态图 ===")
    
    builder = StateGraph(TreeState)
    
    # 添加节点
    builder.add_node("start", generate_initial_response_fn)
    builder.add_node("expand", expand_fn)
    
    # 添加边
    builder.add_edge(START, "start")
    builder.add_conditional_edges("start", should_loop, ["expand", END])
    builder.add_conditional_edges("expand", should_loop, ["expand", END])
    
    graph = builder.compile()
    print("  状态图构建完成")
    
    return graph


# =============================================================================
# 6. 主程序
# =============================================================================

def run_lats(graph, question: str) -> str:
    """
    运行 LATS 搜索
    
    输入:
        graph: 编译后的状态图
        question (str): 用户问题
    
    输出: str - 最终答案内容
    """
    print("\n" + "#"*60)
    print(f"# 开始 LATS 搜索")
    print("#"*60)
    print(f"\n问题: {question}\n")
    
    last_step = None
    step_count = 0
    
    for step in graph.stream({"input": question}):
        last_step = step
        step_count += 1
        step_name, step_state = next(iter(step.items()))
        print(f"\n{'='*40}")
        print(f"搜索步骤 {step_count}: {step_name}")
        print(f"当前树高度: {step_state['root'].height}")
        print(f"是否已解决: {step_state['root'].is_solved}")
        print(f"{'='*40}")
    
    print("\n" + "#"*60)
    print("# 获取最佳解决方案")
    print("#"*60)
    
    # 确定从哪个状态获取结果
    if "expand" in last_step:
        final_root = last_step["expand"]["root"]
    else:
        final_root = last_step["start"]["root"]
    
    solution_node = final_root.get_best_solution()
    best_trajectory = solution_node.get_trajectory(include_reflections=False)
    
    final_answer = best_trajectory[-1].content
    
    print("\n" + "-"*40)
    print("最终答案:")
    print("-"*40)
    print(final_answer)
    
    return final_answer


def main():
    """主程序入口"""
    print("\n" + "="*60)
    print("LATS (Language Agent Tree Search) 启动")
    print("="*60)
    
    # 1. 初始化环境
    setup_environment()
    
    # 2. 创建组件
    llm = create_llm()
    tools, tool_node = create_tools()
    chains = create_chains(llm, tools)
    
    # 3. 创建函数
    reflection_chain = create_reflection_chain(chains["reflection_llm_chain"])
    generate_candidates = create_generate_candidates(llm, tools)
    expansion_chain = chains["prompt_template"] | generate_candidates
    
    generate_initial_response_fn = create_generate_initial_response(
        chains["initial_answer_chain"],
        chains["parser"],
        tool_node,
        reflection_chain
    )
    
    expand_fn = create_expand(
        expansion_chain,
        chains["parser"],
        tool_node,
        reflection_chain
    )
    
    # 4. 构建图
    graph = build_graph(generate_initial_response_fn, expand_fn)
    
    # 5. 运行测试
    questions = [
        # "Generate a table with the average size and weight, as well as the oldest recorded instance for each of the top 5 most common birds.",
        "Write out magnus carlson series of moves in his game against Alireza Firouzja and propose an alternate strategy",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n\n{'#'*60}")
        print(f"# 测试问题 {i}")
        print(f"{'#'*60}")
        run_lats(graph, question)


if __name__ == "__main__":
    main()
