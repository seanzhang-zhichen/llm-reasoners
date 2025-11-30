import math
import re
from typing import List, Optional

import numexpr
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

_MATH_DESCRIPTION = (
    "math(problem: str, context: Optional[list[str]]) -> float:\n"
    " - 解决提供的数学问题。\n"
    ' - `problem` 可以是简单的数学问题（例如 "1 + 3"）或文字问题（例如 "如果有3个苹果和2个苹果，总共有多少个苹果"）。\n'
    " - 你不能在一次调用中计算多个表达式。例如，`math('1 + 3, 2 + 4')` 不起作用。"
    "如果你需要计算多个表达式，你需要分别调用它们，比如先调用 `math('1 + 3')`，然后调用 `math('2 + 4')`\n"
    " - 尽可能减少 `math` 操作的次数。例如，不要先调用 "
    '2. math("$1 的 10% 是多少") 然后再调用 3. math("$1 + $2")，'
    '你必须改为调用 2. math("$1 的 110% 是多少")，这样可以减少数学操作的次数。\n'
    # 上下文特定规则如下
    " - 你可以选择性地提供一个字符串列表作为 `context` 来帮助代理解决问题。"
    "如果需要多个上下文来回答问题，你可以将它们作为字符串列表提供。\n"
    " - `math` 操作不会看到之前操作的输出，除非你将其作为 `context` 提供。"
    "如果你需要对其执行数学运算，你必须将之前操作的输出作为 `context` 提供。\n"
    " - 你绝对不能将 `search` 类型操作的输出作为 `problem` 参数中的变量提供。"
    "这是因为 `search` 返回一个包含实体信息的文本块，而不是数字或值。"
    "因此，当你需要提供 `search` 操作的输出时，你必须将其作为 `math` 操作的 `context` 参数提供。"
    '例如，1. search("Barack Obama") 然后 2. math("$1 的年龄") 是绝对不允许的。'
    '应该使用 2. math("Barack Obama 的年龄", context=["$1"]) 代替。\n'
    " - 当你询问关于 `context` 的问题时，请指定单位。"
    '例如，"xx 的身高是多少？" 或 "xx 是多少百万？" 而不是 "xx 是多少？"\n'
)


_SYSTEM_PROMPT = """将数学问题转换为可以使用 Python 的 numexpr 库执行的表达式。使用运行此代码的输出回答问题。

问题：${{包含数学问题的问题。}}
```text
${{解决问题的单行数学表达式}}
```
...numexpr.evaluate(text)...
```output
${{运行代码的输出}}
```
答案：${{答案}}

开始。

问题：37593 * 67 是多少？
ExecuteCode({{code: "37593 * 67"}})
...numexpr.evaluate("37593 * 67")...
```output
2518731
```
答案：2518731

问题：37593^(1/5)
ExecuteCode({{code: "37593**(1/5)"}})
...numexpr.evaluate("37593**(1/5)")...
```output
8.222831614237718
```
答案：8.222831614237718
"""

_ADDITIONAL_CONTEXT_PROMPT = """以下是从其他函数提供的额外上下文。\
    使用它来替换问题中的任何 ${{#}} 变量或其他词语。\
    \n\n${context}\n\n请注意，上下文变量尚未在代码中定义。\
你必须提取相关数字并直接将其放入代码中。"""


class ExecuteCode(BaseModel):
    """numexpr.evaluate() 函数的输入。"""

    reasoning: str = Field(
        ...,
        description="代码表达式背后的推理，包括如何包含上下文（如果适用）。",
    )

    code: str = Field(
        ...,
        description="要通过 numexpr.evaluate() 执行的简单代码表达式。",
    )


def _evaluate_expression(expression: str) -> str:
    """
    使用numexpr库安全地评估数学表达式。
    
    该函数使用numexpr.evaluate()来执行数学表达式，提供了安全的执行环境：
    - 限制对全局变量的访问
    - 提供常用数学常量（pi, e）
    - 处理执行异常
    - 清理输出格式（移除方括号）
    
    参数:
        expression (str): 要评估的数学表达式字符串，例如 "2 + 3", "sqrt(16)", "pi * 2"
        
    返回:
        str: 表达式计算结果字符串
        
    异常:
        ValueError: 当表达式无法评估时抛出，包含错误详情
        
    示例:
        输入:
            expression = "2 + 3"
        输出:
            "5"
            
        输入:
            expression = "sqrt(16)"
        输出:
            "4.0"
            
        输入:
            expression = "pi * 2"
        输出:
            "6.283185307179586"
            
        输入:
            expression = "37593 * 67"
        输出:
            "2518731"
            
        输入:
            expression = "37593**(1/5)"
        输出:
            "8.222831614237718"
            
        输入（错误情况）:
            expression = "invalid expression"
        抛出:
            ValueError: 评估 "invalid expression" 失败。引发错误：...
    """
    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # 限制对全局变量的访问
                local_dict=local_dict,  # 添加常用数学函数
            )
        )
    except Exception as e:
        raise ValueError(
            f'评估 "{expression}" 失败。引发错误：{repr(e)}。'
            " 请使用有效的数值表达式重试"
        )

    # 从输出中删除任何前导和尾随括号
    return re.sub(r"^\[|\]$", "", output)


def get_math_tool(llm: ChatOpenAI):
    """
    创建并返回一个数学计算工具，该工具可以将自然语言数学问题转换为可执行的代码表达式。
    
    该函数构建一个StructuredTool，它能够：
    1. 接受自然语言数学问题或简单数学表达式
    2. 使用LLM将问题转换为Python numexpr可执行的代码
    3. 执行代码并返回结果
    4. 支持上下文信息来帮助解决问题
    
    参数:
        llm (ChatOpenAI): 语言模型实例，用于将数学问题转换为代码表达式
        
    返回:
        StructuredTool: 配置好的数学工具，可以像函数一样调用
        
    工具签名:
        math(problem: str, context: Optional[List[str]]) -> str
        
    示例:
        输入:
            llm = ChatFireworks(model="...")
            math_tool = get_math_tool(llm)
            
            # 使用工具
            result = math_tool.invoke({
                "problem": "2 + 3",
                "context": None
            })
        输出:
            "5"
            
        输入:
            result = math_tool.invoke({
                "problem": "旧金山的温度加5是多少？",
                "context": ["旧金山的温度是32度"]
            })
        输出:
            "37"
            
        输入:
            result = math_tool.invoke({
                "problem": "37593 * 67",
                "context": None
            })
        输出:
            "2518731"
            
        输入:
            result = math_tool.invoke({
                "problem": "计算圆的面积，半径为5",
                "context": None
            })
        输出:
            "78.53981633974483"  # pi * 5^2
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{problem}"),
            MessagesPlaceholder(variable_name="context", optional=True),
        ]
    )
    extractor = prompt | llm.with_structured_output(
        ExecuteCode, method="function_calling"
    )

    def calculate_expression(
        problem: str,
        context: Optional[List[str]] = None,
        config: Optional[RunnableConfig] = None,
    ):
        """
        计算数学表达式或解决数学问题。
        
        这是数学工具的核心函数，它：
        1. 将问题（可能包含上下文）传递给LLM
        2. LLM生成可执行的代码表达式
        3. 执行表达式并返回结果
        
        参数:
            problem (str): 数学问题或表达式，可以是：
                - 简单表达式："2 + 3"
                - 自然语言问题："如果有3个苹果和2个苹果，总共有多少个苹果"
                - 需要上下文的问题："$1 加 5 是多少"（需要context提供$1的值）
            context (Optional[List[str]]): 可选的上下文信息列表，用于提供：
                - 之前任务的结果（通过$1, $2等引用）
                - 额外的背景信息
            config (Optional[RunnableConfig]): 可选的运行配置
            
        返回:
            str: 计算结果字符串，如果执行失败则返回错误信息的字符串表示
            
        示例:
            输入:
                problem = "2 + 3"
                context = None
                config = None
            输出:
                "5"
                
            输入:
                problem = "旧金山的温度加5是多少？"
                context = ["旧金山的温度是32度"]
                config = None
            输出:
                "37"
                
            输入:
                problem = "$1 的三次方是多少？"
                context = ["32"]  # $1的值
                config = None
            输出:
                "32768"
                
            输入（错误情况）:
                problem = "无效表达式"
                context = None
                config = None
            输出:
                "ValueError('评估 \"无效表达式\" 失败。引发错误：...')"
        """
        chain_input = {"problem": problem}
        if context:
            context_str = "\n".join(context)
            if context_str.strip():
                context_str = _ADDITIONAL_CONTEXT_PROMPT.format(
                    context=context_str.strip()
                )
                chain_input["context"] = [SystemMessage(content=context_str)]
        code_model = extractor.invoke(chain_input, config)
        try:
            return _evaluate_expression(code_model.code)
        except Exception as e:
            return repr(e)

    return StructuredTool.from_function(
        name="math",
        func=calculate_expression,
        description=_MATH_DESCRIPTION,
    )