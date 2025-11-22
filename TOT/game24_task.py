"""
Game24 任务实现
目标：使用 4 个数字和基本运算（+ - * /）得到 24
"""
import re
import os
import sympy
import pandas as pd
from base import Task, DATA_PATH
from prompts import *


def get_current_numbers(y: str) -> str:
    """
    从部分输出中提取当前剩余的数字
    
    例如：从 "1 + 2 = 3 (left: 3 3 4)" 中提取 "3 3 4"
    
    参数:
        y: 部分输出字符串，格式如 "1 + 2 = 3 (left: 3 3 4)"
    
    返回:
        str: 当前剩余的数字字符串，如 "3 3 4"
    """
    last_line = y.strip().split('\n')[-1]  # 获取最后一行
    return last_line.split('left: ')[-1].split(')')[0]  # 提取 "left: " 和 ")" 之间的内容


class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='24.csv'):
        """
        初始化 Game24 任务
        
        参数:
            file: CSV 文件名，包含题目数据（默认 '24.csv'）
        """
        super().__init__()
        # 读取题目数据文件
        path = os.path.join(DATA_PATH, file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}\n请确保 data/24.csv 文件存在")
        self.data = list(pd.read_csv(path)['Puzzles'])  # 将题目列表存储在 self.data 中
        
        # 价值评估缓存，避免重复评估相同的提示词
        self.value_cache = {}
        
        # 游戏需要 4 步完成（4 个数字 -> 3 个数字 -> 2 个数字 -> 1 个数字（24））
        self.steps = 4
        
        # 每步的停止标记，用于控制生成何时停止
        self.stops = ['\n'] * 4

    def __len__(self) -> int:
        """返回题目总数"""
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        """获取指定索引的题目"""
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        """
        验证输出是否正确
        
        检查两个条件：
        1. 是否使用了所有输入数字（且没有使用其他数字）
        2. 计算结果是否等于 24
        """
        # 提取最后一行中的答案表达式（去掉 "Answer: " 前缀和 "= 24" 后缀）
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        
        # 从表达式中提取所有数字
        numbers = re.findall(r'\d+', expression)
        
        # 从原始题目中提取所有数字
        problem_numbers = re.findall(r'\d+', self.data[idx])
        
        # 检查使用的数字是否与题目数字完全一致（顺序无关）
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}  # 数字不匹配，返回错误
        
        try:
            # 使用 sympy 计算表达式的值，检查是否等于 24
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # 如果表达式无法计算（语法错误等），返回错误
            return {'r': 0}
            
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        """包装标准提示词"""
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        """包装思维链（Chain of Thought）提示词"""
        return cot_prompt.format(input=x) + y
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        """包装提议提示词"""
        # 提取当前剩余的数字
        current_numbers = get_current_numbers(y if y else x)
        
        # 如果已经得到 24，说明所有步骤已完成，需要生成最终答案
        if current_numbers == '24':
            # 使用 cot 提示词，并要求在已有步骤基础上生成答案
            prompt = cot_prompt.format(input=x) + 'Steps:' + y
        else:
            # 否则，使用提议提示词，要求生成下一步可能的操作
            prompt = propose_prompt.format(input=current_numbers)
        
        return prompt
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        """包装价值评估提示词"""
        last_line = y.strip().split('\n')[-1]
        
        # 判断是否是最后一步（没有 "left: " 说明已经给出最终答案）
        if 'left: ' not in last_line:  # last step
            # 最后一步：评估答案是否正确
            ans = last_line.lower().replace('answer: ', '')
            return value_last_step_prompt.format(input=x, answer=ans)
        
        # 中间步骤：评估当前状态是否有可能达到 24
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        """将 GPT 的价值评估文本输出转换为数值分数"""
        # 如果已经完成 4 步但没有答案，说明路径错误，返回 0
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        
        # 提取每个评估结果的最后一行（通常是 "sure"、"likely" 或 "impossible"）
        value_names = [_.split('\n')[-1] for _ in value_outputs]
        
        # 定义文本到数值的映射
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        
        # 累加所有评估结果的分数
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        
        return value

