"""
基础任务类
定义所有任务需要实现的基本接口
"""
import os

# 数据文件路径
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


class Task:
    """任务基类，定义任务的基本接口"""
    def __init__(self):
        pass

    def __len__(self) -> int:
        """返回题目总数"""
        pass

    def get_input(self, idx: int) -> str:
        """获取指定索引的题目"""
        pass

    def test_output(self, idx: int, output: str):
        """验证输出是否正确"""
        pass

