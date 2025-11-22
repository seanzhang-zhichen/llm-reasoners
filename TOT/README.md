# Game24 Tree of Thoughts 独立版本

这是 Game24 任务的独立版本，包含所有必要的代码，可以直接运行验证 Tree of Thoughts (ToT) 方法的效果。

## 文件结构

```
game24_standalone/
├── main.py              # 入口文件
├── game24_task.py       # Game24 任务定义
├── prompts.py           # 提示词模板
├── bfs_method.py        # BFS 树搜索方法
├── models.py            # OpenAI API 调用
├── base.py              # 基础任务类
├── requirements.txt     # 依赖包
├── README.md           # 说明文档
└── data/
    └── 24.csv          # 题目数据
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 设置 API Key

在运行前，需要设置 OpenAI API Key：

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY='your-api-key-here'
```

**Windows CMD:**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## 使用方法

### 1. 朴素方法（Baseline）

不使用树搜索，直接生成多个候选：

```bash
python main.py --naive_run --prompt_sample cot --n_generate_sample 10 --task_start_index 0 --task_end_index 10
```

### 2. 树搜索方法（Tree of Thoughts）

使用 propose + value + greedy 方法：

```bash
python main.py --method_generate propose --method_evaluate value --method_select greedy --n_generate_sample 1 --n_evaluate_sample 3 --n_select_sample 5 --task_start_index 0 --task_end_index 10
```

### 3. 使用 sample 方法

```bash
python main.py --method_generate sample --method_evaluate value --method_select greedy --prompt_sample cot --n_generate_sample 2 --n_evaluate_sample 1 --n_select_sample 3 --task_start_index 0 --task_end_index 10
```

## 参数说明

- `--backend`: 使用的模型（gpt-4, gpt-3.5-turbo, gpt-4o）
- `--temperature`: 温度参数（0-1，默认 0.7）
- `--task_start_index`: 起始题目索引（默认 0）
- `--task_end_index`: 结束题目索引（默认 10）
- `--naive_run`: 使用朴素方法
- `--prompt_sample`: 提示词类型（standard 或 cot）
- `--method_generate`: 生成方法（sample 或 propose）
- `--method_evaluate`: 评估方法（value 或 vote）
- `--method_select`: 选择方法（greedy 或 sample）
- `--n_generate_sample`: 每个候选生成的样本数
- `--n_evaluate_sample`: 每个候选评估的样本数
- `--n_select_sample`: 每步选择的候选数


## 输出

程序会在 `./logs/` 目录下生成 JSON 格式的日志文件，包含：
- 每题的输入和输出
- 验证结果
- API 使用情况

## 示例输出

```
0 sum(accs) 1.0 cnt_avg 1.0 cnt_any 1
1 sum(accs) 1.0 cnt_avg 1.0 cnt_any 2
...
1.0 1.0  # 平均准确率 100%，所有题目都有正确答案
usage_so_far {'completion_tokens': 5000, 'prompt_tokens': 3000, 'cost': 0.39}
```

