"""
Game24 独立运行入口
使用 Tree of Thoughts 方法解决 Game24 问题
"""
import os
import json
import argparse
import logging
from datetime import datetime

# 在导入其他模块之前先加载 .env 文件
try:
    import dotenv
    dotenv.load_dotenv(override=True)
    print("Loaded .env file")
except ImportError:
    # 如果没有安装 python-dotenv，跳过
    pass

from game24_task import Game24Task
from bfs_method import solve, naive_solve
from models import gpt_usage

# 配置日志
def setup_logging(log_file=None):
    """设置日志配置"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    handlers = [logging.StreamHandler()]  # 控制台输出
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    return logging.getLogger(__name__)


def run(args):
    """主运行函数"""
    # 生成日志文件名
    if args.naive_run:
        file = f'./logs/game24_{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
        log_file = f'./logs/game24_{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.log'
    else:
        file = f'./logs/game24_{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
        log_file = f'./logs/game24_{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.log'
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(file), exist_ok=True)
    
    # 设置日志
    logger = setup_logging(log_file)
    logger.info("="*60)
    logger.info("开始运行 Game24 任务")
    logger.info(f"参数: {args}")
    logger.info("="*60)
    
    # 获取任务对象
    task = Game24Task()
    logger.info(f"加载任务数据，共 {len(task)} 道题目")
    
    # 初始化统计变量
    logs = []
    cnt_avg = 0
    cnt_any = 0

    # 遍历指定范围的题目
    for i in range(args.task_start_index, args.task_end_index):
        logger.info(f"\n{'='*60}")
        logger.info(f"处理题目 {i}/{args.task_end_index - 1}")
        logger.info(f"题目: {task.get_input(i)}")
        
        try:
            # 求解
            logger.info(f"开始求解...")
            start_time = datetime.now()
            
            if args.naive_run:
                logger.info(f"使用朴素方法 (prompt_sample={args.prompt_sample}, n_generate_sample={args.n_generate_sample})")
                ys, info = naive_solve(args, task, i, to_print=False)
            else:
                logger.info(f"使用树搜索方法 (generate={args.method_generate}, evaluate={args.method_evaluate}, select={args.method_select})")
                ys, info = solve(args, task, i, to_print=False)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"求解完成，耗时 {elapsed:.2f} 秒，生成 {len(ys)} 个候选")

            # 验证和记录
            logger.info("开始验证结果...")
            infos = [task.test_output(i, y) for y in ys]
            correct_count = sum(1 for info in infos if info['r'] == 1)
            logger.info(f"验证完成: {correct_count}/{len(infos)} 个候选正确")
            
            info.update({
                'idx': i,
                'ys': ys,
                'infos': infos,
                'usage_so_far': gpt_usage(args.backend),
                'elapsed_time': elapsed
            })
            logs.append(info)
            
            # 实时保存日志
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=4, ensure_ascii=False)
            logger.info(f"结果已保存到 {file}")
            
            # 统计指标
            accs = [info['r'] for info in infos]
            cnt_avg += sum(accs) / len(accs)
            cnt_any += any(accs)
            logger.info(f"当前统计: sum(accs)={sum(accs)}, cnt_avg={cnt_avg:.2f}, cnt_any={cnt_any}")
            print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')
            
        except Exception as e:
            logger.error(f"处理题目 {i} 时出错: {str(e)}", exc_info=True)
            print(f"Error processing problem {i}: {e}")
            continue
    
    # 最终统计
    n = args.task_end_index - args.task_start_index
    final_avg = cnt_avg / n
    final_any = cnt_any / n
    usage = gpt_usage(args.backend)
    
    logger.info("="*60)
    logger.info("运行完成")
    logger.info(f"平均准确率: {final_avg:.4f}")
    logger.info(f"至少一个正确答案的比例: {final_any:.4f}")
    logger.info(f"API 使用情况: {usage}")
    logger.info("="*60)
    
    print(f"\n最终结果: 平均准确率={final_avg:.4f}, 至少一个正确答案的比例={final_any:.4f}")
    print('usage_so_far', usage)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Game24 Tree of Thoughts Solver')
    
    # 模型配置
    parser.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-4o'], default='gpt-4')
    parser.add_argument('--temperature', type=float, default=0.7)
    
    # 任务配置
    parser.add_argument('--task_start_index', type=int, default=0)
    parser.add_argument('--task_end_index', type=int, default=10)
    
    # 运行模式
    parser.add_argument('--naive_run', action='store_true', help='使用朴素方法（不使用树搜索）')
    parser.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'], help='提示词类型（仅在 naive_run 或 method_generate=sample 时使用）')
    
    # 树搜索方法配置
    parser.add_argument('--method_generate', type=str, choices=['sample', 'propose'], help='生成方法')
    parser.add_argument('--method_evaluate', type=str, choices=['value', 'vote'], help='评估方法')
    parser.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy', help='选择方法')
    
    # 采样数量配置
    parser.add_argument('--n_generate_sample', type=int, default=1, help='每个候选生成的样本数')
    parser.add_argument('--n_evaluate_sample', type=int, default=1, help='每个候选评估的样本数')
    parser.add_argument('--n_select_sample', type=int, default=1, help='每步选择的候选数')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)

