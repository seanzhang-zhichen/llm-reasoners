"""
广度优先搜索（BFS）方法实现
使用树搜索（Tree of Thoughts）方法解决问题，通过生成、评估、选择三个步骤逐步构建解决方案
"""
import itertools
import numpy as np
import logging
from functools import partial
from models import gpt

logger = logging.getLogger(__name__)


def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    """评估单个候选解决方案的价值分数"""
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        logger.debug(f"使用缓存的价值分数: {task.value_cache[value_prompt]}")
        return task.value_cache[value_prompt]
    
    logger.debug(f"评估候选: {y[:100]}..." if len(y) > 100 else f"评估候选: {y}")
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    
    # 打印评估输出
    for i, output in enumerate(value_outputs):
        logger.info(f"评估输出 {i+1}/{len(value_outputs)}: {output.strip()}")
    
    value = task.value_outputs_unwrap(x, y, value_outputs)
    logger.info(f"评估价值分数: {value:.3f}")
    
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    """批量评估多个候选解决方案的价值分数"""
    values = []
    local_value_cache = {}
    for y in ys:
        if y in local_value_cache:
            value = 0
        else:
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values


def get_votes(task, x, ys, n_evaluate_sample):
    """使用投票方法评估多个候选解决方案"""
    vote_prompt = task.vote_prompt_wrap(x, ys)
    logger.debug(f"投票提示词长度: {len(vote_prompt)} 字符")
    logger.info(f"对 {len(ys)} 个候选进行投票评估")
    
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    
    # 打印投票输出
    for i, output in enumerate(vote_outputs):
        logger.info(f"投票输出 {i+1}/{len(vote_outputs)}: {output.strip()}")
    
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    logger.info(f"投票结果: {values}")
    return values


def get_proposals(task, x, y):
    """使用 propose 方法生成下一步的候选操作"""
    propose_prompt = task.propose_prompt_wrap(x, y)
    logger.debug(f"提议提示词长度: {len(propose_prompt)} 字符")
    
    raw_output = gpt(propose_prompt, n=1, stop=None)[0]
    logger.info(f"模型原始输出: {raw_output[:300]}..." if len(raw_output) > 300 else f"模型原始输出: {raw_output}")
    
    proposals = raw_output.split('\n')
    filtered_proposals = [y + _ + '\n' for _ in proposals if _.strip()]  # 过滤空行
    
    logger.info(f"解析出 {len(filtered_proposals)} 个提议:")
    for i, prop in enumerate(filtered_proposals[:5]):  # 只显示前5个
        logger.info(f"  提议 {i+1}: {prop.strip()}")
    if len(filtered_proposals) > 5:
        logger.info(f"  ... 还有 {len(filtered_proposals) - 5} 个提议")
    
    return filtered_proposals


def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    """使用 sample 方法生成候选解决方案"""
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    
    logger.debug(f"生成提示词长度: {len(prompt)} 字符")
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    
    # 打印生成的样本
    for i, sample in enumerate(samples):
        logger.info(f"样本 {i+1}/{len(samples)}: {sample[:200]}..." if len(sample) > 200 else f"样本 {i+1}/{len(samples)}: {sample}")
    
    return [y + _ for _ in samples]


def solve(args, task, idx, to_print=True):
    """使用树搜索（Tree of Thoughts）方法解决问题"""
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    logger.info(f"配置 GPT 模型: {args.backend}, temperature={args.temperature}")
    
    x = task.get_input(idx)
    logger.info(f"输入: {x}")
    ys = ['']
    infos = []
    
    for step in range(task.steps):
        logger.info(f"\n--- 步骤 {step + 1}/{task.steps} ---")
        logger.info(f"当前候选数: {len(ys)}")
        
        # 生成
        logger.info(f"生成阶段 (method={args.method_generate})...")
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, 
                                 prompt_sample=args.prompt_sample, 
                                 stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        logger.info(f"生成了 {len(new_ys)} 个新候选")
        
        # 打印生成的候选（前几个）
        logger.info("生成的候选内容:")
        for i, candidate in enumerate(new_ys[:5]):
            logger.info(f"  候选 {i+1}: {candidate[:150]}..." if len(candidate) > 150 else f"  候选 {i+1}: {candidate}")
        if len(new_ys) > 5:
            logger.info(f"  ... 还有 {len(new_ys) - 5} 个候选")
        
        # 评估
        logger.info(f"评估阶段 (method={args.method_evaluate}, n_evaluate_sample={args.n_evaluate_sample})...")
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
        
        logger.info(f"评估完成，价值分数: {values}")
        logger.info(f"最高价值: {max(values):.2f}, 最低价值: {min(values):.2f}, 平均价值: {sum(values)/len(values):.2f}")
        
        # 选择
        logger.info(f"选择阶段 (method={args.method_select}, n_select_sample={args.n_select_sample})...")
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]
        logger.info(f"选择了 {len(select_new_ys)} 个候选进入下一步")
        
        # 打印选中的候选详情
        logger.info("选中的候选:")
        for i, (select_id, candidate) in enumerate(zip(select_ids, select_new_ys)):
            logger.info(f"  选中 {i+1} (排名 {select_id+1}, 价值 {values[select_id]:.2f}): {candidate[:150]}..." 
                       if len(candidate) > 150 else 
                       f"  选中 {i+1} (排名 {select_id+1}, 价值 {values[select_id]:.2f}): {candidate}")
        
        # 日志
        if to_print:
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), 
                                                      key=lambda x: x[1], reverse=True))
            logger.debug(f'所有候选: {sorted_new_ys}')
            logger.debug(f'价值分数: {sorted_values}')
        
        infos.append({
            'step': step,
            'x': x,
            'ys': ys,
            'new_ys': new_ys,
            'values': values,
            'select_new_ys': select_new_ys
        })
        ys = select_new_ys
    
    logger.info(f"\n最终生成 {len(ys)} 个候选解决方案")
    logger.info("最终候选解决方案:")
    for i, final_y in enumerate(ys):
        logger.info(f"  最终候选 {i+1}: {final_y}")
    if to_print:
        print(ys)
    return ys, {'steps': infos}


def naive_solve(args, task, idx, to_print=True):
    """朴素求解方法：不使用树搜索，直接生成多个候选并返回"""
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    logger.info(f"配置 GPT 模型: {args.backend}, temperature={args.temperature}")
    
    x = task.get_input(idx)
    logger.info(f"输入: {x}")
    logger.info(f"生成 {args.n_generate_sample} 个候选 (prompt_sample={args.prompt_sample})...")
    
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    logger.info(f"生成了 {len(ys)} 个候选解决方案")
    
    # 打印所有生成的候选
    logger.info("所有生成的候选解决方案:")
    for i, candidate in enumerate(ys):
        logger.info(f"  候选 {i+1}: {candidate}")
    
    return ys, {}

