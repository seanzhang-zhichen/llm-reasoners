"""
OpenAI API 调用模块
提供与 GPT 模型交互的接口，包括请求重试、token 统计和成本计算
"""
import os
import openai
import backoff
import logging

logger = logging.getLogger(__name__)

# 全局变量：累计的 token 使用量
completion_tokens = prompt_tokens = 0

# ========== API 配置 ==========
# 尝试加载 .env 文件
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

# 从环境变量读取 API Key
api_key = os.getenv("OPENAI_API_KEY", "").strip()
if api_key:
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")

# 从环境变量读取 API Base
api_base = os.getenv("OPENAI_API_BASE", "").strip()
if api_base:
    openai.api_base = api_base
else:
    api_base = None


@backoff.on_exception(backoff.expo, openai.error.OpenAIError, max_tries=5)
def completions_with_backoff(**kwargs):
    """带指数退避重试的 API 调用函数"""
    # 确保每次调用都使用最新的 api_key 和 api_base
    # 对于旧版本的 openai 库，需要显式传递这些参数
    # 注意：如果 kwargs 中已有这些参数，会被覆盖为全局设置的值
    if api_key:
        kwargs['api_key'] = api_key
    if api_base:
        kwargs['api_base'] = api_base
    
    try:
        return openai.ChatCompletion.create(**kwargs)
    except openai.error.AuthenticationError as e:
        # 认证错误不应该重试，直接抛出并提供更友好的错误信息
        print("\n" + "="*60)
        print("❌ API 认证失败！")
        print("="*60)
        print(f"错误详情: {str(e)}")
        print(f"API Base: {api_base if api_base else '默认 (https://api.openai.com/v1)'}")
        print(f"API Key: {api_key[:7]}...{api_key[-4:] if len(api_key) > 11 else '***'}")
        print("\n可能的原因：")
        print("1. API key 无效或已过期")
        print("2. 代理 API 端点配置不正确")
        print("3. API key 格式不符合代理 API 的要求")
        print("\n解决方案：")
        print("1. 检查环境变量 OPENAI_API_KEY 是否正确设置")
        print("2. 确认 API key 是否有效")
        print("3. 如果使用代理，检查 OPENAI_API_BASE 是否正确配置")
        print("4. 确保 API key 没有多余的空格或换行符")
        print("5. 检查代理 API 是否需要特殊的认证格式")
        print("="*60 + "\n")
        raise


def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    """使用 GPT 模型生成文本（简化接口）"""
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    

def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    """使用 ChatGPT API 生成文本（完整接口）"""
    global completion_tokens, prompt_tokens
    outputs = []
    
    # 记录请求信息
    prompt_length = len(messages[0]['content']) if messages else 0
    logger.debug(f"API 调用: model={model}, n={n}, prompt_length={prompt_length}")
    
    # 如果 n > 20，需要分批调用（OpenAI API 限制每次最多 20 个样本）
    batch_num = 0
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        batch_num += 1
        
        logger.debug(f"批次 {batch_num}: 请求 {cnt} 个样本")
        
        # 调用 API（带自动重试）
        res = completions_with_backoff(
            model=model, 
            messages=messages, 
            temperature=temperature, 
            max_tokens=max_tokens, 
            n=cnt, 
            stop=stop
        )
        
        # 提取生成的文本内容
        batch_outputs = [choice.message.content for choice in res.choices]
        outputs.extend(batch_outputs)
        
        # 打印批次输出
        for i, output in enumerate(batch_outputs):
            output_idx = len(outputs) - len(batch_outputs) + i
            logger.info(f"  输出 {output_idx+1}: {output[:200]}..." if len(output) > 200 else f"  输出 {output_idx+1}: {output}")
        
        # 累计 token 使用量
        batch_completion = res.usage.completion_tokens
        batch_prompt = res.usage.prompt_tokens
        completion_tokens += batch_completion
        prompt_tokens += batch_prompt
        
        logger.debug(f"批次 {batch_num} 完成: prompt_tokens={batch_prompt}, completion_tokens={batch_completion}")
    
    logger.info(f"API 调用完成: 生成 {len(outputs)} 个结果, 总 tokens={prompt_tokens + completion_tokens}")
    return outputs
    

def gpt_usage(backend="gpt-4"):
    """计算累计的 API 使用情况和成本"""
    global completion_tokens, prompt_tokens
    
    # 根据模型类型计算成本（美元）
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.00250 + prompt_tokens / 1000 * 0.01
    else:
        cost = 0
    
    return {
        "completion_tokens": completion_tokens, 
        "prompt_tokens": prompt_tokens, 
        "cost": cost
    }

